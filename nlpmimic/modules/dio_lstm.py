"""
An LSTM with Recurrent Dropout and the option to use highway
connections between layers.
"""

from typing import Optional, Tuple, List, Dict

import torch

from allennlp.nn import util
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_dropout_mask
from allennlp.nn.initializers import block_orthogonal
from allennlp.modules.similarity_functions.bilinear import BilinearSimilarity

class DioLstm(torch.nn.Module):
    """
    An LSTM with Recurrent Dropout and the option to use highway
    connections between layers. Note: this implementation is slower
    than the native Pytorch LSTM because it cannot make use of CUDNN
    optimizations for stacked RNNs due to the highway layers and
    variational dropout.

    Parameters
    ----------
    input_size : int, required.
        The dimension of the inputs to the LSTM.
    hidden_size : int, required.
        The dimension of the outputs of the LSTM.
    recurrent_dropout_probability: float, optional (default = 0.0)
        the dropout probability to be used in a dropout scheme as stated in
        `a theoretically grounded application of dropout in recurrent neural networks
        <https://arxiv.org/abs/1512.05287>`_ . implementation wise, this simply
        applies a fixed dropout mask per sequence to the recurrent connection of the
        lstm.
    use_input_projection_bias : bool, optional (default = true)
        whether or not to use a bias on the input projection layer. this is mainly here
        for backwards compatibility reasons and will be removed (and set to false)
        in future releases.

    returns
    -------
    output_accumulator : packedsequence
        the outputs of the lstm for each timestep. a tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 recurrent_dropout_probability: float = 0.0,
                 use_input_projection_bias: bool = True) -> None:
        super(DioLstm, self).__init__()
        # required to be wrapped with a :class:`pytorchseq2seqwrapper`.
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.recurrent_dropout_probability = recurrent_dropout_probability

        self.ins_input_linearity = torch.nn.Linear(input_size, 5 * hidden_size, bias=use_input_projection_bias)
        self.ins_state_linearity = torch.nn.Linear(hidden_size, 5 * hidden_size, bias=True)
       
        self.out_input_linearity = torch.nn.Linear(input_size, 5 * hidden_size, bias=use_input_projection_bias)
        self.out_state_linearity = torch.nn.Linear(hidden_size, 5 * hidden_size, bias=True)
       
        self.ins_bilinearity = BilinearSimilarity(2 * hidden_size, 2 * hidden_size)
        self.out_bilinearity = BilinearSimilarity(2 * hidden_size, 2 * hidden_size)

        self.out_bias = torch.zeros((5 * hidden_size,), dtype=torch.float)
        self.ins_bias = self.out_bias.clone()
        self.ins_bias[hidden_size:3 * hidden_size] = 1
        
        self.out_vect = torch.Tensor(2 * hidden_size, 1)

        self.reset_parameters()

    def reset_parameters(self):
        # use sensible default initializations for parameters.
        block_orthogonal(self.ins_input_linearity.weight.data, [self.hidden_size, self.input_size])
        block_orthogonal(self.ins_state_linearity.weight.data, [self.hidden_size, self.hidden_size])

        self.ins_state_linearity.bias.data.fill_(0.0)
        # initialize forget gate biases to 1.0 as per an empirical
        # exploration of recurrent network architectures, (jozefowicz, 2015).
        self.ins_state_linearity.bias.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)

        block_orthogonal(self.out_input_linearity.weight.data, [self.hidden_size, self.input_size])
        block_orthogonal(self.out_state_linearity.weight.data, [self.hidden_size, self.hidden_size])

        self.out_state_linearity.bias.data.fill_(0.0)
        # initialize forget gate biases to 1.0 as per an empirical
        # exploration of recurrent network architectures, (jozefowicz, 2015).
        self.out_state_linearity.bias.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)

        torch.nn.init.xavier_uniform_(self.out_vect)

    def idx(self, offset: int, ilayer: int) -> int:
        return ilayer * (ilayer - 1) // 2 + offset 
    
    def set_root(self, cell: Dict):
        cell['h'] = self.out_vect[0:self.hidden_size, 0]
        cell['c'] = self.out_vect[self.hidden_size:, 0]
        cell['v'] = self.out_vect[:, 0] 
        cell['s'] = torch.tensor(0.) # 1 in logrithmic form 
    
    def set_leaf(self, cell: Dict, seqt: torch.Tensor):
        if self.training:
            cell['h'] = seqt[0:self.hidden_size, 0]
            cell['c'] = seqt[self.hidden_size:, 0]
            cell['v'] = seqt[:, 0]
            cell['s'] = torch.tensor(0.) # why 1?
        else:
            cell['h_max'] = seqt[0:self.hidden_size, 0]
            cell['c_max'] = seqt[self.hidden_size:, 0]
            cell['v_max'] = seqt[:, 0]
            cell['s_max'] = torch.tensor(0.) # why 1?

    def soft_sum(self, cell: Dict):
        const = torch.tensor(1.)
        #const = torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float)) 
        if len(cell['s_set']) < 2:
            cell['h'] = cell['h_set'][0] / const
            cell['c'] = cell['c_set'][0] / const
            cell['s'] = cell['s_set'][0]
            cell['v'] = torch.cat((cell['h'], cell['c']), 0)
        else:
            span_weights = torch.tensor(cell['s_set'])
            expt_weights = span_weights - torch.max(span_weights)
            expt_weights = expt_weights.exp()
            soft_weights = expt_weights / expt_weights.sum()
            
            #print(span_weights)
             
            h_set = torch.stack(cell['h_set'], 1)
            c_set = torch.stack(cell['c_set'], 1)
            #print(h_set[1, :5])
            
            cell['h'] = h_set.mv(soft_weights) / const
            cell['c'] = c_set.mv(soft_weights) / const
            cell['s'] = soft_weights.dot(span_weights)
            cell['v'] = torch.cat((cell['h'], cell['c']), 0)
    

    def tree_lstm(self, projected_input, projected_state, lmem, rmem):
        # Main LSTM equations using relevant chunks of the big linear
        # projections of the hidden state and inputs.
        input_gate = torch.sigmoid(projected_input[0 * self.hidden_size:1 * self.hidden_size] +
                                   projected_state[0 * self.hidden_size:1 * self.hidden_size])
        lforget_gate = torch.sigmoid(projected_input[1 * self.hidden_size:2 * self.hidden_size] +
                                     projected_state[1 * self.hidden_size:2 * self.hidden_size])
        rforget_gate = torch.sigmoid(projected_input[2 * self.hidden_size:3 * self.hidden_size] +
                                     projected_state[2* self.hidden_size:3 * self.hidden_size])
        memory_init = torch.tanh(projected_input[3 * self.hidden_size:4 * self.hidden_size] +
                                 projected_state[3 * self.hidden_size:4 * self.hidden_size])
        output_gate = torch.sigmoid(projected_input[4 * self.hidden_size:5 * self.hidden_size] +
                                    projected_state[4 * self.hidden_size:5 * self.hidden_size])
        memory = lforget_gate * lmem + rforget_gate * rmem + input_gate * memory_init
        
        timestep_output = output_gate * torch.tanh(memory)
        
        return timestep_output, memory

    
    def compose(self, lcell: Dict, rcell: Dict, pcell: Dict, span: str, inside: bool):
        if not self.training:
            lv, rv = lcell['v_max'], rcell['v_max']
            ls, rs = lcell['s_max'], rcell['s_max']
            lh, rh = lcell['h_max'], rcell['h_max']
            lc, rc = lcell['c_max'], rcell['c_max']
        else:
            lv, rv = lcell['v'], rcell['v']
            ls, rs = lcell['s'], rcell['s']
            lh, rh = lcell['h'], rcell['h']
            lc, rc = lcell['c'], rcell['c']

        if inside:
            compat = self.ins_bilinearity(lv, rv) + ls + rs

            projected_input = self.ins_input_linearity(lh) + self.ins_bias
            projected_state = self.ins_state_linearity(rh)
        else: 
            compat = self.out_bilinearity(lv, rv) + ls + rs

            projected_input = self.out_input_linearity(lh) + self.out_bias
            projected_state = self.out_state_linearity(rh)

        h, c = self.tree_lstm(projected_input, projected_state, lc, rc)
        
        if 'spans' not in pcell:
            pcell['h_set'], pcell['c_set'] = [h], [c]
            pcell['s_set'], pcell['spans'] = [compat], [span]
        else:
            pcell['h_set'].append(h)
            pcell['c_set'].append(c)
            pcell['s_set'].append(compat)
            pcell['spans'].append(span)

        if not self.training:
            if 's_max' not in pcell or pcell['s_max'] < compat: 
                pcell['h_max'], pcell['c_max'] = h, c
                pcell['s_max'], pcell['spanm'] = compat, span
                pcell['v_max'] = torch.cat((h, c), 0)
        return h, c


    def extract_span(self, icell: int, chart: Dict[str, List[Dict]], spans: List):
        spanm = chart['max'][icell]['spanm']
        x0, y0, x1, y1 = list(map(int, spanm.split('.')))
        
        spans.append([x0, y0])
        spans.append([x1, y1])

        if x0 != y0:
            lcell = self.idx(x0, chart['nword'] - (y0 - x0))
            self.extract_span(lcell, chart, spans)
        if x1 != y1:
            rcell = self.idx(x1, chart['nword'] - (y1 - x1))
            self.extract_span(rcell, chart, spans)
        
    
    def inference(self, 
                  seqt: torch.Tensor,
                  chart: Dict[str, List[Dict]]):
        mchart, nword = chart['max'], chart['nword']

        for i in range(0, nword):
            icell = self.idx(i, nword)
            embed = torch.unsqueeze(seqt[0, i, :], 1)
            self.set_leaf(mchart[icell], embed)
        
        for ilayer in range(1, nword):
            for left in range(0, nword - ilayer):
                x0 = left
                y1 = left + ilayer
                c2 = self.idx(left, nword - ilayer)
                for right in range(left, left + ilayer):
                    y0 = right
                    x1 = right + 1
                    c0 = self.idx(x0, nword - (y0 - x0))
                    c1 = self.idx(x1, nword - (y1 - x1))
                    span = '{}.{}.{}.{}'.format(x0, y0, x1, y1)
                   
                    h, c = self.compose(mchart[c0], mchart[c1], mchart[c2], span, True)                 
                

    def do_inside(self,
                  seqt: torch.Tensor,
                  chart: Dict[str, List[Dict]]):
        ichart, nword = chart['ins'], chart['nword']
         
        for i in range(0, nword):
            icell = self.idx(i, nword)
            embed = torch.unsqueeze(seqt[0, i, :], 1)
            self.set_leaf(ichart[icell], embed)
            
        for ilayer in range(1, nword):
            for left in range(0, nword - ilayer):
                x0 = left
                y1 = left + ilayer
                c2 = self.idx(left, nword - ilayer)
                for right in range(left, left + ilayer):
                    y0 = right
                    x1 = right + 1
                    c0 = self.idx(x0, nword - (y0 - x0))
                    c1 = self.idx(x1, nword - (y1 - x1))
                    span = '{}.{}.{}.{}'.format(x0, y0, x1, y1)
                   
                    h, c = self.compose(ichart[c0], ichart[c1], ichart[c2], span, True)                 
                
                self.soft_sum(ichart[c2])

    
    def do_outside(self, chart: Dict[str, List[Dict]]):
        ichart, ochart, nword = chart['ins'], chart['out'], chart['nword'] 

        for ilayer in range(nword - 1, -1, -1):
            for left in range(0, nword - ilayer):
                x0 = left
                x1 = left + ilayer + 1
                c2 = self.idx(left, nword - ilayer)
                for right in range(left + ilayer + 1, nword):
                    y0 = right
                    y1 = right
                    c0 = self.idx(x0, nword - (y0 - x0))
                    c1 = self.idx(x1, nword - (y1 - x1))
                    span = '{}.{}.{}.{}'.format(x0, y0, x1, y1) # x0, y0; x1

                    h, c = self.compose(ochart[c0], ichart[c1], ochart[c2], span, False)

                y0 = left + ilayer
                y1 = left - 1
                for right in range(0, left):
                    x0 = right
                    x1 = right
                    c0 = self.idx(x0, nword - (y0 - x0))
                    c1 = self.idx(x1, nword - (y1 - x1))
                    span = '{}.{}.{}.{}'.format(x0, y0, x1, y1) # x0, y0, y1
                    
                    h, c = self.compose(ochart[c0], ichart[c1], ochart[c2], span, False)
                
                if c2 != 0:
                    self.soft_sum(ochart[c2])
                else:
                    self.set_root(ochart[c2])
            
    
    def forward(self,  # pylint: disable=arguments-differ
                inputs: PackedSequence,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        parameters
        ----------
        inputs : packedsequence, required.
            a tensor of shape (batch_size, num_timesteps, input_size)
            to apply the lstm over.

        initial_state : tuple[torch.tensor, torch.tensor], optional, (default = none)
            a tuple (state, memory) representing the initial hidden state and memory
            of the lstm. each tensor has shape (1, batch_size, output_dimension).

        returns
        -------
        a packedsequence containing a torch.floattensor of shape
        (batch_size, num_timesteps, output_dimension) representing
        the outputs of the lstm per timestep and a tuple containing
        the lstm state, with shape (1, batch_size, hidden_size) to
        match the pytorch api.
        """
        if not isinstance(inputs, PackedSequence):
            raise Configurationerror('inputs must be packedsequence but got %s' % (type(inputs)))

        sequence_tensor, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        bsize = sequence_tensor.size()[0]
        nword = sequence_tensor.size()[1]
        nstep = nword * (nword + 1) // 2    

        #print('---module size: ', bsize, nstep, sequence_tensor.size())
        
        chart = {'ins': [dict() for _ in range(nstep)], 'out': [dict() for _ in range(nstep)]}
        chart['nword'], chart['nstep'] = nword, nstep

        if self.training:
            self.do_inside(sequence_tensor, chart)
            self.do_outside(chart)        
             
            h_outs, c_outs, v_outs = [], [], []
            for cell in chart['out'][nstep - nword:]:
                #h_outs.append(cell['v'][:self.hidden_size])
                #c_outs.append(cell['v'][self.hidden_size:])
                v_outs.append(cell['v'])
            '''
            h_outs = torch.stack(h_outs, 1)
            h_outs = torch.unsqueeze(h_outs, 0)
            
            c_outs = torch.stack(c_outs, 1)
            c_outs = torch.unsqueeze(c_outs, 0)
            '''
            v_outs = torch.stack(v_outs, 0)
            v_outs = torch.unsqueeze(v_outs, 0)
            v_outs = pack_padded_sequence(v_outs, batch_lengths, batch_first=True)
            
            return v_outs, None
        else:
            spans = [[0, nword - 1]]
            chart['max'] = [dict() for _ in range(nstep)]
            self.inference(sequence_tensor, chart)
            self.extract_span(0, chart, spans)
            spans = torch.tensor(spans, dtype=torch.long)
            
            batch_lengths = torch.tensor(spans[:, 0].size())
            spans = torch.unsqueeze(spans, 0)
            spans = pack_padded_sequence(spans, batch_lengths, batch_first=True)
            
            return spans, None 
