from typing import List 
import torch

from allennlp.modules import Seq2SeqEncoder
from allennlp.modules import TextFieldEmbedder

@Seq2SeqEncoder.register("srl_graph_decoder")
class SrlGraphDecoder(Seq2SeqEncoder):

    def __init__(self, 
                 input_dim: int, 
                 dense_layer_dims: List[int],
                 dropout: float = 0.0) -> None:
        super(SrlGraphDecoder, self).__init__()
        self.signature = 'decoder'

        self._input_dim = input_dim 
        self._dense_layer_dims = dense_layer_dims 
        
        self._dense_layers = []
        for i_layer, dim in enumerate(self._dense_layer_dims):
            dense_layer = torch.nn.Linear(input_dim, dim, bias=True)
            setattr(self, 'dense_layer_{}'.format(i_layer), dense_layer)
            self._dense_layers.append(dense_layer)
            input_dim = dim
        
        self._dropout = torch.nn.Dropout(dropout)
    
    def add_parameters(self, output_dim: int, lemma_embedder: TextFieldEmbedder, nlabel: int = None) -> None:
        self._output_dim = output_dim 
        #self._lemma_embedder = lemma_embedder 
        
        label_layer = torch.nn.Linear(self._dense_layer_dims[-1], self._output_dim)
        setattr(self, 'label_projection_layer', label_layer)

    def kernel(self, ctx_lemmas, ctx_labels, role, pred):
        p_prod_r, p_diff_r = pred * role, pred - role 
        ctx_l_prod_r, ctx_l_diff_r = ctx_lemmas * ctx_labels, ctx_lemmas - ctx_labels 
        
        y1 = ctx_lemmas
        y2 = ctx_labels
        y3 = p_diff_r * ctx_l_prod_r
        y4 = ctx_l_diff_r * p_prod_r
        y5 = p_diff_r * ctx_l_diff_r

        features = [role, pred, p_prod_r, p_diff_r, y1, y2, y3, y4, y5]
        return features
    
    def argument_model(self):
        """ p(a_i | a-i, p)
        """
        pass

    def role_model(self):
        """ p(r_i | a-i, p)
        """
        pass

    def forward(self, 
                z: torch.Tensor,
                embedded_edges: torch.Tensor,
                embedded_predicates: torch.Tensor,
                nodes_contexts: torch.Tensor = None) -> torch.Tensor:
        nnode = embedded_edges.size(1)
        embedded_nodes = []

        if z is not None:
            nsample = z.size(1) 
            # (nsample, batch_size, nnode, dim)
            z = z.transpose(0, 1).unsqueeze(2).expand(-1, -1, nnode, -1) 
            embedded_nodes.append(z)
        else:
            nsample = 1

        embedded_edges = embedded_edges.unsqueeze(0).expand(nsample, -1, -1, -1) 
        embedded_predicates = embedded_predicates.unsqueeze(0).expand(nsample, -1, nnode, -1)
        
        if nodes_contexts is None:
            embedded_nodes += [embedded_edges, embedded_predicates]
        else:
            nodes_contexts = nodes_contexts.unsqueeze(0).expand(nsample, -1, -1, -1) 
            ctx_lemmas, ctx_labels = torch.chunk(nodes_contexts, 2, -1) 
            embedded_nodes += self.kernel(ctx_lemmas, ctx_labels, embedded_edges, embedded_predicates)

        embedded_nodes = torch.cat(embedded_nodes, -1)
        embedded_nodes = self.multi_dense(embedded_nodes)

        logits = self.label_projection_layer(embedded_nodes)
        return logits 
    
    def multi_dense(self, embedded_nodes: torch.Tensor) -> torch.Tensor:
        for dense_layer in self._dense_layers:
            embedded_nodes = torch.tanh(dense_layer(embedded_nodes))
            embedded_nodes = self._dropout(embedded_nodes)
        return embedded_nodes
    
    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

