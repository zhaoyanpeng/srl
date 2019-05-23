from typing import List 
import torch

from allennlp.modules import Seq2SeqEncoder

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
    
    def add_parameters(self, output_dim: int, lemma_embedder_weight: torch.Tensor) -> None:
        self._output_dim = output_dim 
        #self._lemma_embedder_weight = lemma_embedder_weight 

        label_layer = torch.nn.Linear(self._dense_layer_dims[-1], self._output_dim)
        setattr(self, 'label_projection_layer', label_layer)

    def forward(self, 
                z: torch.Tensor,
                embedded_edges: torch.Tensor,
                embedded_predicates: torch.Tensor = None) -> torch.Tensor:
        nsample = z.size(1) 
        nnode = embedded_edges.size(1)

        # (nsample, batch_size, nnode, dim)
        z = z.transpose(0, 1).unsqueeze(2).expand(-1, -1, nnode, -1) 
        embedded_edges = embedded_edges.unsqueeze(0).expand(nsample, -1, -1, -1) 
        embedded_nodes = torch.cat([z, embedded_edges], -1)
        if embedded_predicates is not None:
            embedded_predicates = embedded_predicates.unsqueeze(0).expand(nsample, -1, nnode, -1)
            embedded_nodes = torch.cat([embedded_nodes, embedded_predicates], -1)

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

