from typing import Dict, List, TextIO, Optional, Any

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode

from nlpmimic.training.metrics import DependencyBasedF1Measure


@Model.register("srl_lower")
class LowerSemanticRoleLabeler(Model):
    def __init__(self, vocab: Vocabulary,
                 outputfile: str,
                 token_embedder: TextFieldEmbedder,
                 lemma_embedder: TextFieldEmbedder = None,
                 psign_dim: int = None, # psign: predicate sign bit (0/1) 
                 float_format: str = '{:.9f}',
                 seq_encoder: Seq2SeqEncoder = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(LowerSemanticRoleLabeler, self).__init__(vocab, regularizer)
        self.minimum_float = 1e-25

        self.outputfile = open(outputfile, 'w')

        self.seq_encoder = seq_encoder

        self.token_embedder = token_embedder
        self.lemma_embedder = lemma_embedder
        self.psign_embedder = None
        if psign_dim is not None:
            self.psign_embedder = Embedding(2, psign_dim)
        
        self.float_format = float_format
        self.global_counter = 0
        
        initializer(self)
    
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                lemmas: Dict[str, torch.LongTensor],
                predicate_indicators: torch.LongTensor,
                predicates: torch.LongTensor,
                argument_indices: torch.LongTensor = None,
                predicate_index: torch.LongTensor = None,
                argument_mask: torch.LongTensor = None,
                srl_frames: torch.LongTensor = None,
                reconstruction_loss: bool = False,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """ purely supervised learning of semantic role labels.
        """
        if self.training:

            embedded_tokens = self.token_embedder(tokens)
            batch_size, seq_length, dim = embedded_tokens.size()

            arg_labels = torch.gather(srl_frames, 1, argument_indices)
            arg_labels = [arg_labels[iseq].data.tolist() for iseq in range(batch_size)]

            arg_indices = argument_indices.unsqueeze(-1).expand(-1, -1, dim)
            arg_vectors = torch.gather(embedded_tokens, 1, arg_indices) 

            arg_indices = [argument_indices[iseq].data.tolist() for iseq in range(batch_size)]
            arg_lengths = get_lengths_from_binary_sequence_mask(argument_mask).data.tolist()

            #print(embedded_tokens)
            
            for iseq, arg_length in enumerate(arg_lengths):
                self.global_counter += 1
                this_arg_labels = arg_labels[iseq][:arg_length] 
                this_arg_indices = arg_indices[iseq][:arg_length]

                #tags = [self.vocab.get_token_from_index(x, namespace="srl_tags") for x in this_arg_labels]
                #print(tags)

                for iarg, (label, pos) in enumerate(zip(this_arg_labels, this_arg_indices)):
                    arg_vector = arg_vectors[iseq, iarg, :].detach().cpu().data.tolist() 
                    arg_vector = map(lambda x: self.float_format.format(x).rstrip('0'), arg_vector)
                    arg_vector = ' '.join(arg_vector)

                    arg_key = '{}.{}.{}'.format(self.global_counter, pos, label)

                    self.outputfile.write('{} {}\n'.format(arg_key, arg_vector)) 

        output_dict = {'loss': torch.tensor(self.global_counter)}
        return output_dict 
    
    def close(self):
        self.outputfile.close()

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return output_dict

    @overrides       
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'f1-measure-overall': 0.0} 

