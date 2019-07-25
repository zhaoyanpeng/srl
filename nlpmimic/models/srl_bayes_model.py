from typing import Dict, List, TextIO, Optional, Any
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator

@Model.register("srl_vae_bayes")
class VaeSemanticRoleLabeler(Model):
    def __init__(self, vocab: Vocabulary,
                 classifier: Model,
                 autoencoder: Model = None,
                 feature_dim: int = None,
                 alpha: float = 0.0,
                 nsampling: int = 10,
                 kl_prior: str = None,
                 reweight: bool = True,
                 straight_through: bool = True,
                 continuous_label: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(VaeSemanticRoleLabeler, self).__init__(vocab, regularizer)
        self.minimum_float = 1e-25

        self.classifier = classifier
        self.autoencoder = autoencoder
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.nsampling = nsampling
        self.kl_prior = kl_prior
        self.reweight = reweight
        self.straight_through = straight_through
        self.continuous_label = continuous_label

        # auto-regressive model of the decoder will need lemma weights 
        self.nlemma = self.vocab.get_vocab_size("lemmas")
        #lemma_embedder = getattr(self.classifier.lemma_embedder, 'token_embedder_{}'.format('lemmas'))
        #self.lemma_vectors = lemma_embedder.weight
        
        input_dim, output_dim = self.classifier.lemma_embedder.get_output_dim(), self.nlemma
        if self.feature_dim is not None: # use negative sampling loss
            output_dim = self.feature_dim

        dense_layer = torch.nn.Bilinear(input_dim, input_dim, output_dim, bias=True) 
        setattr(self, 'bilinear_layer', dense_layer)
        dense_layer = torch.nn.Linear(input_dim, output_dim, bias=False) 
        setattr(self, 'p_linear_layer', dense_layer)
        dense_layer = torch.nn.Linear(input_dim, output_dim, bias=False) 
        setattr(self, 'r_linear_layer', dense_layer)

        self.tau = self.classifier.tau
        initializer(self)
    
    def softmax_loss(self):
        pass
    
    def score_tuple(self, features, lemma_indice):
        # features X target_lemmas
        dim = features.size(-1)
        lemma_indice = {'lemmas': lemma_indice}
        embedded_lemmas = self.classifier.encode_lemma(lemma_indice, None)
        embedded_lemmas = embedded_lemmas.view(-1, dim) # (bsize * nnode, dim) 
        scores = torch.bmm(features, embedded_lemmas.unsqueeze(-1))
        scores = torch.sigmoid(scores)
        scores = scores.squeeze(-1).squeeze(-1)
        return scores 

    def negative_sampling_loss(self, features, lemma_indice, mask):
        # features: (bsize, nnode, dim) 
        # lemmas  : (bsize, nnode, dim)
        bsize, dim = features.size(0), features.size(-1)
        features = features.view(-1, dim).unsqueeze(1) 

        gold_scores = self.score_tuple(features, lemma_indice)

        loss = -gold_scores * self.nsampling
        for _ in range(self.nsampling):
            nsample = features.size(0)
            samples = torch.randint(0, self.nlemma - 1, (nsample,), device=features.device)
            samples = samples.view(bsize, -1)
            samples = (samples + 1 + lemma_indice) % self.nlemma
            
            fake_scores = self.score_tuple(features, samples)
            loss += fake_scores

        loss *= mask.view(-1).float()
        loss /= self.nsampling
        #loss = loss.sum() / argument_mask.float().sum()
        loss = loss.sum() / bsize
        return loss
    
    def extract_features(self, v_labels, v_predts):
        v_labels = v_labels.contiguous()
        v_predts = v_predts.contiguous()

        b_logits = self.bilinear_layer(v_labels, v_predts)
        p_logits = self.p_linear_layer(v_labels)
        r_logits = self.r_linear_layer(v_predts) 
        features = b_logits + p_logits + r_logits
        return features

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                lemmas: Dict[str, torch.LongTensor],
                predicates: torch.LongTensor,
                predicate_indicators: torch.LongTensor,
                argument_indices: torch.LongTensor = None,
                predicate_index: torch.LongTensor = None,
                argument_mask: torch.LongTensor = None,
                srl_frames: torch.LongTensor = None,
                retrive_crossentropy: bool = False,
                supervisely_training: bool = False, # deliberately added here
                compute_mutual_infos: bool = False,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        _, arg_labels, arg_lemmas = self.classifier.select_args(
            None, srl_frames, lemmas['lemmas'], argument_indices) 

        # labels # (bsize, nrole, dim)
        # in inference, need to score all roles: p(r | p, a)
        batch_size = arg_labels.size(0) 
        encoded_labels = self.classifier.embed_labels(None)  
        encoded_labels = encoded_labels.unsqueeze(0).expand(batch_size, -1, -1)
        
        # predicates # (bsize, nrole, dim)
        nrole = encoded_labels.size(1)
        encoded_predts = self.classifier.encode_predt(predicates, predicate_indicators)
        encoded_predts = encoded_predts.expand(-1, nrole, -1) 

        # features (bsize, nrole, dim)
        # can be logits of arguments or argument features
        features = self.extract_features(encoded_labels, encoded_predts)
        
        # lemmas (bsize, nnode, dim)
        this_lemmas = {"lemmas": arg_lemmas}
        embedded_lemmas = self.classifier.encode_lemma(this_lemmas, None)
        embedded_lemmas = embedded_lemmas.transpose(-1, -2)

        role_logits = torch.bmm(features, embedded_lemmas) 
        role_logits = role_logits.transpose(-1, -2).contiguous()
        
        output_dict = {"logits": role_logits, "mask": argument_mask}
        self.classifier.add_argument_outputs(0, argument_mask, role_logits, arg_labels + 1, output_dict, \
            all_labels=srl_frames, arg_mask=argument_mask, arg_indices=argument_indices, metadata=metadata) 
        
        if retrive_crossentropy:
            output_dict['ce_loss'] = self.classifier.labeled_loss(argument_mask, role_logits, arg_labels)

        
        ### evaluation only
        if not self.training: 
            return output_dict 
        ### evaluation over


        # features (bsize, nnode, dim)
        # in training, exact roles are known: p(a | p, r)
        feature_dim = features.size(-1)
        roles_indice = arg_labels.unsqueeze(-1).expand(-1, -1, feature_dim)
        features = torch.gather(features, 1, roles_indice) 
        
        if self.feature_dim is not None:
            loss = self.negative_sampling_loss(features, arg_lemmas, argument_mask) 
        else:
            pass
        output_dict['loss'] = loss
        return output_dict 

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.classifier.decode_arguments(output_dict)

    @overrides       
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.classifier.get_metrics(reset=reset)

