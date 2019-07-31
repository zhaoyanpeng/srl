from typing import Dict, List, TextIO, Optional, Any
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator

@Model.register("srl_vae_titov")
class VaeSemanticRoleLabeler(Model):
    def __init__(self, vocab: Vocabulary,
                 classifier: Model,
                 autoencoder: Model = None,
                 feature_dim: int = None,
                 alpha: float = 0.0,
                 nsampling: int = 10,
                 kl_prior: str = None,
                 reweight: bool = True,
                 loss_type: str = 'ivan',
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
        self.loss_type = loss_type 
        self.nsampling = nsampling
        self.kl_prior = kl_prior
        self.reweight = reweight
        self.straight_through = straight_through
        self.continuous_label = continuous_label

        # auto-regressive model of the decoder will need lemma weights 
        lemma_embedder = getattr(self.classifier.lemma_embedder, 'token_embedder_{}'.format('lemmas'))
        self.lemma_dim = lemma_embedder.get_output_dim()
        self.nlemma = self.vocab.get_vocab_size("lemmas")
        
        # global representation of predicates and per-lemma scalar
        lemma_scalar = torch.empty((1, self.nlemma), device=self.classifier.tau.device)
        torch.nn.init.xavier_uniform_(lemma_scalar)
        self.lemma_scalar = torch.nn.Parameter(lemma_scalar, requires_grad=True)

        global_contx = torch.empty((1, self.feature_dim), device=self.classifier.tau.device)
        torch.nn.init.xavier_uniform_(global_contx)
        self.global_contx = torch.nn.Parameter(global_contx, requires_grad=True)
        
        self.tau = self.classifier.tau
        initializer(self)
   
    def compute_potential(self, dim: int, lemmas: torch.Tensor, 
                          ctxs: torch.Tensor, expected_roles: torch.Tensor):
        this_lemmas = {'lemmas': lemmas}

        embedded_arguments = self.classifier.encode_lemma(this_lemmas, None)
        embedded_arguments = embedded_arguments.view(-1, dim).unsqueeze(-1) # (bsize * narg, dim, 1)  
        # (bsize * narg, k, dim) X (bsize * narg, dim, 1)
        args_and_roles = torch.bmm(expected_roles, embedded_arguments).squeeze(-1) 
        # (bsize * narg, 1, k) X (bsize * narg, k, 1)
        scores = torch.bmm(ctxs.unsqueeze(1), args_and_roles.unsqueeze(-1)) 
        scores = scores.squeeze(-1).squeeze(-1)

        args_scalar = torch.gather(self.lemma_scalar, -1, lemmas.view(1, -1)) 
        args_scalar = args_scalar.squeeze(0)

        scores += args_scalar
        scores = torch.sigmoid(scores)
        return scores

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
        out_dict = self.classifier(tokens, predicate_indicators) 

        embedded_seqs, logits, mask = out_dict['embedded_seqs'], out_dict['logits'], out_dict['mask']

        arg_logits, arg_labels, arg_lemmas = self.classifier.select_args(
            logits, srl_frames, lemmas['lemmas'], argument_indices) 

        # basic output stuff 
        output_dict = {"logits": logits, "mask": mask, "logits_softmax": out_dict['logits_softmax']}
        
        if not supervisely_training: # do not need to evaluate labeled data
            self.classifier.add_outputs(0, mask, logits, srl_frames, output_dict, \
                arg_mask=argument_mask, arg_indices=argument_indices, metadata=metadata) 

        if retrive_crossentropy:
            output_dict['ce_loss'] = self.classifier.labeled_loss(argument_mask, arg_logits, arg_labels)
        

        ### evaluation only
        if not self.training: 
            return output_dict 
        ### evaluation over
        
        bsize, nrole = argument_indices.size(0), self.classifier.nclass
        k, narg, dim = self.feature_dim, argument_indices.size(1), self.lemma_dim

        global_predt = self.classifier.encode_global_predt(device=self.classifier.tau.device)
        
        # (bsize, 1, nrole * k * dim): k featuers & dim of lemma embeddings
        embedded_predicates = self.classifier.encode_predt(predicates, predicate_indicators)
        embedded_predicates = embedded_predicates.squeeze(1)
        embedded_predicates += global_predt.unsqueeze(0) 
        embedded_predicates = embedded_predicates.view(bsize, nrole, -1) # (bsize, nrole, k * dim)
        
        role_probs = torch.softmax(arg_logits, -1)  

        # (bsize, narg, nrole) X (bsize, nrole, k * dim)
        expected_roles = torch.bmm(role_probs, embedded_predicates) # (bsize, narg, k * dim)
        expected_roles = expected_roles.view(bsize, narg, k, dim) # (bsize, narg, k, dim)
        expected_roles = expected_roles.view(-1, k, dim) # (bsize * narg, k, dim)
        
        # (bsize, narg, dim)
        this_lemmas = {'lemmas': arg_lemmas}
        embedded_arguments = self.classifier.encode_lemma(this_lemmas, None)
        embedded_arguments = embedded_arguments.view(-1, dim).unsqueeze(-1)   
        # (bsize * narg, k, dim) X (bsize * narg, dim, 1) -> features
        args_and_roles = torch.bmm(expected_roles, embedded_arguments).squeeze(-1) 
        args_and_roles = args_and_roles.view(bsize, -1, k) # (bsize, narg, k)
        
        args_and_roles = args_and_roles * argument_mask.unsqueeze(-1).float()
        args_and_roles_sum = torch.sum(args_and_roles, 1, keepdim=True) 
        ctxs = args_and_roles_sum - args_and_roles # (bsize, narg, k)
        ctxs = ctxs.view(-1, k) # (bsize * narg, k)

        #ctxs += self.global_contx

        gold_scores = self.compute_potential(dim, arg_lemmas, ctxs, expected_roles) 

        loss = -gold_scores if self.loss_type == "ivan" else 0
        #loss = -gold_scores * self.nsampling
        for idx in range(self.nsampling):
            nsample = ctxs.size(0)
            samples = torch.randint(0, self.nlemma - 1, (nsample,), device=ctxs.device)
            samples = samples.view(bsize, -1)
            samples = (samples + 1 + arg_lemmas) % self.nlemma

            fake_scores = self.compute_potential(dim, samples, ctxs, expected_roles) 

            if self.loss_type == 'ivan':
                loss += fake_scores
            elif self.loss_type == 'relu':
                this_loss = torch.relu(1 - gold_scores + fake_scores)
                loss += this_loss

        if (argument_mask.sum(-1) == 0).any():
            raise ValueError("Empty argument set encountered.")

        loss *= argument_mask.view(-1).float()
        #loss /= self.nsampling
        loss = loss.view(bsize, narg)
        loss = loss.sum(-1) / argument_mask.sum(-1).float()
        loss = loss.sum() / bsize
        output_dict['loss'] = loss

        return output_dict 

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.classifier.decode(output_dict)

    @overrides       
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.classifier.get_metrics(reset=reset)

