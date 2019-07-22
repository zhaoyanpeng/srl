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
        lemma_embedder = getattr(self.classifier.lemma_embedder, 'token_embedder_{}'.format('lemmas'))
        nlemma = self.vocab.get_vocab_size("lemmas")
        if autoencoder is not None:
            self.autoencoder.add_parameters(self.classifier.nclass,
                                            nlemma,
                                            lemma_embedder.weight)
        
        # global representation of predicates and per-lemma scalar
        lemma_scalar = torch.empty((1, nlemma), device=self.classifier.tau.device)
        torch.nn.init.xavier_uniform_(lemma_scalar)
        self.lemma_scalar = torch.nn.Parameter(lemma_scalar, requires_grad=True)

        global_contx = torch.empty((1, self.feature_dim), device=self.classifier.tau.device)
        torch.nn.init.xavier_uniform_(global_contx)
        self.global_contx = torch.nn.Parameter(global_contx, requires_grad=True)
        
        self.tau = self.classifier.tau
        initializer(self)
    
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
        pivot = 0 # either labeled or unlabeled data
        out_dict = self.classifier(tokens, predicate_indicators) 
        embedded_seqs = out_dict['embedded_seqs']
        logits, mask = out_dict['logits'], out_dict['mask']

        arg_logits, arg_labels, arg_lemmas = self.classifier.select_args(
            logits, srl_frames, lemmas['lemmas'], argument_indices) 

        # basic output stuff 
        output_dict = {"logits": logits[pivot:],
                       "logits_softmax": out_dict['logits_softmax'][pivot:],
                       "mask": mask[pivot:]}
        
        if not supervisely_training: # do not need to evaluate labeled data
            self.classifier.add_outputs(pivot, mask, logits, srl_frames, output_dict, \
                arg_mask=argument_mask, arg_indices=argument_indices, metadata=metadata) 

        if retrive_crossentropy:
            output_dict['ce_loss'] = self.classifier.labeled_loss(
                argument_mask[pivot:], arg_logits[pivot:], arg_labels[pivot:])
        

        ### evaluation only
        if not self.training: 
            return output_dict 
        ### evaluation over
        
        global_predt = self.classifier.encode_global_predt(device=self.classifier.tau.device)
        
        # (bsize, narg, dim)
        embedded_arguments = self.classifier.encode_lemma(lemmas, argument_indices)
        embedded_predicates = self.classifier.encode_predt(predicates, predicate_indicators)
        bsize, nrole = embedded_predicates.size(0), self.classifier.nclass
        embedded_predicates = embedded_predicates.squeeze(1)
        embedded_predicates += global_predt.unsqueeze(0) # global predicate representation
        embedded_predicates = embedded_predicates.view(bsize, nrole, -1) # (bsize, nrole, dim)
        
        indice = arg_labels.unsqueeze(-1) 
        role_probs = torch.softmax(arg_logits, -1) # (bsize, narg, nrole)
        
        k, narg, dim = self.feature_dim, embedded_arguments.size(1), embedded_arguments.size(-1)  
        expected_roles = torch.bmm(role_probs, embedded_predicates) # (bsize, narg, k * dim)
        expected_roles = expected_roles.view(bsize, narg, k, dim) # (bsize, narg, k, dim)
        
        expected_roles = expected_roles.view(-1, k, dim) # (bsize * narg, k, dim)
        embedded_arguments = embedded_arguments.view(-1, dim).unsqueeze(-1) # (bsize * narg, dim, 1)  
        args_and_roles = torch.bmm(expected_roles, embedded_arguments).squeeze(-1) # (bsize * narg, k)
        args_and_roles = args_and_roles.view(bsize, -1, k) # (bsize, narg, k)
        
        args_and_roles = args_and_roles * argument_mask.unsqueeze(-1).float()
        args_and_roles_sum = torch.sum(args_and_roles, 1, keepdim=True) 
        ctxs = args_and_roles_sum - args_and_roles # (bsize, narg, k)
        ctxs = ctxs.view(-1, k) # (bsize * narg, k)

        ctxs += self.global_contx

        args_and_roles = args_and_roles.view(-1, k)
        gold_scores = torch.bmm(ctxs.unsqueeze(1), args_and_roles.unsqueeze(-1)) 
        gold_scores = gold_scores.squeeze(-1).squeeze(-1)

        args_scalar = torch.gather(self.lemma_scalar, -1, arg_lemmas.view(1, -1)) 
        args_scalar = args_scalar.squeeze(0)

        gold_scores += args_scalar
        gold_scores = torch.sigmoid(gold_scores)
        
        loss = -gold_scores * self.nsampling
        for _ in range(self.nsampling):
            nlemma = self.vocab.get_vocab_size("lemmas")
            nsample = ctxs.size(0)
            samples = torch.randint(0, nlemma - 1, (nsample,), device=ctxs.device)
            samples = samples.view(bsize, -1)
            samples = (samples + 1 + argument_indices) % nlemma
            this_lemmas = {'lemmas': samples}

            # (bsize, narg, dim)
            embedded_arguments = self.classifier.encode_lemma(this_lemmas, None)
            embedded_arguments = embedded_arguments.view(-1, dim).unsqueeze(-1) # (bsize * narg, dim, 1)  
            args_and_roles = torch.bmm(expected_roles, embedded_arguments).squeeze(-1) # (bsize * narg, k)
            args_and_roles = args_and_roles.view(bsize, -1, k) # (bsize, narg, k)
        
            args_and_roles = args_and_roles * argument_mask.unsqueeze(-1).float()
            args_and_roles = args_and_roles.view(-1, k)

            fake_scores = torch.bmm(ctxs.unsqueeze(1), args_and_roles.unsqueeze(-1)) 
            fake_scores = fake_scores.squeeze(-1).squeeze(-1)

            args_scalar = torch.gather(self.lemma_scalar, -1, samples.view(1, -1)) 
            args_scalar = args_scalar.squeeze(0)

            fake_scores += args_scalar
            fake_scores = torch.sigmoid(fake_scores)
            
            loss += fake_scores
        loss *= argument_mask.view(-1).float()
        loss /= self.nsampling
        loss = loss.sum() / argument_mask.float().sum()
        output_dict['loss'] = loss

        return output_dict 

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.classifier.decode(output_dict)

    @overrides       
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.classifier.get_metrics(reset=reset)

