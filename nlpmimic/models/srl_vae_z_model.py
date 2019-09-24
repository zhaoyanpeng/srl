from typing import Dict, List, TextIO, Optional, Any
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator

@Model.register("srl_vae_hub_z")
class VaeSemanticRoleLabeler(Model):
    def __init__(self, vocab: Vocabulary,
                 classifier: Model,
                 autoencoder: Model,
                 n_sample: int = 10,    # n samples of latent role labels
                 ll_alpha: float = 0.0, # weight of the supervised loss
                 kl_prior: str = None,  # regularizing logits of role labels, uniqueness
                 reweight: bool = True, # weighting losses of sampled role labels
                 prior_type: str = 'uniform',   # prior of latent roles
                 max_pooling: bool = False,
                 rm_argument: bool = False,
                 straight_through: bool = True,
                 continuous_label: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(VaeSemanticRoleLabeler, self).__init__(vocab, regularizer)
        self.minimum_float = 1e-25

        self.classifier = classifier
        self.autoencoder = autoencoder

        self.ll_alpha = ll_alpha
        self.kl_prior = kl_prior
        self.n_sample = n_sample
        self.reweight = reweight
        self.prior_type = prior_type

        self.max_pooling = max_pooling
        self.rm_argument = rm_argument

        self.straight_through = straight_through
        self.continuous_label = continuous_label

        # auto-regressive model of the decoder will need lemma weights 
        self.autoencoder.add_parameters(self.classifier.nclass,
                                        self.vocab.get_vocab_size("lemmas"),
                                        self.classifier.lemma_embedder)

        self.tau = self.classifier.tau
        initializer(self)

    def anneal_kl(self, kl_alpha: float=1., ky_alpha: float=1.):
        self.autoencoder.anneal_kl(kl_alpha=kl_alpha, ky_alpha=ky_alpha)

    #@profile    
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
                iwanto_do_evaluation: bool = False,
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
        
        if not supervisely_training or iwanto_do_evaluation: # do not need to evaluate labeled data
            self.classifier.add_outputs(pivot, mask, logits, srl_frames, output_dict, \
                arg_mask=argument_mask, arg_indices=argument_indices, metadata=metadata) 

        if retrive_crossentropy:
            output_dict['ce_loss'] = self.classifier.labeled_loss(
                argument_mask[pivot:], arg_logits[pivot:], arg_labels[pivot:])
        

        ### evaluation only
        if not self.training: 
            return output_dict 
        ### evaluation over

        
        # below we finalize all the training stuff
        embedded_nodes = self.classifier.encode_args(
            lemmas, predicates, predicate_indicators, argument_indices, embedded_seqs) 
        lemma_ctx = None
        if getattr(self.classifier, "lectx_embedder", None):
            lemma_ctx = self.classifier.encode_lemma_ctx(arg_lemmas)
        
        skeletons = self.classifier.encode_skeletons(tokens, predicate_indicators, 
            argument_mask, argument_indices, max_pooling = self.max_pooling, rm_argument=self.rm_argument)


        ### labeled halve
        if supervisely_training:
            # classification loss for the labeled data
            C = self.classifier.labeled_loss(argument_mask, arg_logits, arg_labels, average=None) 
            # used in decoding
            encoded_labels = self.classifier.embed_labels(arg_labels, labels_add_one=True)  

            y_log, _ = self.autoencoder.kld(mask = argument_mask)
            L = self.autoencoder(argument_mask, arg_lemmas, embedded_nodes, 
                arg_labels, encoded_labels, lemma_ctx, skeletons)

            L, C = -torch.mean(L + y_log), torch.mean(C) 
            output_dict['L'] = L 
            output_dict['C'] = C 
            output_dict['loss'] = L + self.ll_alpha * C  
            if self.autoencoder.likelihood is not None:
                output_dict['LL'] = torch.mean(self.autoencoder.likelihood)
            if self.autoencoder.kldistance is not None:
                output_dict['KL'] = torch.mean(self.autoencoder.kldistance)
        else: ### unlabled halve
            y_logs, y_ls, y_lprobs, lls, kls, uniques = [], [], [], [], [], []
            for _ in range(self.n_sample):
                gumbel_hard, gumbel_soft, gumbel_soft_log, sampled_labels = \
                    self.classifier.gumbel_relax(argument_mask, arg_logits)
                # used in decoding
                labels_relaxed = gumbel_hard if self.straight_through else gumbel_false
                encoded_labels = self.classifier.embed_labels(None, labels_relaxed=labels_relaxed)  
                
                onehots = labels_relaxed if self.continuous_label else None
                L_y = self.autoencoder(argument_mask, arg_lemmas, embedded_nodes, 
                    sampled_labels, encoded_labels, lemma_ctx, skeletons, edge_type_onehots = onehots)
                if self.autoencoder.likelihood is not None:
                    lls.append(self.autoencoder.likelihood)
                if self.autoencoder.kldistance is not None:
                    kls.append(self.autoencoder.kldistance)
                
                # log posteriors
                hard_lprobs = (gumbel_hard * gumbel_soft_log).sum(-1)
                hard_lprobs = hard_lprobs.masked_fill(argument_mask == 0, 0)
                y_log = torch.sum(hard_lprobs, -1) 

                y_lprobs.append(y_log) # posteriros

                # kl term, we may use a pretrained decoder to compute priors of y
                # TODO currently it is the same as entropy
                onehots = gumbel_hard if self.kl_prior is not None else None
                y_log, uniqueness = self.autoencoder.kld(
                                            posteriors = None, 
                                            mask = argument_mask, 
                                            node_types = arg_lemmas, 
                                            embedded_nodes = embedded_nodes, 
                                            embedded_edges = encoded_labels,
                                            edge_type_onehots = onehots)
                if uniqueness is not None:
                    uniques.append(uniqueness)
                y_logs.append(y_log)
                y_ls.append(L_y)

            # samples (nsample, batch_size) 
            y_lprobs = torch.stack(y_lprobs, 0)
            y_logs = torch.stack(y_logs, 0)
            y_ls = torch.stack(y_ls, 0)

            # along sample dimension
            y_probs = torch.exp(y_logs)
            if self.reweight:
                y_probs = y_probs.softmax(0)

            y_ls = (y_ls + y_logs) * y_probs
            H = torch.log(y_probs + self.minimum_float) * y_probs
            
            L_u, H = y_ls.sum(0), H.sum(0)
            L_u, H = -torch.mean(L_u), -torch.mean(H) 

            output_dict['L_u'] = L_u 
            output_dict['H'] = H 
            output_dict['loss'] = L_u + H # u = L_u - H
            
            if len(uniques) > 0: # loss > 0 to be minimized
                uniques = torch.stack(uniques, 0)
                uniques = torch.mean(uniques, 0)
                uniques = torch.mean(uniques)
                output_dict['kl_loss'] = uniques 

            if len(lls) > 0 and len(kls) > 0:
                lls = torch.stack(lls, 0)
                kls = torch.stack(kls, 0)
                if (kls < 0).any():
                    raise ValueError('KL should be non-negative.') 
                output_dict['KL'] = torch.mean(kls) 
                output_dict['LL'] = torch.mean(lls)
        return output_dict 

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.classifier.decode(output_dict)

    @overrides       
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.classifier.get_metrics(reset=reset)

