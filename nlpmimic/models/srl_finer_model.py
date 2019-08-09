from typing import Dict, List, TextIO, Optional, Any
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator

"""
A basic VAE model treating only role labels as latent variables
"""

@Model.register("srl_vae_finer")
class VaeSemanticRoleLabeler(Model):
    def __init__(self, vocab: Vocabulary,
                 classifier: Model,
                 autoencoder: Model,
                 alpha: float = 0.0,
                 nsampling: int = 10,
                 kl_prior: str = None,
                 reweight: bool = True,
                 straight_through: bool = True,
                 continuous_label: bool = True,
                 way2relax_argmax: str = 'softmax', # or sinkhorn
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(VaeSemanticRoleLabeler, self).__init__(vocab, regularizer)
        self.minimum_float = 1e-25

        self.classifier = classifier
        self.autoencoder = autoencoder

        self.alpha = alpha
        self.nsampling = nsampling
        self.kl_prior = kl_prior
        self.reweight = reweight
        self.straight_through = straight_through
        self.continuous_label = continuous_label
        self.way2relax_argmax = way2relax_argmax

        # auto-regressive model of the decoder will need lemma weights 
        lemma_embedder = getattr(self.classifier.lemma_embedder, 'token_embedder_{}'.format('lemmas'))
        self.autoencoder.add_parameters(self.classifier.nclass,
                                        self.vocab.get_vocab_size("lemmas"),
                                        self.classifier.lemma_embedder)

        self.tau = self.classifier.tau
        initializer(self)
   
    def supervised(self, argument_mask, arg_logits, arg_labels, embedded_nodes, ctx_lemmas, output_dict):
        # classification loss for the labeled data
        C = self.classifier.labeled_loss(argument_mask, arg_logits, arg_labels, average=None) 
        # used in decoding
        embedded_labels = self.classifier.embed_labels(arg_labels, labels_add_one=True)  

        L = self.autoencoder(argument_mask, arg_lemmas, embedded_nodes, arg_labels, embedded_labels, ctx_lemmas)

        L, C = -torch.mean(L), torch.mean(C) 

        output_dict['L'] = L 
        output_dict['C'] = C 
        output_dict['loss'] = L + self.alpha * C 
        output_dict['LL'] = torch.mean(self.autoencoder.likelihood)

    def unsupervised_softmax(self, argument_mask, arg_logits, arg_labels, arg_lemmas, embedded_nodes, ctx_lemmas, output_dict):
        y_logs, y_ls, y_lprobs, lls, kls, uniques = [], [], [], [], [], []
        for _ in range(self.nsampling):
            gumbel_hard, gumbel_soft, gumbel_soft_log, sampled_labels = \
                self.classifier.gumbel_relax(argument_mask, arg_logits)
            # used in decoding, p(a | p, r) with continuous p (predicates) and r (roles)
            labels_relaxed = gumbel_hard if self.straight_through else gumbel_soft
            encoded_labels = self.classifier.embed_labels(None, labels_relaxed=labels_relaxed)  

            onehots = labels_relaxed if self.continuous_label else None #  
            L_y = self.autoencoder(argument_mask, arg_lemmas, embedded_nodes, sampled_labels, 
                encoded_labels, ctx_lemmas, edge_type_onehots = onehots)
            lls.append(L_y)
            
            # log posteriors
            indice = torch.max(gumbel_hard, -1, keepdim=True)[1]
            hard_lprobs = torch.gather(gumbel_soft_log, -1, indice).squeeze(-1)
            hard_lprobs = hard_lprobs.masked_fill(argument_mask == 0, 0)
            y_log = torch.sum(hard_lprobs, -1) # posteriors q(r | x, p, a)
            
            y_lprobs.append(y_log)

            # TODO it is equivalent to the entropy, kl(q || c), c: constant
            # this is not true, we are using the Gumbel distribution
            onehots = gumbel_hard if self.kl_prior is not None else None
            y_log, uniqueness = self.autoencoder.kld(y_log, 
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
        y_logs = torch.mean(y_logs, 0)
        y_ls = torch.mean(y_ls, 0)

        if len(uniques) > 0: # loss > 0 to be minimized
            uniques = torch.stack(uniques, 0)
            uniques = torch.mean(uniques, 0)
            uniques = torch.mean(uniques)
            output_dict['kl_loss'] = uniques 

        # along batch dimension
        KL = torch.mean(y_logs) 
        L_u = torch.mean(y_ls)

        output_dict['KL'] = KL 
        output_dict['L_u'] = L_u
        output_dict['loss'] = -L_u + KL

    def unsupervised_sinkhorn(self, argument_mask, arg_logits, arg_labels, arg_lemmas, embedded_nodes, ctx_lemmas, output_dict):
        #maximum = torch.max(arg_logits, -1, keepdim=True)[0]
        #arg_logits = arg_logits - maximum # to avoid potential overflow 

        y_logs, y_ls, y_lprobs, lls, kls, uniques = [], [], [], [], [], []
        for _ in range(self.nsampling):
            gumbel_hard, gumbel_soft, gumbel_soft_log, sampled_labels = \
                self.classifier.gumbel_relax(argument_mask, arg_logits)
            # used in decoding, p(a | p, r) with continuous p (predicates) and r (roles)
            labels_relaxed = gumbel_hard if self.straight_through else gumbel_soft
            encoded_labels = self.classifier.embed_labels(None, labels_relaxed=labels_relaxed)  

            onehots = labels_relaxed if self.continuous_label else None #  
            L_y = self.autoencoder(argument_mask, arg_lemmas, embedded_nodes, sampled_labels, 
                encoded_labels, ctx_lemmas, edge_type_onehots = onehots)
            lls.append(L_y)
            
            # log posteriors
            indice = torch.max(gumbel_hard, -1, keepdim=True)[1]
            hard_lprobs = torch.gather(gumbel_soft_log, -1, indice).squeeze(-1)
            hard_lprobs = hard_lprobs.masked_fill(argument_mask == 0, 0)
            y_log = torch.sum(hard_lprobs, -1) # posteriors q(r | x, p, a)
            
            y_lprobs.append(y_log)

            # TODO it is equivalent to the entropy, kl(q || c), c: constant
            # this is not true, we are using the Gumbel distribution
            onehots = gumbel_hard if self.kl_prior is not None else None
            y_log, uniqueness = self.autoencoder.kld(
                                         arg_logits, 
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
        y_logs = torch.mean(y_logs, 0)
        y_ls = torch.mean(y_ls, 0)

        if len(uniques) > 0: # loss > 0 to be minimized
            uniques = torch.stack(uniques, 0)
            uniques = torch.mean(uniques, 0)
            uniques = torch.mean(uniques)
            output_dict['kl_loss'] = uniques 

        # along batch dimension
        KL = torch.mean(y_logs) 
        L_u = torch.mean(y_ls)

        output_dict['KL'] = KL 
        output_dict['L_u'] = L_u
        output_dict['loss'] = -L_u + KL

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
        embedded_seqs, logits, mask = out_dict['embedded_seqs'], out_dict['logits'], out_dict['mask']

        arg_logits, arg_labels, arg_lemmas = self.classifier.select_args(
            logits, srl_frames, lemmas['lemmas'], argument_indices) 

        # basic output stuff 
        output_dict = {"logits": logits[pivot:],
                       "logits_softmax": out_dict['logits_softmax'][pivot:],
                       "mask": mask[pivot:]}
        
        if not supervisely_training: # do not need to evaluate labeled data
            self.classifier.add_outputs(pivot, mask, logits, srl_frames, output_dict, \
                arg_mask=argument_mask, arg_indices=argument_indices, predicates=predicates, metadata=metadata) 

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
        lemma_ctx = self.classifier.encode_lemma_ctx(arg_lemmas)

        ### labeled halve
        if supervisely_training:
            self.supervised(argument_mask, arg_logits, arg_labels, embedded_nodes, lemma_ctx, output_dict) 
        else: ### unlabled halve
            if self.way2relax_argmax == 'softmax':
                self.unsupervised_softmax(argument_mask, 
                    arg_logits, arg_labels, arg_lemmas, embedded_nodes, lemma_ctx, output_dict) 
            elif self.way2relax_argmax == 'sinkhorn':
                self.unsupervised_sinkhorn(argument_mask, 
                    arg_logits, arg_labels, arg_lemmas, embedded_nodes, lemma_ctx, output_dict) 
        return output_dict 

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.classifier.decode(output_dict)

    @overrides       
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.classifier.get_metrics(reset=reset)

