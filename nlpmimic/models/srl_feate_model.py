from typing import Dict, List, TextIO, Optional, Any
from overrides import overrides
import torch, sys
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator

@Model.register("srl_vae_feate")
class VaeSemanticRoleLabeler(Model):
    def __init__(self, vocab: Vocabulary,
                 classifier: Model,
                 feature_dim: int = None,
                 unique_role: bool = False,
                 loss_type: str = 'ivan',
                 nsampling: int = 10,
                 nsampling_power: float = 0.75, # power of word frequencies in negative sampling
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(VaeSemanticRoleLabeler, self).__init__(vocab, regularizer)
        self.minimum_float = 1e-25

        self.classifier = classifier
        self.feature_dim = feature_dim
        self.unique_role = unique_role
        self.loss_type = loss_type 
        self.nsampling = nsampling
            
        # auto-regressive model of the decoder will need lemma weights 
        argmt_embedder = getattr(self.classifier.argmt_embedder, 'token_embedder_{}'.format('argmts'))
        self.argmt_dim = argmt_embedder.get_output_dim()
        self.nlemma = self.vocab.get_vocab_size("argmts")
        
        # categorical distribution over arguments
        lemma_distr = torch.zeros((self.nlemma,))
        lemma_table = list(self.vocab._retained_counter['argmts'].items())
        for k, v in self.vocab._retained_counter['argmts'].items():
            lemma_index = self.vocab.get_token_index(k, namespace='argmts')
            lemma_distr[lemma_index] = v
        self.lemma_distr = torch.pow(lemma_distr, nsampling_power)
        
        # global representation of predicates and per-lemma scalar
        lemma_scalar = torch.empty((1, self.nlemma), device=self.classifier.tau.device)
        #torch.nn.init.xavier_uniform_(lemma_scalar)
        lemma_scalar.zero_()
        self.lemma_scalar = torch.nn.Parameter(lemma_scalar, requires_grad=True)
        
        # W * x + b where b is the label scalar
        label_scalar = torch.empty((1, self.classifier.nclass), device=self.classifier.tau.device)
        #torch.nn.init.xavier_uniform_(label_scalar)
        label_scalar.zero_()
        self.label_scalar = torch.nn.Parameter(label_scalar, requires_grad=True)

        feate_scalar = torch.empty((1, self.feature_dim), device=self.classifier.tau.device)
        #torch.nn.init.xavier_uniform_(feate_scalar)
        feate_scalar.zero_()
        self.feate_scalar = torch.nn.Parameter(feate_scalar, requires_grad=True)
        
        self.tau = self.classifier.tau
        initializer(self)

    def compute_potential_batch(self, dim: int, lemmas: torch.Tensor, 
                           ctxs: torch.Tensor, expected_roles: torch.Tensor, args_scalar=None):
        nsample = lemmas.size(-1) 
        this_lemmas = {'argmts': lemmas}
        # (bsize, narg, nsample, dim)
        embedded_arguments = self.classifier.encode_lemma(this_lemmas)
        embedded_arguments = embedded_arguments.view(-1, nsample, dim)  
        # (bsize * narg, nsample, dim) X (bsize * narg, dim, k) 
        args_and_roles = torch.bmm(embedded_arguments, expected_roles.transpose(-1, -2)) 
        # (bsize * narg, nsample, k) X (bsize * narg, k, 1) 
        scores = torch.bmm(args_and_roles, ctxs.unsqueeze(-1)) 
        scores = scores.squeeze(-1)
        
        if args_scalar is None:
            args_scalar = torch.gather(self.lemma_scalar, -1, lemmas.view(1, -1)) 
            args_scalar = args_scalar.squeeze(0)
            args_scalar = args_scalar.view(-1, nsample)

        scores = scores + args_scalar
        return scores

    def forward(self,  # type: ignore
                predicate: Dict[str, torch.LongTensor],
                arguments: Dict[str, torch.LongTensor],
                feate_ids: Dict[str, torch.LongTensor],
                srl_frames: torch.LongTensor = None,
                feate_lens: torch.LongTensor = None,
                retrive_crossentropy: bool = False,
                supervisely_training: bool = False, # deliberately added here
                compute_mutual_infos: bool = False,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # e_p: (bsize, nrole * lemma_dim * k)
        e_p, m_p, e_a, m_a, e_f, m_f = self.classifier.encode_all(predicate, arguments, feate_ids)
        
        k, dim = self.feature_dim, self.argmt_dim
        (bsize, narg), nrole = srl_frames.size(), self.classifier.nclass 

        # role labeling model
        arg_logits = e_f.view(bsize, narg, -1, nrole)  
        feate_dim = arg_logits.size(-2) # # of discrete features

        feate_lens = feate_lens.view(-1)
        arg_logits = arg_logits.view(-1, feate_dim, nrole)

        for i in range(arg_logits.size(0)):
            arg_logits[i, feate_lens[i]:, : ] *= 0   # mask non-features 
        arg_logits = arg_logits.view(bsize, narg, feate_dim, nrole)
        arg_logits = torch.sum(arg_logits, -2)
        arg_logits = arg_logits + self.label_scalar.unsqueeze(0) # bias

        # basic output stuff 
        argument_mask = m_a
        output_dict = {"logits": arg_logits, "mask": argument_mask}

        if not supervisely_training: # do not need to evaluate labeled data
            self.classifier.add_outputs(argument_mask, arg_logits, srl_frames, 
                output_dict, predicates=predicate['predts'], metadata=metadata) 
        
        if retrive_crossentropy:
            output_dict['ce_loss'] = None 


        ### evaluation only
        if not self.training: 
            return output_dict 
        ### evaluation over


        minimum = 1e-15
        role_probs = torch.softmax(arg_logits, -1)
        if self.unique_role:
            log_role_probs = F.log_softmax(arg_logits, -1)
                 
            log_flip_probs = torch.log(1 - role_probs + minimum)
            log_flip_probs = log_flip_probs * argument_mask.unsqueeze(-1).float()

            log_flip_probs_sum = torch.sum(log_flip_probs, 1, keepdim=True)
            log_flip_probs_sum = log_flip_probs_sum - log_flip_probs 
            log_flip_probs_sum = log_flip_probs_sum + log_role_probs

            role_probs = torch.softmax(log_flip_probs, -1)

        # (bsize, 1, nrole * k * dim): k featuers & dim of lemma embeddings
        global_predt = self.classifier.encode_global_predt(device=self.classifier.tau.device)
        embedded_predicates = e_p.squeeze(1)
        embedded_predicates = embedded_predicates + global_predt.unsqueeze(0) 
        embedded_predicates = embedded_predicates.view(bsize, nrole, -1) # (bsize, nrole, k * dim)

        # (bsize, narg, nrole) X (bsize, nrole, k * dim)
        expected_roles = torch.bmm(role_probs, embedded_predicates) # (bsize, narg, k * dim)
        expected_roles = expected_roles.view(bsize, narg, k, dim) # (bsize, narg, k, dim)
        expected_roles = expected_roles.view(-1, k, dim) # (bsize * narg, k, dim)

        # (bsize, narg, dim)
        embedded_arguments = e_a 
        embedded_arguments = embedded_arguments.view(-1, dim).unsqueeze(-1)   
        # (bsize * narg, k, dim) X (bsize * narg, dim, 1) -> features
        args_and_roles = torch.bmm(expected_roles, embedded_arguments).squeeze(-1) 
        args_and_roles = args_and_roles.view(bsize, -1, k) # (bsize, narg, k)
        
        args_and_roles = args_and_roles * argument_mask.unsqueeze(-1).float()
        args_and_roles_sum = torch.sum(args_and_roles, 1, keepdim=True) 

        args_and_roles_sum = args_and_roles_sum + self.feate_scalar.unsqueeze(0)

        ctxs = args_and_roles_sum - args_and_roles # (bsize, narg, k)
        ctxs = ctxs.view(-1, k) # (bsize * narg, k)

        # 
        arg_lemmas = arguments["argmts"]
        arg_lemmas = arg_lemmas.unsqueeze(-1)
        #gold_scores = self.compute_potential(dim, arg_lemmas, ctxs, expected_roles) 

        args_scalar = None
        if self.loss_type=='ivan' and False:
            args_scalar = torch.gather(self.lemma_scalar, -1, arg_lemmas.view(1, -1)) 
            args_scalar = args_scalar.squeeze(0) # (bsize * narg, 1)
            args_scalar = args_scalar.view(-1, 1)
        args_scalar = None

        gold_scores = self.compute_potential_batch(dim, arg_lemmas, ctxs, expected_roles, args_scalar=args_scalar) 

        distr = self.lemma_distr.to(device=ctxs.device)

        nsample = ctxs.size(0)

        distr = distr.unsqueeze(0).expand(nsample, -1)
        samples = torch.multinomial(distr, self.nsampling, replacement=False)
        samples = samples.view(bsize, -1, self.nsampling) # (bsize, narg, nsample)
        samples = (samples + 1 + arg_lemmas) % self.nlemma
        fake_scores = self.compute_potential_batch(dim, samples, ctxs, expected_roles, args_scalar=args_scalar) 
        """
        fake_scores = []
        for i in range(nsample):
            sample = samples[:, :, i : i + 1] 
            fake_score = self.compute_potential_batch(dim, samples, ctxs, expected_roles, args_scalar=args_scalar) 
            fake_scores.append(fake_score)
        fake_scores = torch.cat(fake_scores, -1)
        """

        if (argument_mask.sum(-1) == 0).any():
            raise ValueError("Empty argument set encountered.")

        if self.loss_type == 'ivan':
            gold_scores = F.logsigmoid(gold_scores)
            fake_scores = F.logsigmoid(-fake_scores)
            loss = torch.cat([gold_scores, fake_scores], -1) 
            loss = loss * argument_mask.view(-1, 1).float()
            loss = -loss.sum() / argument_mask.sum().float()
        elif self.loss_type == 'relu':
            #gold_scores = F.logsigmoid(gold_scores)
            #fake_scores = F.logsigmoid(fake_scores)
            gold_scores = F.sigmoid(gold_scores)
            fake_scores = F.sigmoid(fake_scores)
            loss = torch.relu(1 - gold_scores + fake_scores)
            loss = loss.sum(-1) 

            loss = loss * argument_mask.view(-1).float()
            #loss /= self.nsampling
            loss = loss.view(bsize, narg)
            loss = loss.sum(-1) / argument_mask.sum(-1).float()
            loss = torch.mean(loss)
        else:
            gold_scores = F.logsigmoid(gold_scores)
            fake_scores = F.logsigmoid(fake_scores)
            loss = - gold_scores + fake_scores
            loss = loss.sum(-1) 

            loss = loss * argument_mask.view(-1).float()
            #loss /= self.nsampling
            loss = loss.view(bsize, narg)
            loss = loss.sum(-1) / argument_mask.sum(-1).float()
            loss = torch.mean(loss)

        """
        for idx in range(self.nsampling):
            #samples = torch.randint(0, self.nlemma - 1, (nsample,), device=ctxs.device)
            samples = torch.multinomial(distr, nsample, replacement=True) # weighted 
            samples = samples.view(bsize, -1)
            samples = (samples + 1 + arg_lemmas) % self.nlemma
            # scores are negative
            fake_scores = self.compute_potential(dim, samples, ctxs, expected_roles) 
        """ 
        output_dict['loss'] = loss

        return output_dict 

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.classifier.decode(output_dict)

    @overrides       
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.classifier.get_metrics(reset=reset)

    def compute_potential(self, dim: int, lemmas: torch.Tensor, 
                          ctxs: torch.Tensor, expected_roles: torch.Tensor):
        this_lemmas = {'argmts': lemmas}
        
        embedded_arguments = self.classifier.encode_lemma(this_lemmas)
        embedded_arguments = embedded_arguments.view(-1, dim).unsqueeze(-1) # (bsize * narg, dim, 1)  
        # (bsize * narg, k, dim) X (bsize * narg, dim, 1)
        args_and_roles = torch.bmm(expected_roles, embedded_arguments).squeeze(-1) 
        # (bsize * narg, 1, k) X (bsize * narg, k, 1)
        scores = torch.bmm(ctxs.unsqueeze(1), args_and_roles.unsqueeze(-1)) 
        scores = scores.squeeze(-1).squeeze(-1)

        args_scalar = torch.gather(self.lemma_scalar, -1, lemmas.view(1, -1)) 
        args_scalar = args_scalar.squeeze(0)

        scores = scores + args_scalar
        scores = F.logsigmoid(scores)
        return scores
