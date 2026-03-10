import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TRM(nn.Module):
    def __init__(self,top_m_ratio:float = 0.33, alpha: float = 2.0):
        super().__init__()
        self.top_m_ratio = top_m_ratio
        self.alpha = alpha

    def forward(self,Z):
        B, T, D = Z.shape
        norms = Z.norm(dim=-1)
        mu = norms.mean(dim=1,keepdim=True)
        sigma = norms.std(dim=1,keepdim=True)+ 1e-6
        zscore = (norms - mu) / sigma

        w = torch.sigmoid(-self.alpha * zscore)
        z_tilde = Z * w.unsqueeze(-1)
        M = max(1,int(self.top_m_ratio * T))
        topk_idx = torch.topk(w,k=M,dim=1,largest=True).indices
        mask = torch.zeros_like(w,dtype=torch.float32)
        mask.scatter(1,topk_idx,1.0)
        mask = mask >0
        return z_tilde, mask, w


class CATS(nn.Module):
    def __init__(self,dim:int,score_hidden:int =256,tau:float=0.8,beta_kl :float = 0.6, lambda_cons: float = 0.5,perturb_sigma: float = 0.03,use_dot_product :bool =True):
        super().__init__()
        self.tau = tau
        self.beta_kl = beta_kl
        self.lambda_cons = lambda_cons
        self.perturb_sigma = perturb_sigma
        self.use_dot_product = use_dot_product

        if self.use_dot_product:
            self.wx = nn.Linear(dim,dim,bias=False)
            self.wz = nn.Linear(dim,dim,bias=False)
        else:
            self.scorer = Mlp(in_features=dim *3,hidden_features=score_hidden,out_features=1,drop=0.0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _score(self,x:torch.Tensor, Z:torch.Tensor):
        B,T,D = Z.shape
        if self.use_dot_product:
            q = self.wx(x).unsqueeze(1)
            k = self.wz(Z)
            s = (q * k).sum(dim=-1) / (D **0.5)
        else:
            x_rep = x.unsqueeze(1).expand(-1,T,-1)
            feat = torch.cat([x_rep,Z,x_rep * Z],dim=-1)
            s = self.scorer(feat).squeeze(-1)
        return s

    def _masked_softmax(self,s:torch.Tensor,mask:torch.Tensor):
        s = s.float()
        very_neg = -1e9
        s_masked =  torch.where(mask,s/self.tau, torch.full_like(s, very_neg))
        s_masked = s_masked - s_masked.max(dim=1,keepdim=True).values

        pi = F.softmax(s_masked,dim=1)
        pi = pi *mask.float()
        z = pi.sum(dim=1,keepdim=True).clamp_min(1e-12)
        pi = pi / z
        return pi.to(mask.dtype if mask.dtype.is_floating_point else torch.float32)

    def _kl_to_uniform(self, pi: torch.Tensor, mask:torch.Tensor):
        pi_masked = pi *mask.float()
        M = mask.sum(dim=1).clamp_min(1)
        eps = 1e-12
        H = -(pi_masked * (pi_masked.clamp_min(eps).log())).sum(dim=1)
        kl = torch.log(M.float())-H
        return kl.mean()

    def _consistency(self,x:torch.Tensor,Z:torch.Tensor,mask:torch.Tensor,pi:torch.Tensor):
        noise = torch.zeros_like(Z).normal_(mean=0.0,std=self.perturb_sigma)
        Zp = Z +noise * mask.float().unsqueeze(-1)

        s_prime = self._score(x,Zp)
        pi_prime = self._masked_softmax(s_prime,mask)

        eps = 1e-12
        kl1 = (pi* (pi.clamp_min(eps).log()-pi_prime.clamp_min(eps).log())).sum(dim=1)
        kl2 = (pi_prime * (pi_prime.clamp_min(eps).log()-pi.clamp_min(eps).log())).sum(dim=1)
        return 0.5 * (kl1 + kl2).mean()

    def forward(self,x:torch.Tensor,Z_tlide:torch.Tensor,mask:torch.Tensor,if_compute_loss = True):
        s = self._score(x,Z_tlide)
        pi = self._masked_softmax(s,mask)
        if if_compute_loss:
            loss_kl = self._kl_to_uniform(pi,mask) *self.beta_kl
            loss_cons = self._consistency(x,Z_tlide,mask,pi) * self.lambda_cons
            return pi, loss_kl, loss_cons
        else: return pi,None,None


class GRM(nn.Module):
    def __init__(self,
                 dim:int,
                 bottleneck: int =256,
                 gate_learnable_clamp:bool = True,):
        super().__init__()
        self.proj_z = nn.Linear(dim,dim,bias=False)
        self.ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim,bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck,dim)
        )
        self.gate_affine = nn.Linear(dim*2,1)
        self.use_clamp = gate_learnable_clamp

        if self.use_clamp:
            self.alpha = nn.Parameter(torch.tensor(1.2))
            self.beta = nn.Parameter(torch.tensor(-0.1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self,x:torch.Tensor,Z_tlide:torch.Tensor,pi:torch.Tensor):
        Zp = self.proj_z(Z_tlide)
        a = torch.einsum("bt,btd->bd",pi,Zp)
        a =self.ln(a)
        a = a + 0.5 *self.mlp(a)

        gate_inp = torch.cat([x,a],dim=-1)
        g_lin = self.gate_affine(gate_inp)
        if self.use_clamp:
            gamma = self.alpha * torch.sigmoid(g_lin) +self.beta
        else:
            gamma = torch.sigmoid(g_lin)*F.silu(g_lin)

        v = x + gamma *a
        return v,a

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256,
                 trm_top_m_ratio: float = 0.33,
                 trm_alpha: float = 2.0,
                 cats_tau: float = 0.8,
                 cats_beta_kl: float = 0.05,
                 cats_lambda_cons: float = 0.5,
                 cats_perturb_sigma: float = 0.03,
                 cats_use_dot_product: bool = True,
                 grm_gate_learnable_clamp: bool = True
                 ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

        self.trm = TRM(top_m_ratio=trm_top_m_ratio, alpha=trm_alpha)
        self.cats = CATS(dim=in_dim,
                             score_hidden=bottleneck_dim,
                             tau=cats_tau,
                             beta_kl=cats_beta_kl,
                             lambda_cons=cats_lambda_cons,
                             perturb_sigma=cats_perturb_sigma,
                             use_dot_product=cats_use_dot_product,
                             )

        self.grm = GRM(dim=in_dim, bottleneck=bottleneck_dim, gate_learnable_clamp=grm_gate_learnable_clamp)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, if_images = False,if_tar = False, if_compute_loss= False):
        if if_images:
            if if_tar:
                cls = x[:, 0, :]
                Z = x[:, 1:, :]
                Z_tlide, mask, _ = self.trm(Z)
                pi, loss_kl, loss_cons = self.cats(cls, Z_tlide, mask,if_compute_loss=if_compute_loss)
                v, _ = self.grm(cls, Z_tlide, pi)
                x = v
            else:
                x = x[:, 0, :]
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x)
        return x_proj, logits, loss_kl, loss_cons,x
    


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None,device="cuda"):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x