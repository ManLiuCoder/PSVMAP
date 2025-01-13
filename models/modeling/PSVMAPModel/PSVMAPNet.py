from matplotlib.cbook import flatten
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from torch.nn.functional import dropout
from models.modeling import utils
from models.modeling.backbone_vit.vit_model import vit_base_patch16_224 as create_model
from os.path import join
import pickle
from einops.layers.torch import Rearrange
Norm =nn.LayerNorm


class PreNormResidual(nn.Module):
    def __init__(self, dim,fn,flag):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.flag =flag

    def forward(self, x):
        x_in = self.norm(x)
        if self.flag == True:
            x_in=rearrange(x_in, "b p c -> b c p")
        x_in = self.fn(x_in)
        if self.flag == True:
            x_in=rearrange(x_in, "b p c -> b c p")
        return x_in + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

class MLPMixer(nn.Module):
    def __init__(self, num_att, channels, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.,depth=3):
        super().__init__()
        chan_first, chan_last = nn.Linear, nn.Linear
        self.blocks = nn.Sequential(
            *[nn.Sequential(
            Rearrange('b a c -> b c a', a = num_att, c = channels),
            PreNormResidual(num_att,FeedForward(num_att, expansion_factor, dropout, chan_first),False),
            PreNormResidual(num_att,FeedForward(channels, expansion_factor_token, dropout, chan_last),True),
            Rearrange('b c a -> b a c', a = num_att, c = channels))for _ in range(depth)]
            ) 
    def forward(self, x):
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x[:, :, :])
        return hidden_states_out

    

def trunc_normal_(tensor, mean=0, std=.01):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

class SimpleReasoning(nn.Module):
    def __init__(self, np,ng):
        super(SimpleReasoning, self).__init__()
        self.hidden_dim= np//ng 
        self.fc1 = nn.Linear(np, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, np)
        self.avgpool = nn.AdaptiveMaxPool1d(1)
        self.act=nn.GELU()

    def forward(self, x):
        x_1 = self.fc1(self.avgpool(x).flatten(1)) 
        x_1 = self.act(x_1)
        x_1 = F.sigmoid(self.fc2(x_1)).unsqueeze(-1)
        x_1 = x_1*x + x
        return x_1

class Tokenmix(nn.Module):
    def __init__(self, np,dim):
        super(Tokenmix, self).__init__()
        # dim =196
        # hidden_dim = dim*3 #512
        if dim == 196:
            hidden_dim = 512
        elif dim == 49:
            hidden_dim = 256
        elif dim == 784:
            hidden_dim = 1024
        else:
            hidden_dim = 2048
        dropout = 0.
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.act=nn.GELU()
        self.norm = nn.LayerNorm(np)
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout))
    def forward(self, x):
        redisual =  x
        x=self.norm(x)
        x = rearrange(x, "b p c -> b c p")
        x = self.net(x)
        x = rearrange(x, "b c p-> b p c")
        out = redisual+x
        return out


class AnyAttention(nn.Module):
    def __init__(self, dim, qkv_bias=False):
        super(AnyAttention, self).__init__()
        self.norm_q, self.norm_k, self.norm_v = Norm(dim), Norm(dim), Norm(dim)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = dim ** (-0.5)
        self.act=nn.ReLU()
        self.proj = nn.Linear(dim, dim)
    def get_qkv(self, q, k, v):
        q, k, v = self.norm_q(q), self.norm_k(k), self.norm_v(v)
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        return q, k, v
    def forward(self, q=None, k=None, v=None):
        q, k, v = self.get_qkv(q, k, v)
        attn = torch.einsum("b q c, b k c -> b q k", q, k) 
        attn = self.act(attn)
        attn *= self.scale
        attn_mask = F.softmax(attn, dim=-1)
        out = torch.einsum("b q k, b k c -> b q c", attn_mask, v.float())
        out = self.proj(out)
        return attn, out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(hidden_features) or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
        bound1 = 1 / math.sqrt(fan_in1)
        nn.init.uniform_(self.fc1.bias, -bound1, bound1)
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2.weight)
        bound2 = 1 / math.sqrt(fan_in2)
        nn.init.uniform_(self.fc2.bias, -bound2, bound2)


class CMLP(nn.Module):
    # CMLP
    def __init__(self, dim, fpn_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.reduction = nn.Linear(dim, fpn_dim * 4, bias=False)
        self.fpn_dim = fpn_dim

    def forward(self, x):
        x=x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = self.norm(x)
        x = self.reduction(x)
        x = x.reshape(
            B, H, W, 2, 2, self.fpn_dim
        ).permute(0, 1, 3, 2, 4, 5).reshape(
            B, 2 * H, 2 * W, self.fpn_dim
        )
        x=x.permute(0,3,1,2)
        return x

class Block(nn.Module):
    def __init__(self, dim, v_dim,ffn_exp=4, drop_path=0.1, num_heads=1, num_parts=0,num_g=6):
        super(Block, self).__init__()
        self.dec_attn = AnyAttention(dim, True)
        self.ffn1 = Mlp(dim, hidden_features=dim * ffn_exp, act_layer=nn.GELU, norm_layer=Norm)
        self.drop_path = nn.Identity()
        self.reason = Tokenmix(dim,v_dim)
        self.enc_attn = AnyAttention(dim, True)
        self.group_compact = SimpleReasoning(num_parts,num_g)
        self.maxpool1d = nn.AdaptiveMaxPool1d(1)
        self.enc_ffn = Mlp(dim, hidden_features=dim, act_layer=nn.GELU) 


    def forward(self, x, parts=None):
        # in: x[b,768,196] parts[b,312,768]
        b,c,h,w = x.shape
        x= x.view(b,c,h*w).permute(0, 2, 1)
        attn_0,attn_out = self.enc_attn(q=parts, k=x, v=x) # q[b,312,768] kv[b,196,768]
        attn_0=self.maxpool1d(attn_0).flatten(1)
        parts1 = parts + attn_out
        parts2 = self.group_compact(parts1)
        if self.enc_ffn is not None:
            parts_out = parts2 + self.enc_ffn(parts2)+parts1
        parts_d= parts+parts_out
        attn_1,attn_out = self.enc_attn(q=parts_d, k=x, v=x)
        attn_1=self.maxpool1d(attn_1).flatten(1)
        parts1_d = parts_d + attn_out
        parts_comp = self.group_compact(parts1_d)
        if self.enc_ffn is not None:
            parts_in = parts_comp + self.enc_ffn(parts_comp)+parts1_d
        attn_mask,feats = self.dec_attn(q=x, k=parts_in, v=parts_in)
        feats = x + feats
        feats = self.reason(feats)
        feats = feats + self.ffn1(feats)
        # feats = rearrange(feats, "b p c -> b c p")
        feat_out= feats.permute(0, 2, 1).view(b,c,h,w)
        
        return feat_out,attn_0,attn_1,parts_in

class PSVMAPNet(nn.Module):
    def __init__(self, basenet, c,
                 attritube_num, cls_num, ucls_num, group_num, w2v,
                 scale, h, w, device=None):

        super(PSVMAPNet, self).__init__()
        self.attritube_num = attritube_num
        self.group_num=group_num
        self.feat_channel = c
        self.h_size = h
        self.w_size = w
        self.batch =10
        self.cls_num= cls_num
        self.ucls_num = ucls_num
        self.scls_num = cls_num - ucls_num
        self.num_branches = 3
        self.w2v_att = torch.from_numpy(w2v).float().to(device)
        self.W = nn.Parameter(trunc_normal_(torch.empty(self.w2v_att.shape[1], self.feat_channel)),
                              requires_grad=True)
        self.V = nn.Parameter(trunc_normal_(torch.empty(self.feat_channel*3, self.attritube_num)),
                              requires_grad=True)
        self.V0 = nn.Parameter(trunc_normal_(torch.empty(self.feat_channel, self.attritube_num)),
                              requires_grad=True)
        assert self.w2v_att.shape[0] == self.attritube_num
        if scale<=0:
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        else:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)


        self.backbone=basenet
        self.mlp_backbone=MLPMixer(attritube_num,c)

        self.drop_path = 0.1

        self.cls_token = basenet.cls_token
        self.pos_embed = basenet.pos_embed

        self.avgpool2d = nn.AdaptiveAvgPool2d(1)
        self.CLS_loss = nn.CrossEntropyLoss()
        self.Reg_loss = nn.MSELoss()

        self.blocks4 = Block(self.feat_channel,
                  num_heads=1,
                  num_parts=self.attritube_num,
                  num_g=self.group_num,
                  ffn_exp=4,
                  drop_path=0.1,
                  v_dim=h*w//4)
        self.blocks3 = Block(self.feat_channel,
                  num_heads=1,
                  num_parts=self.attritube_num,
                  num_g=self.group_num,
                  ffn_exp=4,
                  drop_path=0.1,
                  v_dim=h*w)
        self.blocks2 = Block(self.feat_channel,
                  num_heads=1,
                  num_parts=self.attritube_num,
                  num_g=self.group_num,
                  ffn_exp=4,
                  drop_path=0.1,
                  v_dim=h*w*4)
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(c, c, kernel_size=2, stride=2),)
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2) 
   
    def knowledge_distillation_loss(self,student_outputs, teacher_outputs, temperature,teacher_confidence):
        KLDivLoss = nn.KLDivLoss(reduction='none')
        student_probs = F.softmax(student_outputs / temperature, dim=1) 
        teacher_probs = F.softmax(teacher_outputs / temperature, dim=1)
        loss = KLDivLoss(student_probs.log(), teacher_probs)
        # return loss
        weighted_loss = loss * teacher_confidence.unsqueeze(1) 
        return weighted_loss.sum()/teacher_confidence.sum()

    def compute_KLloss(self, confidences,uncertainty,branch_outputs):
        confidences = torch.stack(confidences)
        branch_outputs = torch.stack(branch_outputs)
        uncertainty = torch.stack(uncertainty)

        teacher_confidence,teacher_indices = torch.max(confidences, dim=0) 
        student_indices = torch.argmin(confidences, dim=0) 

        teacher_output = torch.transpose(branch_outputs,0,1)[torch.arange(teacher_indices.size(0)),teacher_indices,:] 
        student_output = torch.transpose(branch_outputs,0,1)[torch.arange(student_indices.size(0)),student_indices,:]
        student_uncertainty = torch.transpose(uncertainty,0,1)[torch.arange(student_indices.size(0)),student_indices]
        loss = self.knowledge_distillation_loss(student_output, teacher_output, 10, student_uncertainty)   
        return loss

   
    def calculate_entropy(self, logits):
        probabilities = F.softmax(logits, dim=1)
        log_probabilities = torch.log(probabilities + 1e-6)  # 避免取对数时出现无穷大
        entropy = -torch.sum(probabilities * log_probabilities, dim=1)
        return entropy
    def calculate_uncertainty(self, logits):
        uncertaintyH = [self.calculate_entropy(output) for output in logits]
        return uncertaintyH

    def calculate_weightsMax(self, branch_outputs):
        certainty = [self.calculate_AbCertainty(output) for output in branch_outputs]
        weights_sum = sum(certainty)
        weights = [confidence / weights_sum for confidence in certainty]
        return certainty,weights

    def calculate_AbCertainty(self, logits):
        probabilities = F.softmax(logits, dim=1)
        max_probabilities, _ = torch.max(probabilities, dim=1)
        certainty = max_probabilities
        return certainty
    def compute_score(self, gs_feat,seen_att,att_all,flag):
        gs_feat = gs_feat.view(self.batch, -1)
        gs_feat_norm = torch.norm(gs_feat, p=2, dim = 1).unsqueeze(1).expand_as(gs_feat)
        gs_feat_normalized = gs_feat.div(gs_feat_norm + 1e-5)
        temp_norm = torch.norm(att_all, p=2, dim=1).unsqueeze(1).expand_as(att_all)
        seen_att_normalized = att_all.div(temp_norm + 1e-5)
        score_o = torch.einsum('bd,nd->bn', gs_feat_normalized, seen_att_normalized)  # [8,150]
        d, _ = seen_att.shape
        score_o = score_o*self.scale
        if d == self.cls_num:
            score = score_o
        if d == self.scls_num:
            score = score_o[:, :d]
            uu = self.ucls_num
            if self.training:
                mean1 = score_o[:, :d].mean(1)
                std1 = score_o[:, :d].std(1)
                mean2 = score_o[:, -uu:].mean(1)
                std2 = score_o[:, -uu:].std(1)
                mean_score = F.relu(mean1 - mean2)
                std_score = F.relu(std1 - std2)
                mean_loss = mean_score.mean(0) + std_score.mean(0)
                if flag == True:
                    return score, mean_loss
                else:
                    return score
        if d == self.ucls_num:
            score = score_o[:, -d:]
        if flag == True:
            return score, _
        else:
            return score
   
    
    def forward(self, x, att=None, label=None, seen_att=None, att_all=None):
        scores_outputs =[]
        f_out =[]
        feats_out =[]
        t_out=[]
        
        self.batch = x.shape[0]
        parts1 = torch.einsum('lw,wv->lv', self.w2v_att, self.W)
        parts1 = parts1.expand(self.batch, -1, -1)

        parts_ori = self.mlp_backbone(parts1)
        parts2 = parts_ori[0]
        parts3 = parts_ori[1]
        parts4 = parts_ori[2]

        x4, hidden_states_out = self.backbone(x)
        x4 = x4.permute(0, 2, 1).reshape(self.batch, -1, self.h_size, self.w_size)
        x2 = hidden_states_out[8].permute(0, 2, 1).reshape(self.batch, -1, self.h_size, self.w_size)
        x3 = hidden_states_out[9].permute(0, 2, 1).reshape(self.batch, -1, self.h_size, self.w_size)

        feats2 = self.fpn2(x2) 
        feats3 = self.fpn3(x3)   
        feats4 = self.fpn4(x4)  

        feats2_out, att2_0, att2_1,parts2_out = self.blocks2(feats2,parts=parts2)
        feats_out.append(feats2_out)
        feats3_out, att3_0, att3_1,parts3_out = self.blocks3(feats3,parts=parts3) 
        feats_out.append(feats3_out)
        feats4_out, att4_0, att4_1,parts4_out = self.blocks4(feats4,parts=parts4) 
        feats_out.append(feats4_out)

        f_out = [self.avgpool2d(fin).view(self.batch, -1)for fin in feats_out]
        t_out = [torch.einsum('bc,cd->bd', finput, self.V0) for finput in f_out]
        scores_outputs = [self.compute_score(input, seen_att, att_all,False) for input in t_out]
        
    
        certainty,weights = self.calculate_weightsMax(scores_outputs)
        uncertainty = self.calculate_uncertainty(scores_outputs)
        fused_output = torch.zeros_like(f_out[0])
        for i in range(self.num_branches):
            fused_output += torch.unsqueeze(weights[i],dim=-1) * f_out[i]
        
        out = torch.einsum('bc,cd->bd', fused_output, self.V0)
        score, b = self.compute_score(out, seen_att, att_all,True)
        if not self.training:
            return score
        
        Lkl= self.compute_KLloss(certainty,uncertainty,scores_outputs)


        Lreg1 = self.Reg_loss(att4_0, att)+self.Reg_loss(att4_1, att)+self.Reg_loss(att3_0, att)+self.Reg_loss(att3_1, att)+self.Reg_loss(att2_0, att)+self.Reg_loss(att2_1, att)
        Lcls = self.CLS_loss(score, label)
        scale = self.scale.item()
        loss_dict = {
            'Reg_loss': Lreg1,
            'Cls_loss': Lcls,
            'scale': scale,
            'bias_loss': b,
            'kl_loss': Lkl
        }

        return loss_dict


def build_PSVMAPNet(cfg):
    dataset_name = cfg.DATASETS.NAME
    info = utils.get_attributes_info(dataset_name)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]
    group_num = info["g"] 
    c,w,h = 768, 14, 14
    scale = cfg.MODEL.SCALE
    vit_model = create_model(num_classes=-1)
    # load pretrain model
    vit_model_path = "...pth"
    weights_dict = torch.load(vit_model_path)
    del_keys = ['head.weight', 'head.bias'] if vit_model.has_logits \
        else ['head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    vit_model.load_state_dict(weights_dict, strict=False)
    w2v_file = dataset_name+"_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device(cfg.MODEL.DEVICE)

    return PSVMAPNet(basenet=vit_model,
                  c=c,scale=scale,
                  attritube_num=attritube_num,
                  group_num=group_num, w2v=w2v,
                  cls_num=cls_num, ucls_num=ucls_num,h=h,w=w,
                  device=device)