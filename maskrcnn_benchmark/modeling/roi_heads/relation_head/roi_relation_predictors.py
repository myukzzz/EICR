# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F


from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from SHA_GCL_extra.kl_divergence import KL_divergence
from .model_Hybrid_Attention import SHA_Context
from .model_Cross_Attention import CA_Context
from .utils_relation import layer_init
from maskrcnn_benchmark.data import get_dataset_statistics

from SHA_GCL_extra.utils_funcion import FrequencyBias_GCL
from SHA_GCL_extra.extra_function_utils import generate_num_stage_vector, generate_sample_rate_vector, \
    generate_current_sequence_for_bias, get_current_predicate_idx,generate_weight_rate_vector
from SHA_GCL_extra.group_chosen_function import get_group_splits
import random


#DenseCLIP
#from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
#from denseclip.untils import tokenize

from torch.nn import init
import math
from torch.nn.parameter import Parameter
#from .bdc_module import *

#ECCV22
import pickle
from copy import deepcopy
import json

from .utils_motifs import load_word_vectors


#TSNE
import matplotlib.pyplot as plt
from sklearn import manifold
import pandas as pd

colors = ['black', 'tomato', 'yellow', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
          'hotpink']  # 设置散点颜色
Label_Com = ['S-1', 'T-1', 'S-2', 'T-2', 'S-3','T-3',
             'S-4', 'T-4', 'S-5', 'T-5', 'S-6', 'T-6',
             'S-7', 'T-7', 'S-8', 'T-8', 'S-9', 'T-9',
             'S-10', 'T-10', 'S-11', 'T-11', 'S-12', 'T-12']  ##图例名称

### 设置字体格式
font1 = {'family': 'Times New Roman',

         'weight': 'bold',
         'size': 8,
         }
def visual(X):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    # '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    return X_norm
def plot_with_labels(S_lowDWeights, Trure_labels, name):
    plt.cla()  # 清除当前图形中的当前活动轴,所以可以重复利用

    # 降到二维了，分别给x和y
    True_labels = Trure_labels.reshape((-1, 1))

    S_data = np.hstack((S_lowDWeights, True_labels))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})

    for index in range(10):
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=20, marker='o', c=colors[index], alpha=0.65)

    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    #
    plt.title(name, fontsize=32, fontweight='normal', pad=20)


def sample_rate(class_stat):
    # if Dataset_choice == 'VG':
    #     predicate_new_order_count = [3024465, 109355, 67144, 47326, 31347, 21748, 15300, 10011, 11059, 10764, 6712,
    #                                  5086, 4810, 3757, 4260, 3167, 2273, 1829, 1603, 1413, 1225, 793, 809, 676, 352,
    #                                  663, 752, 565, 504, 644, 601, 551, 460, 394, 379, 397, 429, 364, 333, 299, 270,
    #                                  234, 171, 208, 163, 157, 151, 71, 114, 44, 4]
    #     assert len(predicate_new_order_count) == 51
    # elif Dataset_choice == 'GQA_200':
    #     predicate_new_order_count = [200000, 64218, 47205, 32126, 25203, 21104, 15890, 15676, 7688, 6966, 6596, 6044, 5250, 4260, 4180, 4131, 2859, 2559, 2368, 2351, 2134, 1673, 1532, 1373, 1273, 1175, 1139, 1123, 1077, 941, 916, 849, 835, 808, 782, 767, 628, 603, 569, 540, 494, 416, 412, 412, 398, 395, 394, 390, 345, 327, 302, 301, 292, 275, 270, 267, 267, 264, 258, 251, 233, 233, 229, 224, 215, 214, 209, 204, 198, 195, 192, 191, 185, 181, 176, 158, 158, 154, 151, 148, 143, 136, 131, 130, 130, 128, 127, 125, 124, 124, 121, 118, 112, 112, 106, 105, 104, 103, 102, 52, 52]
    #     assert len(predicate_new_order_count) == 101
    # else:
    #     exit('wrong mode in Dataset_choice')
    outp = []
    median = np.median(class_stat[1:])
    for j in range(len(class_stat)):
            outp.append(0.0)
    for j in range(len(class_stat)):
        if class_stat[j] > median or j == 0:
            num = median / class_stat[j]
            if j == 0:
                num = 0.01
            if num < 0.01:
                num = 0.01
            outp[j] = num
        else:
            outp[j] = 1.0
    # for i in range(len(num_stage_predicate)):
    #     opiece = []
    #     for j in range(len(predicate_new_order_count)):
    #         opiece.append(0.0)
    #     num_list = predicate_new_order_count[0:(num_stage_predicate[i]+1)]
    #     median = np.median(num_list[1:])
    #     for j in range(len(num_list)):
    #         if num_list[j] > median:
    #             num = median / num_list[j]
    #             if j == 0:
    #                 num = num * 10.0
    #             if num < 0.01:
    #                 num = 0.01
    #             opiece[j] = num
    #         else:
    #             opiece[j] = 1.0
    #     outp.append(opiece)
    return outp

def weight_rate(class_stat):
    # if Dataset_choice == 'VG':
    #     predicate_new_order_count = [3024465, 109355, 67144, 47326, 31347, 21748, 15300, 10011, 11059, 10764, 6712,
    #                                  5086, 4810, 3757, 4260, 3167, 2273, 1829, 1603, 1413, 1225, 793, 809, 676, 352,
    #                                  663, 752, 565, 504, 644, 601, 551, 460, 394, 379, 397, 429, 364, 333, 299, 270,
    #                                  234, 171, 208, 163, 157, 151, 71, 114, 44, 4]
    #     assert len(predicate_new_order_count) == 51
    # elif Dataset_choice == 'GQA_200':
    #     predicate_new_order_count = [200000, 64218, 47205, 32126, 25203, 21104, 15890, 15676, 7688, 6966, 6596, 6044, 5250, 4260, 4180, 4131, 2859, 2559, 2368, 2351, 2134, 1673, 1532, 1373, 1273, 1175, 1139, 1123, 1077, 941, 916, 849, 835, 808, 782, 767, 628, 603, 569, 540, 494, 416, 412, 412, 398, 395, 394, 390, 345, 327, 302, 301, 292, 275, 270, 267, 267, 264, 258, 251, 233, 233, 229, 224, 215, 214, 209, 204, 198, 195, 192, 191, 185, 181, 176, 158, 158, 154, 151, 148, 143, 136, 131, 130, 130, 128, 127, 125, 124, 124, 121, 118, 112, 112, 106, 105, 104, 103, 102, 52, 52]
    #     assert len(predicate_new_order_count) == 101
    # else:
    #     exit('wrong mode in Dataset_choice')
    outp = []
    median = np.median(class_stat[1:])
    for j in range(len(class_stat)):
            outp.append(0.0)
    for j in range(len(class_stat)):
        #if class_stat[j] > median or j == 0:
            num = median / class_stat[j]
            if j == 0:
                num = 0.01
            if num < 0.01:
                num = 0.01
            outp[j] = num
        #else:
            #outp[j] = 1.0
    # for i in range(len(num_stage_predicate)):
    #     opiece = []
    #     for j in range(len(predicate_new_order_count)):
    #         opiece.append(0.0)
    #     num_list = predicate_new_order_count[0:(num_stage_predicate[i]+1)]
    #     median = np.median(num_list[1:])
    #     for j in range(len(num_list)):
    #         if num_list[j] > median:
    #             num = median / num_list[j]
    #             if j == 0:
    #                 num = num * 10.0
    #             if num < 0.01:
    #                 num = 0.01
    #             opiece[j] = num
    #         else:
    #             opiece[j] = 1.0
    #     outp.append(opiece)
    return outp



def binary_ce(a_v, p_v, logit):
    if logit:
        p_v = p_v.exp() / (1 + p_v.exp())
    return -(a_v * p_v.log() + (1-a_v) * (1 - p_v).log())

def binary_crossentropy(A, P, logit=False):
    return torch.mean([binary_ce(a_i, p_i, logit) for a_i, p_i in zip(A, P)])


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class xERMLoss(nn.Module):
    def __init__(self, gamma=1):
        super(xERMLoss, self).__init__()
        self.XE_loss = nn.CrossEntropyLoss(reduction='none')
        self.gamma = gamma

    #def forward(self, logit_CF, logits_F, labels):
    # def forward(self, logit_CF, logit_F, labels):
    #     # calculate w_cf
    #     logits_CF = logit_CF.clone().detach()
    #     XE_CF = self.XE_loss(logits_CF, labels)
    #     #XE_F = self.XE_loss(logits_F, labels)
    #     #logits_F = logits_F.clone().detach()
    #     logits_F = logit_F.clone().detach()
    #     XE_F = self.XE_loss(logits_F, labels)
    #
    #     XE_CF = torch.pow(XE_CF, self.gamma)
    #     XE_F = torch.pow(XE_F, self.gamma)
    #
    #     w_cf = XE_CF/(XE_CF + XE_F + 1e-5)
    #     w_f = 1 - w_cf
    #     # factual loss
    #     #loss_F = self.XE_loss(logits_F, labels)
    #     loss_F = self.XE_loss(logit_F, labels)
    #     # counterfacutal loss
    #     prob_CF = F.softmax(logits_CF, -1).clone().detach()
    #     #prob_FCF = F.softmax(logits_F, -1)
    #     prob_FCF = F.softmax(logit_F, -1)
    #     loss_CF = - prob_CF * prob_FCF.log()
    #     loss_CF = loss_CF.sum(1)
    #
    #     loss = (w_cf*loss_CF).mean() + (w_f*loss_F).mean()
    #
    #     return loss

    def forward(self, logit_CF, logits_F, labels):
            # calculate w_cf
            logits_CF = logit_CF.clone().detach()
            XE_CF = self.XE_loss(logits_CF, labels)

            logits_F = logits_F.clone().detach()
            XE_F = self.XE_loss(logits_F, labels)

            XE_CF = torch.pow(XE_CF, self.gamma)
            XE_F = torch.pow(XE_F, self.gamma)

            w_cf = XE_CF / (XE_CF + XE_F + 1e-5)
            w_f = 1 - w_cf
            # factual loss
            loss_F = self.XE_loss(logits_F, labels)

            # counterfacutal loss
            prob_CF = F.softmax(logits_CF, -1).clone().detach()
            prob_FCF = F.softmax(logits_F, -1)

            loss_CF = - prob_CF * prob_FCF.log()
            loss_CF = loss_CF.sum(1)

            loss = (w_cf * loss_CF).mean() + (w_f * loss_F).mean()


            return loss


class ReweightingCE(nn.Module):
    """
    Given a soft label, choose the class with max class as the GT.
    converting label like [0.1, 0.3, 0.6] to [0, 0, 1] and apply CrossEntropy
    """
    def __init__(self, reduction="mean"):
        super(ReweightingCE, self).__init__()
        self.reduction=reduction

    def forward(self, input, target,weights):
        """
        Args:
            input: the prediction
            target: [N, N_classes]. For each slice [weight, 0, 0, 1, 0, ...]
                we need to extract weight.
        Returns:

        """
        # final_target = torch.zeros_like(target)
        # final_target[torch.arange(0, target.size(0)), target.argmax(1)] = 1.
        # idxs = (target[:, 0] != 1).nonzero().squeeze()
        # weights = torch.ones_like(target[:, 0])
        # weights[idxs] = -target[:, 0][idxs]
        # target = final_target
        x = F.log_softmax(input, 1)
        target_onehot = torch.FloatTensor(target.size(0), 51).cuda()
        #GQA200
        #target_onehot=torch.FloatTensor(target.size(0),101).cuda()
        target_onehot.zero_()
        target_onehot.scatter_(1,target.view(-1,1),1)
        loss = torch.sum(- x * target_onehot, dim=1)*weights
        #loss = torch.sum(- x * target_onehot, dim=1)
        #loss=loss*weights
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')


class SoftTriple(nn.Module):
    def __init__(self, la, gamma, tau, margin, dim, cN, K):
        super(SoftTriple, self).__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = Parameter(torch.Tensor(dim, cN*K))
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).cuda()
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target,weights=None):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        if weights==None:
            lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        else:
            x = F.log_softmax(self.la*(simClass-marginM), 1)
            target_onehot=torch.FloatTensor(target.size(0),51).cuda()
            target_onehot.zero_()
            target_onehot.scatter_(1,target.view(-1,1),1)
            lossClassify = torch.sum(- x * target_onehot, dim=1)*weights
            lossClassify = torch.mean(lossClassify)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim

    def forward(self, x):
        B, N, C = x.shape
        # if N>1:
        x = x.permute(1, 0, 2)  # NCHW -> (HW)NC
        # x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        # cls_pos = self.positional_embedding[0:1, :]
        # spatial_pos = F.interpolate(self.positional_embedding[1:,].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        # spatial_pos = self.positional_embedding[1:].reshape(self.spacial_dim, self.embed_dim)[:N]
        spatial_pos = self.positional_embedding.reshape(self.spacial_dim, self.embed_dim)[:N]
        spatial_pos = spatial_pos.reshape(-1, self.embed_dim)
        # spatial_pos = spatial_pos.reshape(self.embed_dim, H*W).permute(1, 0)
        # positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

        positional_embedding = spatial_pos

        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        feature_map = x.permute(1, 2, 0)
        # global_feat = x[:, :, 0]
        #feature_map = x[:, :, :].reshape(B, -1, N)
        # else:
        #     v = self.v_proj(x)
        #     x = self.c_proj(v)
        #     feature_map = x.permute(0, 2, 1)
        # return global_feat, feature_map
        return feature_map

from .utils_motifs import  obj_edge_vectors
from maskrcnn_benchmark.modeling.make_layers import make_fc
from .utils_motifs import to_onehot,encode_box_info,nms_overlaps
@registry.ROI_RELATION_PREDICTOR.register("PrototypeEmbeddingNetwork")
class PrototypeEmbeddingNetwork(nn.Module):
    def __init__(self, config, in_channels):
        super(PrototypeEmbeddingNetwork, self).__init__()
        self.criterion_loss = nn.CrossEntropyLoss()
        self.nms_thresh = 0.7
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES

        self.cfg = config

        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        statistics = get_dataset_statistics(config)

        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        #assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.mlp_dim = 2048  # config.MODEL.ROI_RELATION_HEAD.PENET_MLP_DIM
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)

        self.embed_dim = 300  # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        dropout_p = 0.2  # config.MODEL.ROI_RELATION_HEAD.PENET_DROPOUT

        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR,
                                          wv_dim=self.embed_dim)  # load Glove for objects
        # rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR,
        #                              wv_dim=self.embed_dim)  # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            #self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)

        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_obj = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim * 2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        ])

        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2)

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)

        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)

        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)

        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels


        #sgcl&sgdet
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes)
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        self.direct_relation = True
        # if self.direct_relation == True:
        self.use_cosdis = True
        if self.use_cosdis == False:
            self.rel_compress = nn.Linear(self.mlp_dim, self.num_rel_cls, bias=True)

        # union_relation
        self.union_ce = False
        self.use_protoandce = False
        if self.union_ce == True:
            self.rel_cls_ce = nn.Linear(self.mlp_dim * 2, self.num_rel_cls, bias=True)

        # FCC
        self.use_FCC = False
        self.FCC_weight = 0.1

        # text_emb
        self.use_emb = True

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

    def fcc(self, feature, label, num_classes, FCC_weight):
        """
        Feature clusters compression (FCC)
        gamma: the hyper-parameter for setting scaling factor tau.
        c_type: compression type:
                    'edc' is equal difference compression.
        """

        batch_size = feature.shape[0]

        # compressing feature

        new_features = self.equal_diff_compress(batch_size, feature, label, num_classes, FCC_weight)

        return new_features

    def equal_diff_compress(self, n, feature, label, num_classes, gamma):
        # setting scaling factor tau
        tau = []
        for k in range(num_classes):
            tau.append(round((1 + gamma - k * (gamma / num_classes)), 2))

        raw_shape = feature.shape

        tau_batch = []
        for j in range(n):
            tau_batch.append(tau[label[j]])
        tau_batch = torch.tensor(tau_batch).cuda()

        tau_batch = tau_batch.view(n, 1)
        feature = feature.view(n, -1)

        new_features = torch.mul(feature, tau_batch)
        new_features = new_features.view(raw_shape)

        return new_features

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):

        add_losses = {}
        add_data = {}

        # refine object labels
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)
        #####
        # 4096-4096——直接从Roi_feat上采主宾feat
        entity_rep = self.post_emb(roi_features)  # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)  # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)  # xo

        entity_embeds = self.obj_embed(entity_preds)  # obtaining the word embedding of entities with GloVe

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []

        for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed, proposal in zip(rel_pair_idxs, sub_reps, obj_reps,
                                                                                   entity_preds, entity_embeds,
                                                                                   proposals):
            # 300-1024-2048
            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  # Ws x ts
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  # Wo x to
            # 2048-4096-2048
            sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)
            sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)
            # 4096-2048
            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go

            if self.use_emb:
                sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
                obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo
            else:
                sub = sem_sub
                obj = sem_obj

            ##### for the model convergence——2048-2048
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            #####

            fusion_so.append(fusion_func(sub, obj))  # F(s, o)
            # fusion_so.append(fusion_func_my(sub, obj))  # F(s, o)
            pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1))

        fusion_so = cat(fusion_so, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        # rel_embeds_1 = self.rel_embed(rel_labels)
        # predicate_proto = self.W_pred(rel_embeds_1)

        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp
        # fusion_so-B,2048
        rel_rep = fusion_so - sem_pred * gate_sem_pred  # F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes

        ##### for the model convergence——2048-2048
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)
        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))  # updim到4096同维度2048-2048-4096

        if self.union_ce == False:
            predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
            ######

            rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
            predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

            ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
            rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  # <r_norm, c_norm> / τ
            # the rel_dists will be used to calculate the Le_sim with the ce_loss

            if self.use_cosdis == False:
                fusion_so_1 = fusion_so
                rel_dists_ce = self.rel_compress(fusion_so_1)

            entity_dists = entity_dists.split(num_objs, dim=0)
            rel_dists = rel_dists.split(num_rels, dim=0)  # rel_dists B,51

            if self.use_cosdis == False:
                rel_dists_ce = rel_dists_ce.split(num_rels, dim=0)

        else:
            if self.use_protoandce:
                rel_union_ce = rel_rep
                rel_dists_ce = self.rel_cls_ce(rel_union_ce)
                rel_dists_ce = rel_dists_ce.split(num_rels, dim=0)
                entity_dists = entity_dists.split(num_objs, dim=0)

                predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
                rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
                predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm
                rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()
                rel_dists = rel_dists.split(num_rels, dim=0)

            else:
                rel_union_ce = rel_rep
                rel_dists = self.rel_cls_ce(rel_union_ce)
                rel_dists = rel_dists.split(num_rels, dim=0)
                entity_dists = entity_dists.split(num_objs, dim=0)

        if self.training:

            relation_logits = cat(rel_dists, dim=0)
            refine_obj_logits = cat(entity_dists, dim=0)

            fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            rel_labels = cat(rel_labels, dim=0)

            add_losses['relation_loss'] = self.criterion_loss(relation_logits, rel_labels.long())



            add_losses['obj_loss'] = self.criterion_loss(refine_obj_logits, fg_labels.long())


            ### Prototype Regularization  ---- cosine similarity
            if self.direct_relation == False:
                target_rpredicate_proto_norm = predicate_proto_norm.clone().detach()
                simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
                l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51 * 51)
                add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
            ### end

            ### Prototype Regularization  ---- Euclidean distance
            if self.direct_relation == False:
                gamma2 = 7.0
                predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, 51, -1)
                predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(51, -1, -1)
                proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(
                    dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
                sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
                topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1  # obtain d-, where k2 = 1
                dist_loss = torch.max(torch.zeros(51).cuda(),
                                      -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
                add_losses.update({"dist_loss2": dist_loss})
            ### end

            ###  Prototype-based Learning  ---- Euclidean distance
            if self.direct_relation == False:
                rel_labels = cat(rel_labels, dim=0)
                gamma1 = 1.0
                rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, 51, -1)  # r
                predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
                distance_set = (rel_rep_expand - predicate_proto_expand).norm(
                    dim=2) ** 2  # Distance Set G, gi = ||r-ci||_2^2
                mask_neg = torch.ones(rel_labels.size(0), 51).cuda()
                mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
                distance_set_neg = distance_set * mask_neg
                distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
                sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
                topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(
                    dim=1) / 10  # obtaining g-, where k1 = 10,
                loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(),
                                     distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
                add_losses.update({"loss_dis": loss_sum})  # Le_euc = max(0, (g+) - (g-) + gamma1)
            ### end
            # return entity_dists, rel_dists, add_losses, add_data,rel_dists_ce
            if self.use_protoandce:
                return entity_dists, rel_dists, add_losses, add_data, rel_dists_ce
            return None, None, add_losses

        return entity_dists, rel_dists, add_losses

        # return entity_dists, rel_dists_ce, add_losses, add_data

    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        #pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()

        # return obj_dists, obj_preds
        if self.training:
            return obj_dists, obj_labels.long()
        else:
            return obj_dists, obj_preds

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def fusion_func(x, y):
    return F.relu(x + y) - (x - y) ** 2









@registry.ROI_RELATION_PREDICTOR.register("EICR_model")
class EICR_model(nn.Module):
    def __init__(self, config, in_channels):
        super(EICR_model, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.GLOBAL_SETTING.USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']




        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Motifs':
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        else:
            if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
                self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

            else:
                if config.GLOBAL_SETTING.BASIC_ENCODER == 'VCTree':
                    self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
                #exit('wrong mode!')
        self.base_encoder = config.GLOBAL_SETTING.BASIC_ENCODER
        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)
        if self.base_encoder=='VCTree':
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        else:
            self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)


        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # get model configs
        self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        # generate the auxiliary lists
        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, config.GLOBAL_SETTING.DATASET_CHOICE)
        self.sample_rate_matrix = generate_sample_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE, self.max_group_element_number_list)


        #over-env
        self.weight_rate_matrix = generate_weight_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE,
                                                              self.max_group_element_number_list)



        self.num_groups = len(self.max_elemnt_list)
        self.CE_loss = nn.CrossEntropyLoss()


        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)



        self.rel_criterion_loss = ReweightingCE()
        self.criterion_loss = nn.CrossEntropyLoss()


        #3env
        self.overweighted_weight = 1
        self.overweighted_weight_p = 1

        self.norm_weight= 1
        self.norm_weight_p = 1

        self.weighted_weight = 1
        self.weighted_weight_p = 1

        self.iter = 0
        self.iter_2 = 0
        self.max_iter = 120000

        self.adjust_env = True


        self.dataset_choice=config.GLOBAL_SETTING.DATASET_CHOICE
        self.rel_classifer_all = self.generate_muti_networks(self.num_groups)


        self.use3env = True
        self.use_vision = True
        self.only_vision = False

        #IRM
        self.penalty_v1 = False
        self.penalty_v2 = False
        self.penalty_v2_weight = 2



        #causal_reweight
        self.use_causal= False
        self.causal_weight = 0.4
        self.qhat = self.initial_qhat(class_num=51)
        self.qhat_proto = self.initial_qhat(class_num=51)


        self.useFCC= False
        self.FCC_weight = 0.75


        #addsem
        self.addsem= False
        if self.addsem:
            dropout_p = 0.2
            emb_dim=300

            self.vis2sem = nn.Sequential(*[
                nn.Linear(self.hidden_dim, self.hidden_dim*2), nn.ReLU(True),
                nn.Dropout(dropout_p), nn.Linear(self.hidden_dim*2, self.hidden_dim)
            ])

            self.linear_sub = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.linear_obj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.linear_rel_rep = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.dropout_sub = nn.Dropout(dropout_p)
            self.dropout_obj = nn.Dropout(dropout_p)
            self.dropout_rel_rep = nn.Dropout(dropout_p)
            self.dropout_rel = nn.Dropout(dropout_p)
            self.dropout_pred = nn.Dropout(dropout_p)

            self.norm_sub = nn.LayerNorm(self.hidden_dim)
            self.norm_obj = nn.LayerNorm(self.hidden_dim)
            self.norm_rel_rep = nn.LayerNorm(self.hidden_dim)

            self.sem_feat = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

            self.down_samp = MLP(self.pooling_dim, self.hidden_dim, self.hidden_dim, 2)

            self.gate_pred = nn.Linear(self.hidden_dim  * 3, self.hidden_dim )

            self.rel_embed = nn.Embedding(self.num_rel_cls, emb_dim)
            self.W_pred = MLP(emb_dim, self.hidden_dim // 2, self.hidden_dim, 2)

            self.post_cat_sem = nn.Linear(self.hidden_dim, self.pooling_dim)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        #addproto
        self.addproto= False
        self.catprep = False
        self.use_bias = True
        if self.addproto:
            self.gate_pred = nn.Linear(self.pooling_dim * 2, self.pooling_dim)
            self.rel_embed = nn.Embedding(51, 300)

            self.W_pred = MLP(300, 1024, 2048, 2)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.project_head = MLP(2048, 2048, 4096, 2)
            self.dropout_pred = nn.Dropout(0.2)

        self.intra_class = False
        if self.intra_class:

            f = open("VGclasses.txt", "r")
            class_names = []
            class_list = f.readlines()
            for classes in class_list:
                classes = classes.strip('\n')
                class_names.append(classes)
            self.class_names = class_names

            f_r = open("VGrelationclasses.txt", "r")
            class_names_r = []
            class_list_r = f_r.readlines()
            for classes_r in class_list_r:
                classes_r = classes_r.strip('\n')
                class_names_r.append(classes_r)
            self.class_names_r = class_names_r

            rel_cnt_dic = {}
            path = "em_E.pk"
            l = pickle.load(open(path, "rb"))
            vocab = json.load(open("VG-SGG-dicts-with-attri.json", "r"))
            idx2lb = {int(k): v for k, v in vocab["idx_to_label"].items()}
            idx2pred = {int(k) - 1: v for k, v in vocab["idx_to_predicate"].items()}
            for i, data in enumerate(l):
                labels = data["labels"]
                logits = data["logits"][:, 1:]
                relation_tuple = deepcopy(data["relations"])
                sub_idxs, obj_idxs, rels = relation_tuple[:, 0], relation_tuple[:, 1], relation_tuple[:, 2]
                sub_lbs, obj_lbs = labels[sub_idxs], labels[obj_idxs]
                # [[sub_lb1, obj_lb1], [sub_lb2, obj_lb2]......]
                pairs = np.stack([sub_lbs, obj_lbs], 1).tolist()
                pairs = [(idx2lb[p[0]], idx2lb[p[1]]) for p in pairs]
                # behave as indexes, so -=1
                rels -= 1

                # fill in rel_dic
                # rel_dic: {rel_i: {pair_j: distribution} }
                for j, (pair, r, logit) in enumerate(zip(pairs, rels, logits)):
                    r_name = idx2pred[int(r)]
                    if r_name not in rel_cnt_dic:
                        rel_cnt_dic[r_name] = {}
                    if pair not in rel_cnt_dic[r_name]:
                        rel_cnt_dic[r_name][pair] = 0
                    rel_cnt_dic[r_name][pair] += 1
            self.importance_dic = {}
            self.importance_dic_weight = {}
            for r, pair_cnt_dic in rel_cnt_dic.items():
                arr_value = np.stack([list(pair_cnt_dic.values())])
                pair_med = np.median(arr_value, 1)[0]
                self.importance_dic[r]= {}
                self.importance_dic_weight[r] = {}

                for pair in pair_cnt_dic:
                    cnt = pair_cnt_dic[pair]
                    self.importance_dic_weight[r][pair] = pair_med / cnt
                    if cnt > pair_med:
                        self.importance_dic[r][pair] = pair_med / cnt
                    else:
                        self.importance_dic[r][pair] = 1.0
                    if self.importance_dic[r][pair] < 0.01:
                        self.importance_dic[r][pair] = 0.01
                    if self.importance_dic_weight[r][pair] < 0.01:
                        self.importance_dic_weight[r][pair] = 0.01

        aaaaaa=0


        '''
        torch.int64
        torch.float16
        '''

    def initial_qhat(self,class_num=1000):
        # initialize qhat of predictions (probability)
        qhat = (torch.ones([1, class_num], dtype=torch.float) / class_num).cuda()
        print("qhat size: ".format(qhat.size()))
        return qhat
    def update_qhat(self,probs, qhat, momentum, qhat_mask=None):
        if qhat_mask is not None:
            mean_prob = probs.detach() * qhat_mask.detach().unsqueeze(dim=-1)
        else:
            mean_prob = probs.detach().mean(dim=0)
        qhat = momentum * qhat + (1 - momentum) * mean_prob
        return qhat

    def penalty(self,logits, y, weights=None):
        scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
        logits_f=logits*scale
        loss = self.CE_loss(logits_f, y)
        if weights != None:
            x = F.log_softmax(logits_f, 1)
            if self.dataset_choice== 'VG':
                target_onehot = torch.FloatTensor(y.size(0), 51).cuda()
            else:
                target_onehot=torch.FloatTensor(y.size(0),101).cuda()
            target_onehot.zero_()
            target_onehot.scatter_(1,y.view(-1,1),1)
            celoss = torch.sum(- x * target_onehot, dim=1)*weights
            celoss=torch.mean(celoss)
            grad = torch.autograd.grad(celoss, [scale], create_graph=True)[0]
        else:
            grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]

        fin_loss=torch.sum(grad ** 2)
        return fin_loss


    def fcc(self,feature, label, num_classes, FCC_weight=0.1):
        """
        Feature clusters compression (FCC)
        gamma: the hyper-parameter for setting scaling factor tau.
        c_type: compression type:
                    'edc' is equal difference compression.
        """


        batch_size = feature.shape[0]

        # compressing feature

        new_features = self.equal_diff_compress(batch_size, feature, label, num_classes, FCC_weight)



        return new_features

    def equal_diff_compress(self,n, feature, label, num_classes, gamma):
        # setting scaling factor tau
        tau = []
        for k in range(num_classes):
            tau.append(round((1 + gamma - k * (gamma / num_classes)), 2))

        raw_shape = feature.shape

        tau_batch = []
        for j in range(n):
            tau_batch.append(tau[label[j]])
        tau_batch = torch.tensor(tau_batch).cuda()

        tau_batch = tau_batch.view(n, 1)
        feature = feature.view(n, -1)

        new_features = torch.mul(feature, tau_batch)
        new_features = new_features.view(raw_shape)

        return new_features



    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        self.iter_2 += 1
        if self.iter_2==99:
            print("*******************************************************************************************")

        # reweight
        if self.adjust_env:
            T = 30000


            if self.iter_2 > T:
                self.iter += 1
            #
            alpha = 1-(self.iter / T)
            alpha = max(alpha, 0.1)
            alpha = alpha*0.9
            if self.iter_2 <= T:
                alpha = 0.9

            self.norm_weight=alpha
            self.norm_weight_p=alpha

            self.overweighted_weight=1-alpha
            self.overweighted_weight_p=1-alpha


        #trans



###################################################################################################

##################################################################################


        #######################







        ############



        if self.base_encoder == 'Motifs':
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)
        else:
            if self.base_encoder == 'Self-Attention':
                obj_dists, obj_preds, edge_ctx= self.context_layer(roi_features, proposals, logger)
            else:
                if self.base_encoder == 'VCTree':
                    obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals,
                                                                                      rel_pair_idxs, logger)

        # post decode
        if self.base_encoder=='VCTree':
            edge_rep = F.relu(self.post_emb(edge_ctx))
        else:
            edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)


        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        if self.addsem:
            sem_head = self.vis2sem(head_rep)
            sem_tail = self.vis2sem(tail_rep)
            sem_head = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sem_head))) + sem_head)
            sem_tail = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(sem_tail))) + sem_tail)
            sem_heads = sem_head.split(num_objs, dim=0)
            sem_tails = sem_tail.split(num_objs, dim=0)

            prod_reps_sem = []
            for pair_idx, head_rep, tail_rep in zip(rel_pair_idxs, sem_heads, sem_tails):
                prod_reps_sem.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            prod_rep_sem = cat(prod_reps_sem, dim=0)

            sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
            gate_sem_pred = torch.sigmoid(self.gate_pred(cat((prod_rep_sem, sem_pred), dim=-1)))  # g
            # fusion_so-B,2048
            rel_rep = self.sem_feat(prod_rep_sem) + sem_pred * gate_sem_pred  # F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
            predicate_proto = self.W_pred(self.rel_embed.weight)


            rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)
            rel_rep = self.post_cat_sem(self.dropout_rel(torch.relu(rel_rep)))
            predicate_proto = self.post_cat_sem(self.dropout_pred(torch.relu(predicate_proto)))


        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        if self.base_encoder == 'Self-Attention':
            ctx_gate = self.post_cat(prod_rep)
        else:
            prod_rep = self.post_cat(prod_rep)



        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                if self.catprep:
                    gate_pred = torch.sigmoid(self.gate_pred(cat((prod_rep, union_features), dim=-1)))  # gp
                    # fusion_so-B,2048
                    prod_rep = prod_rep - union_features * gate_pred
                else:
                    if self.base_encoder=='Self-Attention':
                        visual_rep = ctx_gate * union_features
                    else:
                        prod_rep = prod_rep * union_features

        if self.only_vision:
            prod_rep = union_features


        add_losses = {}


        if self.training:


            # 3samples





            if self.base_encoder == 'VCTree':
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)




            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj
            rel_labels = cat(rel_labels, dim=0)
            max_label = max(rel_labels)

            if self.use3env:
                num_groups=3
                cur_chosen_matrix = []
                for i in range(num_groups):
                    cur_chosen_matrix.append([])

                if self.intra_class:

                    for i in range(len(rel_labels)):
                        rel_tar = rel_labels[i].item()
                        sub_class = pair_pred[i][0].item()
                        obj_class = pair_pred[i][1].item()

                        rel_tar_word = self.class_names_r[rel_tar]
                        sub_class_word = self.class_names[sub_class]
                        obj_class_word = self.class_names[obj_class]
                        pair = (sub_class_word,obj_class_word)
                        if rel_tar != 0 and pair in self.importance_dic[rel_tar_word].keys():
                            samplerate = self.importance_dic[rel_tar_word][pair]
                        else:
                            samplerate = 0.01

                        if rel_tar == 0:
                            random_idx = random.randint(0, num_groups - 1)
                            cur_chosen_matrix[random_idx].append(i)
                        else:
                            random_num = random.random()
                            for j in range(num_groups):
                                if j==0:
                                    if random_num<=0.5:
                                        cur_chosen_matrix[j].append(i)
                                else:
                                    if random_num<=samplerate:
                                        cur_chosen_matrix[j].append(i)
                else:
                    for i in range(len(rel_labels)):
                        rel_tar = rel_labels[i].item()
                        if rel_tar == 0:
                            random_idx = random.randint(0, num_groups - 1)
                            cur_chosen_matrix[random_idx].append(i)
                        else:
                            random_num = random.random()
                            for j in range(num_groups):
                                if j==0:
                                    if random_num<=0.5:
                                        cur_chosen_matrix[j].append(i)
                                else:
                                    if random_num<=self.sample_rate_matrix[-1][rel_tar]:
                                        cur_chosen_matrix[j].append(i)

                ##########################

                #############









                for i in range(num_groups):
                    if len(prod_rep) == 0:
                            print("#############################feat==0########################################")
                            break

                    if max_label == 0:
                        if self.base_encoder == 'Self-Attention':
                            group_visual = visual_rep

                        group_input = prod_rep
                        group_label = rel_labels
                        group_pairs = pair_pred
                    else:
                        if self.base_encoder == 'Self-Attention':
                            group_visual = visual_rep[cur_chosen_matrix[i]]

                        group_input = prod_rep[cur_chosen_matrix[i]]
                        group_label = rel_labels[cur_chosen_matrix[i]]
                        group_pairs = pair_pred[cur_chosen_matrix[i]]

                        '''count Cross Entropy loss'''
                    jdx = i
                    if self.addproto:
                        # proto
                        predicate_proto = self.W_pred(self.rel_embed.weight)

                        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))

                        rel_rep_norm = group_input / group_input.norm(dim=1, keepdim=True)  # r_norm
                        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

                        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
                        group_output_now_proto = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()

                    if self.base_encoder == 'Self-Attention':
                        group_output_now = self.rel_compress(group_visual) + self.ctx_compress(group_input)
                    else:
                        group_output_now= self.rel_compress(group_input)
                    if self.use_bias:
                        if self.addproto:
                            group_output_now_proto = group_output_now_proto + self.freq_bias.index_with_labels(group_pairs.long())
                        else:
                            group_output_now = group_output_now + self.freq_bias.index_with_labels(group_pairs.long())



                    if self.useFCC:
                        group_output_now= self.fcc(group_output_now, group_label, 51 ,self.FCC_weight)
                    if self.use_causal:
                            self.qhat = self.update_qhat(torch.softmax(group_output_now.detach(), dim=-1), self.qhat,
                                                         momentum=0.99)
                            delta_logits = torch.log(self.qhat)
                            group_output_now = group_output_now + self.causal_weight * delta_logits




                    if i==2:
                        weights = torch.ones_like(group_label,dtype=torch.float16)

                        if self.intra_class:
                                for i in range(len(group_label)):
                                    rel_tar = group_label[i].item()
                                    sub_class = group_pairs[i][0].item()
                                    obj_class = group_pairs[i][1].item()
                                    rel_tar_word = self.class_names_r[rel_tar]
                                    sub_class_word = self.class_names[sub_class]
                                    obj_class_word = self.class_names[obj_class]
                                    pair = (sub_class_word, obj_class_word)
                                    if rel_tar != 0 and pair in self.importance_dic_weight[rel_tar_word].keys():
                                        weights[i] = self.importance_dic_weight[rel_tar_word][pair]
                                    else:
                                        weights[i] = 0.01

                        else:
                            for i in range(len(group_label)):
                                weights[i] = self.weight_rate_matrix[-1][group_label[i]]
                        if self.addproto:
                            add_losses[
                                '%d_CE_bias_proto_loss' % (jdx + 1)] = 2*self.overweighted_weight * self.rel_criterion_loss(
                                group_output_now_proto, group_label, weights)
                        add_losses['%d_CE_bias_loss' % (jdx + 1)] =self.overweighted_weight*self.rel_criterion_loss(group_output_now, group_label,weights)
                        if self.penalty_v1:
                            add_losses['%d_reg_bias_loss' % (jdx + 1)] =self.overweighted_weight_p*self.penalty(group_output_now, group_label,weights)

                    else:
                        if i == 1:
                            if self.addproto:
                                add_losses[
                                    '%d_CE_bias_proto_loss' % (
                                                jdx + 1)] = 2 * self.weighted_weight*self.criterion_loss(group_output_now_proto, group_label)
                            add_losses['%d_CE_loss' % (jdx + 1)] = self.weighted_weight*self.criterion_loss(group_output_now, group_label)
                            if self.penalty_v1:
                                add_losses['%d_reg_loss' % (jdx + 1)] = self.weighted_weight_p*self.penalty(group_output_now, group_label)

                        else:
                            if self.addproto:
                                add_losses[
                                    '%d_CE_proto_loss' % (
                                                jdx + 1)] = 2 * self.norm_weight*self.criterion_loss(group_output_now_proto, group_label)
                            add_losses['%d_CE_loss' % (jdx + 1)] = self.norm_weight*self.criterion_loss(group_output_now, group_label)
                            if self.penalty_v1:
                                add_losses['%d_reg_loss' % (jdx + 1)] = self.norm_weight_p*self.penalty(group_output_now, group_label)

                if self.penalty_v2:
                    losses = list(add_losses.values())
                    penalty = torch.var(torch.stack(list(add_losses.values())))
                    add_losses['IRM_penalty'] = self.penalty_v2_weight*penalty
            else:
                if self.addproto:
                    # proto
                    predicate_proto = self.W_pred(self.rel_embed.weight)

                    predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))

                    rel_rep_norm = prod_rep / prod_rep.norm(dim=1, keepdim=True)  # r_norm
                    predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

                    ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
                    rel_dists_proto = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  # <r_norm, c_norm> / τ

                if self.base_encoder== 'Self-Attention':
                    rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
                    self.use_bias = False
                else:
                    if self.base_encoder== 'VCTree':
                        rel_dists = self.rel_compress(prod_rep)
                    else:
                        rel_dists = self.rel_compress(prod_rep)



                if self.use_bias:
                    if self.addproto:
                        rel_dists_proto = rel_dists_proto + self.freq_bias.index_with_labels(pair_pred.long())


                    else:
                        rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

                if self.use_causal:
                    self.qhat = self.update_qhat(torch.softmax(rel_dists.detach(), dim=-1), self.qhat,
                                                 momentum=0.99)
                    delta_logits = torch.log(self.qhat)
                    rel_dists = rel_dists + self.causal_weight * delta_logits
                    if self.addproto:
                        self.qhat_proto = self.update_qhat(torch.softmax(rel_dists_proto.detach(), dim=-1), self.qhat_proto,
                                                           momentum=0.99)
                        delta_logits_proto = torch.log(self.qhat_proto)
                        rel_dists_proto = rel_dists_proto + self.causal_weight * delta_logits_proto

                if self.addproto:
                    loss_proto = self.criterion_loss(rel_dists_proto, rel_labels)
                    add_losses['rel_loss_proto'] = 1 * loss_proto
                loss_relation = self.criterion_loss(rel_dists, rel_labels)
                add_losses['rel_loss'] = loss_relation




                if self.addsem:
                    rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
                    predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

                    rel_dists_sem = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()

                    loss_relation_sem = self.criterion_loss(rel_dists_sem, rel_labels)
                    add_losses['rel_loss_sem'] = 5*loss_relation_sem


            return None, None, add_losses
        else:
            if self.addproto:
                predicate_proto = self.W_pred(self.rel_embed.weight)

                predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))

                rel_rep_norm = prod_rep / prod_rep.norm(dim=1, keepdim=True)  # r_norm
                predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

                ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
                rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()
            else:
                if self.base_encoder== 'Self-Attention':
                    rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
                    #self.use_bias = False
                else:
                    if self.base_encoder== 'VCTree':
                        rel_dists = self.ctx_compress(prod_rep)
                    else:
                        rel_dists = self.rel_compress(prod_rep)
            if self.use_bias:
                rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())





            rel_dists = rel_dists.split(num_rels, dim=0)
            obj_dists = obj_dists.split(num_objs, dim=0)
            return obj_dists, rel_dists, add_losses

    def generate_muti_networks(self, num_cls):
            '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
            if self.base_encoder == "Self-Attention" or self.base_encoder == "VCTree":
                self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
                self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
                self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
                self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)

            else:
                self.rel_classifer_1 = nn.Linear(self.pooling_dim, 50 + 1)
                self.rel_classifer_2 = nn.Linear(self.pooling_dim, 50 + 1)
                self.rel_classifer_3 = nn.Linear(self.pooling_dim, 50 + 1)
                self.rel_classifer_4 = nn.Linear(self.pooling_dim, 50 + 1)


            layer_init(self.rel_classifer_1, xavier=True)
            layer_init(self.rel_classifer_2, xavier=True)
            layer_init(self.rel_classifer_3, xavier=True)
            layer_init(self.rel_classifer_4, xavier=True)
            if num_cls == 4:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
            elif num_cls < 4:
                exit('wrong num in compress_all')
            else:
                self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
                layer_init(self.rel_classifer_5, xavier=True)
                if num_cls == 5:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5]
                else:
                    self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                    layer_init(self.rel_classifer_6, xavier=True)
                    if num_cls == 6:
                        classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                         self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                    else:
                        self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                        layer_init(self.rel_classifer_7, xavier=True)
                        classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                         self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                         self.rel_classifer_7]
                        if num_cls > 7:
                            exit('wrong num in compress_all')
            return classifer_all


@registry.ROI_RELATION_PREDICTOR.register("TransLike_GCL")
class TransLike_GCL(nn.Module):
    def __init__(self, config, in_channels):
        super(TransLike_GCL, self).__init__()
        # load parameters
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.GLOBAL_SETTING.USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # get model configs
        self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        # generate the auxiliary lists
        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, config.GLOBAL_SETTING.DATASET_CHOICE)
        self.sample_rate_matrix = generate_sample_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE, self.max_group_element_number_list)
        self.bias_for_group_split = generate_current_sequence_for_bias(num_of_group_element_list, config.GLOBAL_SETTING.DATASET_CHOICE)

        self.num_groups = len(self.max_elemnt_list)
        self.rel_compress_all, self.ctx_compress_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()

        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)

        if self.Knowledge_Transfer_Mode != 'None':
            self.NLL_Loss = nn.NLLLoss()
            self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
            self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
            self.criterion_loss = nn.CrossEntropyLoss()


        #3samples
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.freq_bias = FrequencyBias(config, statistics)
        self.rel_criterion_loss = ReweightingCE()
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)
        self.weight_rate_matrix = generate_weight_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE,
                                                              self.max_group_element_number_list)




        #sample-weighted
        self.norm_weight=0.1
        self.overweighted_weight=1
        self.iter = 0
        self.iter_2 = 0
        self.max_iter = 120000


        use_CLIP = False
        self.use_CLIP = use_CLIP
        # DenseCLIP
        if use_CLIP:
            self.proj_CLIP = nn.Linear(4096 + 51, 4096)
            self.attnpool = AttentionPool2d(80*80, 4096, 32, 512)
            # self.attnpool_fpn = AttentionPool2d(1344 // 32, 256, 32, 512)
            self.context_length = 5
            self.token_embed_dim = 512

            f = open("VGclasses.txt", "r")
            class_names = []
            class_list = f.readlines()
            for classes in class_list:
                classes = classes.strip('\n')
                class_names.append(classes)
            self.class_names = class_names

            f_r = open("VGrelationclasses.txt", "r")
            class_names_r = []
            class_list_r = f_r.readlines()
            for classes_r in class_list_r:
                classes_r = classes_r.strip('\n')
                class_names_r.append(classes_r)
            self.class_names_r = class_names_r

            self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])

            self.texts_r = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names_r])

            text_encoder = {'type': 'CLIPTextContextEncoder', 'context_length': 13, 'embed_dim': 512,
                            'transformer_width': 512, 'transformer_heads': 8, 'transformer_layers': 12,
                            'style': 'pytorch', 'pretrained': 'pretrained/RN101.pt'}
            context_decoder = {'type': 'ContextDecoder', 'transformer_width': 256, 'transformer_heads': 4,
                               'transformer_layers': 3, 'visual_dim': 512, 'dropout': 0.1, 'style': 'pytorch'}
            # pretrained='pretrained/RN101.pt'
            # assert text_encoder.get('pretrained') is None, \
            #         'both text encoder and segmentor set pretrained weight'
            # text_encoder.pretrained = pretrained
            self.text_encoder = build_backbone(text_encoder)
            self.context_decoder = build_backbone(context_decoder)
            context_length = self.text_encoder.context_length - self.context_length
            self.contexts = nn.Parameter(torch.randn(1, context_length, self.token_embed_dim))
            nn.init.trunc_normal_(self.contexts)
            self.text_dim = 512
            self.tau = 0.07
            self.gamma = nn.Parameter(torch.ones(self.text_dim) * 1e-4)
            self.text_encoder.init_weights()
        # print("111111111111111111")
    def compute_score_maps(self, feat, x_local, text_features):
        # B, K, C
        visual_embeddings = x_local
        text_features = F.normalize(text_features, dim=-1)
        visual_embeddings = F.normalize(visual_embeddings, dim=1)
        score_map = torch.einsum('bcn,bkc->bkn', visual_embeddings, text_features) / self.tau
        #score_map0 = F.upsample(score_map3, feat[0].shape[2:], mode='bilinear')
        #score_maps = [score_map0, None, None, score_map3]
        return score_map

    #def compute_text_features(self, x, x_global, x_local, dummy=False):
    def compute_text_features(self, x, x_local, dummy=False):
        """compute text features to each of x
        Args:
            x ([list]): list of features from the backbone,
                x[4] is the output of attentionpool2d
        """
        # global_feat, visual_embeddings =  x_global,x_local
        #
        # B, C, N = visual_embeddings.shape
        #
        # visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, N)],
        #                            dim=2).permute(0, 2, 1)  # B, N, C
        #
        # # text embeddings is (B, K, C)
        # if dummy:
        #     text_embeddings = torch.randn(B, len(self.texts), C, device=global_feat.device)
        # else:
        #     text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        # text_diff = self.context_decoder(text_embeddings, visual_context)
        # text_embeddings = text_embeddings + self.gamma * text_diff


        visual_embeddings =  x_local
        #global_feat=visual_embeddings.mean(dim=2, keepdim=True).unsqueeze(2)
        B, C, N = visual_embeddings.shape

        visual_context = visual_embeddings.permute(0, 2, 1)
        # visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, N)],
        #                            dim=2).permute(0, 2, 1)

        # text embeddings is (B, K, C)
        if dummy:
            text_embeddings = torch.randn(B, len(self.texts_r), C, device=visual_embeddings.device)
        else:
            text_embeddings = self.text_encoder(self.texts_r.to(visual_embeddings.device), self.contexts).expand(B, -1, -1)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff

        return text_embeddings

    #def extract_feat(self, feat, x_global, x_local, use_seg_loss=False, dummy=False):
    def extract_feat(self, feat, x_local, use_seg_loss=False, dummy=False):
        """Directly extract features from the backbone+neck."""

        # text_features = self.compute_text_features(feat, x_global, x_local, dummy=dummy)
        # score_maps = self.compute_score_maps(feat, x_local, text_features)
        # x = feat.permute(0, 2, 1)
        # # x = list(features[:-1])
        # x = torch.cat([x, score_maps], dim=1)
        # x_out = x.permute(0, 2, 1)
        # x = self.proj_CLIP(x_out).squeeze(0)
        # score_maps=score_maps.permute(0, 2, 1)

        text_features = self.compute_text_features(feat, x_local, dummy=dummy)
        score_maps = self.compute_score_maps(feat, x_local, text_features)
        x = feat.permute(0, 2, 1)
        # x = list(features[:-1])
        x = torch.cat([x, score_maps], dim=1)
        x_out = x.permute(0, 2, 1)
        x = self.proj_CLIP(x_out).squeeze(0)
        score_maps=score_maps.permute(0, 2, 1)

        # if self.with_neck:
        #     x = self.neck(x)

        if use_seg_loss:
            return x, score_maps[0]
        else:
            return x


    def penalty(self,logits, y, weights=None):
        scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
        logits_f=logits*scale
        loss = self.CE_loss(logits_f, y)
        if weights != None:
            x = F.log_softmax(logits_f, 1)
            target_onehot = torch.FloatTensor(y.size(0), 51).cuda()
            #GQA200
            #target_onehot=torch.FloatTensor(y.size(0),101).cuda()
            target_onehot.zero_()
            target_onehot.scatter_(1,y.view(-1,1),1)
            celoss = torch.sum(- x * target_onehot, dim=1)*weights
            celoss=torch.mean(celoss)
            grad = torch.autograd.grad(celoss, [scale], create_graph=True)[0]
        else:
            grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]


        fin_loss=torch.sum(grad ** 2)
        return fin_loss

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        add_losses = {}

        self.iter_2 += 1
        if self.iter_2 > 30000:
            self.iter += 1

        alpha = 1 - (self.iter / 30000)
        alpha = max(alpha, 0.1)
        alpha = alpha * 0.9
        if self.iter_2 <= 30000:
            alpha = 0.9

        self.norm_weight = alpha


        self.overweighted_weight = 1 - alpha



        #DenseCLIP
        #obj_dists,obj_preds, edge_ctx,scoremaps_so = self.context_layer(roi_features, proposals, logger)
        # post decode
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.union_single_not_match:#false
            visual_rep = ctx_gate * self.up_dim(union_features)
        else:
            visual_rep = ctx_gate * union_features






        if self.training:
            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj
            # fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            # loss_seg_so = self.criterion_loss(scoremaps_so, fg_labels.long())
            # add_losses['seg_loss_so'] = loss_seg_so

            # DenseCLIP-relation
            if self.use_CLIP:
                new_obj_feats = []
                scoremaps = []
                CLIP_obj_feats = list(visual_rep.split(num_rels, dim=0))
                for obj_feat in CLIP_obj_feats:
                    if len(obj_feat)==0:
                        print("#############################object_feat==0########################################")
                        continue
                    obj_feature = obj_feat.unsqueeze(0)
                    # obj_feature=obj_feat.unsqueeze(0)
                    # obj_feature = obj_feature.unsqueeze(0)
                    x_local = self.attnpool(obj_feature)
                    obj_feature, score_map = self.extract_feat(obj_feature, x_local, use_seg_loss=True)
                    new_obj_feats.append(obj_feat)
                    scoremaps.append(score_map)
                visual_rep = cat(new_obj_feats, dim=0)
                scoremaps = cat(scoremaps, dim=0)




            rel_labels = cat(rel_labels, dim=0)
            if self.use_CLIP:
                loss_seg = self.criterion_loss(scoremaps, rel_labels.long())
                add_losses['seg_loss'] = loss_seg
            max_label = max(rel_labels)

            # 3samples
            num_groups = 3
            cur_chosen_matrix = []

            for i in range(num_groups):
                cur_chosen_matrix.append([])

            for i in range(len(rel_labels)):
                rel_tar = rel_labels[i].item()
                if rel_tar == 0:
                    if self.zero_label_padding_mode == 'rand_insert':
                        random_idx = random.randint(0, num_groups - 1)
                        cur_chosen_matrix[random_idx].append(i)
                    elif self.zero_label_padding_mode == 'rand_choose' or self.zero_label_padding_mode == 'all_include':
                        if self.zero_label_padding_mode == 'rand_choose':
                            rand_zeros = random.random()
                        else:
                            rand_zeros = 1.0
                        if rand_zeros >= 0.4:
                            for zix in range(len(cur_chosen_matrix)):
                                cur_chosen_matrix[zix].append(i)
                else:
                    # rel_idx = self.incre_idx_list[rel_tar]
                    # random_num = random.random()
                    # for j in range(num_groups):
                    #     act_idx = num_groups - j
                    #     threshold_cur = self.sample_rate_matrix[act_idx - 1][rel_tar]
                    #     if random_num <= threshold_cur or act_idx < rel_idx:
                    #         # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
                    #         for k in range(act_idx):
                    #             cur_chosen_matrix[k].append(i)
                    #         break
                    random_num = random.random()
                    for j in range(num_groups):
                        if j == 0:
                            if random_num <= 0.5:
                                cur_chosen_matrix[j].append(i)
                        else:
                            if random_num <= self.sample_rate_matrix[-1][rel_tar]:
                                cur_chosen_matrix[j].append(i)

            for i in range(num_groups):
                if max_label == 0:
                        group_visual = visual_rep
                        group_input = prod_rep
                        group_label = rel_labels
                        group_pairs = pair_pred
                else:
                        group_visual = visual_rep[cur_chosen_matrix[i]]
                        group_input = prod_rep[cur_chosen_matrix[i]]
                        group_label = rel_labels[cur_chosen_matrix[i]]
                        group_pairs = pair_pred[cur_chosen_matrix[i]]
                jdx = i
                group_output_now = self.rel_compress(group_visual) + self.ctx_compress(group_input)
                # group_output_now_bias = self.rel_compress_biased(group_input)
                # group_output_now_overbias = self.rel_compress_overbiased(group_input)
                if self.use_bias:
                    group_output_now = group_output_now + self.freq_bias.index_with_labels(group_pairs.long())
                    # group_output_now_bias=group_output_now_bias + self.freq_bias.index_with_labels(group_pairs.long())
                    # group_output_now_overbias = group_output_now_overbias + self.freq_bias.index_with_labels(group_pairs.long())
                # add_losses['%d_CE_loss' % (jdx + 1)] = self.criterion_loss(group_output_now, group_label)
                if i == 2:
                    weights = torch.ones_like(group_label, dtype=torch.float16)
                    for i in range(len(group_label)):
                        weights[i] = self.weight_rate_matrix[-1][group_label[i]]
                    # idx=group_label.nonzero()
                    # for i in range(len(idx)):
                    #     weights[idx[i]] = self.weight_rate_matrix[-1][group_label[idx[i]]]
                    add_losses['%d_CE_bias_loss' % (jdx + 1)] = self.overweighted_weight *self.rel_criterion_loss(group_output_now, group_label,
                                                                                        weights)
                    add_losses['%d_reg_bias_loss' % (jdx + 1)] = self.overweighted_weight * self.penalty(
                        group_output_now, group_label, weights)
                else:
                    if i==1:
                        add_losses['%d_CE_loss' % (jdx + 1)] = self.criterion_loss(group_output_now, group_label)
                        add_losses['%d_reg_bias_loss' % (jdx + 1)] = self.penalty(
                            group_output_now, group_label)
                    else:
                        add_losses['%d_CE_loss' % (jdx + 1)] = self.norm_weight*self.criterion_loss(group_output_now, group_label)
                        add_losses['%d_reg_bias_loss' % (jdx + 1)] =self.norm_weight* self.penalty(
                            group_output_now, group_label)




            # num_groups = self.incre_idx_list[max_label.item()]
            # if num_groups == 0:
            #     num_groups = max(self.incre_idx_list)
            # cur_chosen_matrix = []
            #
            # for i in range(num_groups):
            #     cur_chosen_matrix.append([])
            #
            # for i in range(len(rel_labels)):
            #     rel_tar = rel_labels[i].item()
            #     if rel_tar == 0:
            #         if self.zero_label_padding_mode == 'rand_insert':
            #             random_idx = random.randint(0, num_groups - 1)
            #             cur_chosen_matrix[random_idx].append(i)
            #         elif self.zero_label_padding_mode == 'rand_choose' or self.zero_label_padding_mode == 'all_include':
            #             if self.zero_label_padding_mode == 'rand_choose':
            #                 rand_zeros = random.random()
            #             else:
            #                 rand_zeros = 1.0
            #             if rand_zeros >= 0.4:
            #                 for zix in range(len(cur_chosen_matrix)):
            #                     cur_chosen_matrix[zix].append(i)
            #     else:
            #         rel_idx = self.incre_idx_list[rel_tar]
            #         random_num = random.random()
            #         for j in range(num_groups):
            #             act_idx = num_groups - j
            #             threshold_cur = self.sample_rate_matrix[act_idx - 1][rel_tar]
            #             if random_num <= threshold_cur or act_idx < rel_idx:
            #                 # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
            #                 for k in range(act_idx):
            #                     cur_chosen_matrix[k].append(i)
            #                 break
            #
            # for i in range(num_groups):
            #     if max_label == 0:
            #         group_visual = visual_rep
            #         group_input = prod_rep
            #         group_label = rel_labels
            #         group_pairs = pair_pred
            #     else:
            #         group_visual = visual_rep[cur_chosen_matrix[i]]
            #         group_input = prod_rep[cur_chosen_matrix[i]]
            #         group_label = rel_labels[cur_chosen_matrix[i]]
            #         group_pairs = pair_pred[cur_chosen_matrix[i]]
            #
            #     '''count Cross Entropy Loss'''
            #     jdx = i
            #     rel_compress_now = self.rel_compress_all[jdx]
            #     ctx_compress_now = self.ctx_compress_all[jdx]
            #     group_output_now = rel_compress_now(group_visual) + ctx_compress_now(group_input)
            #     if self.use_bias:#true
            #         rel_bias_now = self.freq_bias_all[jdx]
            #         group_output_now = group_output_now + rel_bias_now.index_with_labels(group_pairs.long())
            #     # actual_label_piece: if label is out of range, then filter it to ensure the training can continue
            #     actual_label_now = self.pre_group_matrix[jdx][group_label]
            #     add_losses['%d_CE_loss' % (jdx + 1)] = self.CE_loss(group_output_now, actual_label_now)
            #
            #     if self.Knowledge_Transfer_Mode == 'KL_logit_Neighbor':
            #         if i > 0:
            #             '''count knowledge transfer loss'''
            #             jbef = i - 1
            #             rel_compress_bef = self.rel_compress_all[jbef]
            #             ctx_compress_bef = self.ctx_compress_all[jbef]
            #             group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
            #             if self.use_bias:
            #                 rel_bias_bef = self.freq_bias_all[jbef]
            #                 group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
            #             max_vector = self.max_elemnt_list[jbef] + 1
            #
            #             if self.no_relation_restrain:
            #                 kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
            #                 kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
            #                                                reduce=False)
            #                 kd_loss_vecify = kd_loss_matrix * kd_choice_vector
            #                 kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
            #             else:
            #                 kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
            #                                                reduce=True)
            #                 kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
            #             add_losses['%d%d_kl_loss' % (jbef + 1, jdx + 1)] = kd_loss_final
            #
            #     elif self.Knowledge_Transfer_Mode == 'KL_logit_TopDown':
            #         layer_total_loss = 0
            #         for jbef in range(i):
            #             rel_compress_bef = self.rel_compress_all[jbef]
            #             ctx_compress_bef = self.ctx_compress_all[jbef]
            #             group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
            #             if self.use_bias:
            #                 rel_bias_bef = self.freq_bias_all[jbef]
            #                 group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
            #             max_vector = self.max_elemnt_list[jbef] + 1
            #
            #             if self.no_relation_restrain:
            #                 kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
            #                 kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
            #                                                reduce=False)
            #                 kd_loss_vecify = kd_loss_matrix * kd_choice_vector
            #                 kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
            #             else:
            #                 kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
            #                                                reduce=True)
            #                 kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
            #             layer_total_loss += kd_loss_final
            #
            #         if i > 0:
            #             add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss
            #
            #     elif self.Knowledge_Transfer_Mode == 'KL_logit_BiDirection':
            #         layer_total_loss = 0
            #         for jbef in range(i):
            #             rel_compress_bef = self.rel_compress_all[jbef]
            #             ctx_compress_bef = self.ctx_compress_all[jbef]
            #             group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
            #             if self.use_bias:
            #                 rel_bias_bef = self.freq_bias_all[jbef]
            #                 group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
            #             max_vector = self.max_elemnt_list[jbef] + 1
            #
            #             if self.no_relation_restrain:
            #                 kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
            #                 kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
            #                                                   group_output_now[:, 1:max_vector],
            #                                                   reduce=False)
            #                 kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
            #                                                   group_output_bef[:, 1:],
            #                                                   reduce=False)
            #                 kd_loss_vecify = (kd_loss_matrix_td + kd_loss_matrix_bu) * kd_choice_vector
            #                 kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
            #             else:
            #                 kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
            #                                                   group_output_now[:, 1:max_vector],
            #                                                   reduce=True)
            #                 kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
            #                                                   group_output_bef[:, 1:],
            #                                                   reduce=True)
            #                 kd_loss_final = self.knowledge_loss_coefficient * (kd_loss_matrix_td + kd_loss_matrix_bu)
            #             layer_total_loss += kd_loss_final
            #
            #         if i > 0:
            #             add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss

            return None, None, add_losses
        else:
            # rel_compress_test = self.rel_compress_all[-1]
            # ctx_compress_test = self.ctx_compress_all[-1]
            # rel_dists = rel_compress_test(visual_rep) + ctx_compress_test(prod_rep)
            # if self.use_bias:
            #     rel_bias_test = self.freq_bias_all[-1]
            #     rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())

            rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
            # group_output_now_bias = self.rel_compress_biased(group_input)
            # group_output_now_overbias = self.rel_compress_overbiased(group_input)
            if self.use_bias:
                rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

            rel_dists = rel_dists.split(num_rels, dim=0)
            obj_dists = obj_dists.split(num_objs, dim=0)

            return obj_dists, rel_dists, add_losses

    def generate_muti_networks(self, num_cls):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)
        self.rel_compress_1 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[0] + 1)
        self.rel_compress_2 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[1] + 1)
        self.rel_compress_3 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[2] + 1)
        self.rel_compress_4 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[3] + 1)
        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        layer_init(self.rel_classifer_4, xavier=True)
        layer_init(self.rel_compress_1, xavier=True)
        layer_init(self.rel_compress_2, xavier=True)
        layer_init(self.rel_compress_3, xavier=True)
        layer_init(self.rel_compress_4, xavier=True)
        if num_cls == 4:
            classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
            compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3, self.rel_compress_4]
        elif num_cls < 4:
            exit('wrong num in compress_all')
        else:
            self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_classifer_5, xavier=True)
            self.rel_compress_5 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_compress_5, xavier=True)
            if num_cls == 5:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                 self.rel_classifer_4, self.rel_classifer_5]
                compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                self.rel_compress_4, self.rel_compress_5]
            else:
                self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_classifer_6, xavier=True)
                self.rel_compress_6 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_compress_6, xavier=True)
                if num_cls == 6:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6]
                else:
                    self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_classifer_7, xavier=True)
                    self.rel_compress_7 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_compress_7, xavier=True)
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                     self.rel_classifer_7]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6, self.rel_compress_7]
                    if num_cls > 7:
                        exit('wrong num in compress_all')
        return classifer_all, compress_all

    def generate_multi_bias(self, config, statistics, num_cls):
        self.freq_bias_1 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[0])
        self.freq_bias_2 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[1])
        self.freq_bias_3 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[2])
        self.freq_bias_4 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[3])
        if num_cls < 4:
            exit('wrong num in multi_bias')
        elif num_cls == 4:
            freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4]
        else:
            self.freq_bias_5 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[4])
            if num_cls == 5:
                freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4,
                                 self.freq_bias_5]
            else:
                self.freq_bias_6 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                      predicate_all_list=self.bias_for_group_split[5])
                if num_cls == 6:
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6]
                else:
                    self.freq_bias_7 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                          predicate_all_list=self.bias_for_group_split[6])
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6, self.freq_bias_7]
                    if num_cls > 7:
                        exit('wrong num in multi_bias')
        return freq_bias_all

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@registry.ROI_RELATION_PREDICTOR.register("MotifsLikePredictor")
class MotifsLikePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifsLikePredictor, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.GLOBAL_SETTING.USE_BIAS

        self.softtripletloss=SoftTriple(la=20, gamma=0.1, tau=0.2, margin=0.01, dim=4096, cN=51, K=10).cuda()
        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Motifs':
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'VTransE':
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            exit('wrong mode!')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)
        self.criterion_loss = nn.CrossEntropyLoss()
        self.rel_criterion_loss = ReweightingCE()


        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.weight_rate_matrix = generate_weight_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE,
                                                              self.max_group_element_number_list)

        use_CLIP = False
        self.use_CLIP = use_CLIP
        # DenseCLIP
        if use_CLIP:
            self.proj_CLIP = nn.Linear(4096 + 51, 4096)
            self.attnpool = AttentionPool2d(80*80, 4096, 32, 512)
            # self.attnpool_fpn = AttentionPool2d(1344 // 32, 256, 32, 512)
            self.context_length = 5
            self.token_embed_dim = 512

            f = open("VGclasses.txt", "r")
            class_names = []
            class_list = f.readlines()
            for classes in class_list:
                classes = classes.strip('\n')
                class_names.append(classes)
            self.class_names = class_names

            f_r = open("VGrelationclasses.txt", "r")
            class_names_r = []
            class_list_r = f_r.readlines()
            for classes_r in class_list_r:
                classes_r = classes_r.strip('\n')
                class_names_r.append(classes_r)
            self.class_names_r = class_names_r

            self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])

            self.texts_r = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names_r])

            text_encoder = {'type': 'CLIPTextContextEncoder', 'context_length': 13, 'embed_dim': 512,
                            'transformer_width': 512, 'transformer_heads': 8, 'transformer_layers': 12,
                            'style': 'pytorch', 'pretrained': 'pretrained/RN101.pt'}
            context_decoder = {'type': 'ContextDecoder', 'transformer_width': 256, 'transformer_heads': 4,
                               'transformer_layers': 3, 'visual_dim': 512, 'dropout': 0.1, 'style': 'pytorch'}
            # pretrained='pretrained/RN101.pt'
            # assert text_encoder.get('pretrained') is None, \
            #         'both text encoder and segmentor set pretrained weight'
            # text_encoder.pretrained = pretrained
            self.text_encoder = build_backbone(text_encoder)
            self.context_decoder = build_backbone(context_decoder)
            context_length = self.text_encoder.context_length - self.context_length
            self.contexts = nn.Parameter(torch.randn(1, context_length, self.token_embed_dim))
            nn.init.trunc_normal_(self.contexts)
            self.text_dim = 512
            self.tau = 0.07
            self.gamma = nn.Parameter(torch.ones(self.text_dim) * 1e-4)
            self.text_encoder.init_weights()
        self.qhat = self.initial_qhat(class_num=51)
        self.use_causal= False
        self.use_causal_test = False

        self.qhat_proto = self.initial_qhat(class_num=51)
        self.use_proto= False
        self.W_pred = MLP(300, 1024, 2048, 2)

        #最大300dim
        #rel_embed_vecs = self.rel_vectors(rel_classes, wv_dir=self.config.GLOVE_DIR, wv_dim=300)   # load Glove for predicates
        self.rel_embed = nn.Embedding(51, 300)
        # with torch.no_grad():
        #     self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.project_head= MLP(2048, 2048, 4096, 2)
        self.dropout_pred = nn.Dropout(0.2)



    def rel_vectors(self,names, wv_dir, wv_type='glove.6B', wv_dim=300):
        wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

        vectors = torch.Tensor(len(names), wv_dim)  # 51, 200
        vectors.normal_(0, 1)
        for i, token in enumerate(names):
            if i == 0:
                continue
            wv_index = wv_dict.get(token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                # 进行混合然后求平均_split word
                split_token = token.split(' ')
                ss = 0
                s_vec = torch.zeros(wv_dim)
                for s_token in split_token:
                    wv_index = wv_dict.get(s_token)
                    if wv_index is not None:
                        ss += 1
                        s_vec += wv_arr[wv_index]
                    else:
                        print("fail on {}".format(token))
                s_vec /= ss
                vectors[i] = s_vec

        return vectors



    def initial_qhat(self,class_num=1000):
            # initialize qhat of predictions (probability)
            qhat = (torch.ones([1, class_num], dtype=torch.float) / class_num).cuda()
            print("qhat size: ".format(qhat.size()))
            return qhat

    def causal_inference(self,current_logit, qhat, exp_idx, tau=0.4):
        # de-bias pseudo-labels
        debiased_prob = F.softmax(current_logit - tau * torch.log(qhat), dim=1)
        return debiased_prob

    def update_qhat(self,probs, qhat, momentum, qhat_mask=None):
        if qhat_mask is not None:
            mean_prob = probs.detach() * qhat_mask.detach().unsqueeze(dim=-1)
        else:
            mean_prob = probs.detach().mean(dim=0)
        qhat = momentum * qhat + (1 - momentum) * mean_prob
        return qhat


    def compute_score_maps(self, feat, x_local, text_features):
        # B, K, C
        visual_embeddings = x_local
        text_features = F.normalize(text_features, dim=-1)
        visual_embeddings = F.normalize(visual_embeddings, dim=1)
        score_map = torch.einsum('bcn,bkc->bkn', visual_embeddings, text_features) / self.tau
        #score_map0 = F.upsample(score_map3, feat[0].shape[2:], mode='bilinear')
        #score_maps = [score_map0, None, None, score_map3]
        return score_map

    #def compute_text_features(self, x, x_global, x_local, dummy=False):
    def compute_text_features(self, x, x_local, dummy=False):
        """compute text features to each of x
        Args:
            x ([list]): list of features from the backbone,
                x[4] is the output of attentionpool2d
        """
        # global_feat, visual_embeddings =  x_global,x_local
        #
        # B, C, N = visual_embeddings.shape
        #
        # visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, N)],
        #                            dim=2).permute(0, 2, 1)  # B, N, C
        #
        # # text embeddings is (B, K, C)
        # if dummy:
        #     text_embeddings = torch.randn(B, len(self.texts), C, device=global_feat.device)
        # else:
        #     text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        # text_diff = self.context_decoder(text_embeddings, visual_context)
        # text_embeddings = text_embeddings + self.gamma * text_diff


        visual_embeddings =  x_local
        #global_feat=visual_embeddings.mean(dim=2, keepdim=True).unsqueeze(2)
        B, C, N = visual_embeddings.shape

        visual_context = visual_embeddings.permute(0, 2, 1)
        # visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, N)],
        #                            dim=2).permute(0, 2, 1)

        # text embeddings is (B, K, C)
        if dummy:
            text_embeddings = torch.randn(B, len(self.texts), C, device=visual_embeddings.device)
        else:
            text_embeddings = self.text_encoder(self.texts.to(visual_embeddings.device), self.contexts).expand(B, -1, -1)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff

        return text_embeddings

    #def extract_feat(self, feat, x_global, x_local, use_seg_loss=False, dummy=False):
    def extract_feat(self, feat, x_local, use_seg_loss=False, dummy=False):
        """Directly extract features from the backbone+neck."""

        # text_features = self.compute_text_features(feat, x_global, x_local, dummy=dummy)
        # score_maps = self.compute_score_maps(feat, x_local, text_features)
        # x = feat.permute(0, 2, 1)
        # # x = list(features[:-1])
        # x = torch.cat([x, score_maps], dim=1)
        # x_out = x.permute(0, 2, 1)
        # x = self.proj_CLIP(x_out).squeeze(0)
        # score_maps=score_maps.permute(0, 2, 1)

        text_features = self.compute_text_features(feat, x_local, dummy=dummy)
        score_maps = self.compute_score_maps(feat, x_local, text_features)
        x = feat.permute(0, 2, 1)
        # x = list(features[:-1])
        x = torch.cat([x, score_maps], dim=1)
        x_out = x.permute(0, 2, 1)
        x = self.proj_CLIP(x_out).squeeze(0)
        score_maps=score_maps.permute(0, 2, 1)

        # if self.with_neck:
        #     x = self.neck(x)

        if use_seg_loss:
            return x, score_maps[0]
        else:
            return x


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)
        #obj_dists, obj_preds, edge_ctx, scoremaps = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            # if len(head_rep) or len(tail_rep)==0:
            #     continue
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features
        if self.use_CLIP:
            new_obj_feats = []
            scoremaps = []
            CLIP_obj_feats = list(prod_rep.split(num_rels, dim=0))
            for obj_feat in CLIP_obj_feats:
                obj_feature = obj_feat.unsqueeze(0)
                # obj_feature=obj_feat.unsqueeze(0)
                # obj_feature = obj_feature.unsqueeze(0)
                x_local = self.attnpool(obj_feature)
                obj_feature, score_map = self.extract_feat(obj_feature, x_local, use_seg_loss=True)
                new_obj_feats.append(obj_feat)
                scoremaps.append(score_map)
            prod_rep = cat(new_obj_feats, dim=0)
            scoremaps = cat(scoremaps, dim=0)

        if self.use_proto:
            # proto
            predicate_proto = self.W_pred(self.rel_embed.weight)

            predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))

            rel_rep_norm = prod_rep / prod_rep.norm(dim=1, keepdim=True)  # r_norm
            predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

            ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
            rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  # <r_norm, c_norm> / τ

            rel_dists_ce = self.rel_compress(prod_rep)
        else:
            rel_dists = self.rel_compress(prod_rep)
        #rel_dists = self.rel_compress(union_features)





        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        add_losses = {}

        if self.training:
            rel_labels = cat(rel_labels, dim=0)
            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj

            # fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            # loss_seg = self.criterion_loss(scoremaps, fg_labels.long())
            # add_losses['seg_loss'] = loss_seg
            # rel_labels = cat(rel_labels, dim=0)
            # weights = torch.ones_like(rel_labels, dtype=torch.float16)
            # for i in range(len(rel_labels)):
            #     weights[i] = self.weight_rate_matrix[-1][rel_labels[i]]
            # idx=group_label.nonzero()
            # for i in range(len(idx)):
            #     weights[idx[i]] = self.weight_rate_matrix[-1][group_label[idx[i]]]
            # add_losses['CE_bias_loss'] = self.rel_criterion_loss(rel_dists, rel_labels, weights)
            if self.use_causal_test:
                self.qhat = self.update_qhat(torch.softmax(rel_dists.detach(), dim=-1), self.qhat, momentum=0.99)






            if self.use_causal:
                #pseudo_label = self.causal_inference(rel_dists.detach(), self.qhat, exp_idx=0, tau=0.4)

                if self.use_proto:
                    self.qhat_proto = self.update_qhat(torch.softmax(rel_dists.detach(), dim=-1), self.qhat_proto, momentum=0.99)
                    delta_logits_proto = torch.log(self.qhat_proto)
                    self.qhat = self.update_qhat(torch.softmax(rel_dists_ce.detach(), dim=-1), self.qhat, momentum=0.99)
                    delta_logits = torch.log(self.qhat)

                    rel_dists_proto_causal = rel_dists + 1 * delta_logits_proto
                    rel_dists_causal = rel_dists_ce + 1 * delta_logits

                    loss_relation = self.criterion_loss(rel_dists_proto_causal, rel_labels)
                    #loss_relation = self.criterion_loss(rel_dists, rel_labels)
                    loss_relation_ce = self.criterion_loss(rel_dists_causal, rel_labels)

                    rate = (loss_relation.detach() / loss_relation_ce.detach())
                    loss_relation = (1 + rate) * loss_relation
                    #loss_relation = 2 * loss_relation

                    add_losses['rel_loss'] = loss_relation
                    add_losses['rel_loss_ce'] = loss_relation_ce

                else:
                    self.qhat = self.update_qhat(torch.softmax(rel_dists.detach(), dim=-1), self.qhat, momentum=0.99)
                    delta_logits = torch.log(self.qhat)
                    rel_dists_causal = rel_dists + 1 * delta_logits
                    add_losses['rel_loss'] = self.criterion_loss(rel_dists_causal, rel_labels)

            else:
                # reweight
                # weights = torch.ones_like(rel_labels, dtype=torch.float16)
                # for i in range(len(rel_labels)):
                #     weights[i] = self.weight_rate_matrix[-1][rel_labels[i]]
                # loss_relation = self.rel_criterion_loss(rel_dists, rel_labels,
                #                         weights)


                loss_relation = self.criterion_loss(rel_dists, rel_labels)



                #loss_relation_ce = self.criterion_loss(rel_dists_ce, rel_labels)



                #rate=(loss_relation.detach()/loss_relation_ce.detach())
                #loss_relation=(1+rate)*loss_relation
                #loss_relation = 2 * loss_relation


                add_losses['rel_loss'] = loss_relation
                #add_losses['rel_loss_ce'] = loss_relation_ce

            #proxy
            #add_losses['proxy_loss'] = self.softtripletloss(prod_rep, rel_labels)
            if self.use_CLIP:
                add_losses['seg_loss'] = self.criterion_loss(scoremaps, rel_labels)
            return None, None, add_losses
        else:
            if self.use_causal_test:
                delta_logits = torch.log(self.qhat)
                rel_dists_causal = rel_dists - 0.4 * delta_logits
                obj_dists = obj_dists.split(num_objs, dim=0)
                rel_dists = rel_dists_causal.split(num_rels, dim=0)
            else:
                obj_dists = obj_dists.split(num_objs, dim=0)
                rel_dists = rel_dists.split(num_rels, dim=0)
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()


        print("VCTREE*********************************************************************************")
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        #self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #layer_init(self.uni_gate, xavier=True)
        #layer_init(self.frq_gate, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        #layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.use_bias = config.GLOBAL_SETTING.USE_BIAS
        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)
        self.criterion_loss = nn.CrossEntropyLoss()

        self.use_causal= True
        self.causal_weight = 1



        if self.use_causal==True:
            if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
                self.qhat = self.initial_qhat(class_num=51)
            else:
                self.qhat = self.initial_qhat(class_num=101)
        self.iter_2=0


    def initial_qhat(self,class_num=1000):
            # initialize qhat of predictions (probability)
            qhat = (torch.ones([1, class_num], dtype=torch.float) / class_num).cuda()
            print("qhat size: ".format(qhat.size()))
            return qhat

    def update_qhat(self,probs, qhat, momentum, qhat_mask=None):
        if qhat_mask is not None:
            mean_prob = probs.detach() * qhat_mask.detach().unsqueeze(dim=-1)
        else:
            mean_prob = probs.detach().mean(dim=0)
        qhat = momentum * qhat + (1 - momentum) * mean_prob
        return qhat


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        self.iter_2 += 1
        if self.iter_2 == 99:
            print("*******************************************************************************************")


        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        #uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        #frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        rel_dists = self.ctx_compress(prod_rep * union_features)
        #uni_dists = self.uni_compress(self.drop(union_features))
        #frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        #rel_dists = ctx_dists + frq_dists
        #rel_dists = ctx_dists
        #rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        add_losses = {}

        if self.training:

            if self.use_bias:
                rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())


            if self.use_causal:
                self.qhat = self.update_qhat(torch.softmax(rel_dists.detach(), dim=-1), self.qhat,
                                             momentum=0.99)
                delta_logits = torch.log(self.qhat)
                rel_dists = rel_dists + self.causal_weight * delta_logits



            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)
            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj
            rel_labels = cat(rel_labels, dim=0)
            add_losses['rel_loss'] = self.criterion_loss(rel_dists, rel_labels)
            return None, None, add_losses
        else:


            if self.use_bias:
                rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())
            rel_dists = rel_dists.split(num_rels, dim=0)
            obj_dists = obj_dists.split(num_objs, dim=0)

            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("MotifsLike_GCL")
class MotifsLike_GCL(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifsLike_GCL, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.GLOBAL_SETTING.USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            obj_classes, rel_classes,stat_classes = statistics['obj_classes'], statistics['rel_classes'],statistics['stat_classes']
            self.class_stat=sample_rate(stat_classes)
            self.class_stat_weight = weight_rate(stat_classes)


            self.use_stat_class = False
            if self.use_stat_class:
                self.class_stat.sort()
                self.class_stat_weight.sort()



        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Motifs':
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'VTransE':
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            exit('wrong mode!')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        #BBN
        self.post_cat_1 = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress_BBN = nn.Linear(self.pooling_dim*2, self.num_rel_cls, bias=True)

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        #relation inference
        self.subject_inference = nn.Linear(self.pooling_dim+self.hidden_dim, 151, bias=True)
        self.object_inference = nn.Linear(self.pooling_dim+self.hidden_dim, 151, bias=True)

        self.rel_compress_biased = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)
        self.rel_compress_overbiased = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)
        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # get model configs
        self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        # generate the auxiliary lists
        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, config.GLOBAL_SETTING.DATASET_CHOICE)
        self.sample_rate_matrix = generate_sample_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE, self.max_group_element_number_list)

        self.weight_rate_matrix = generate_weight_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE,
                                                              self.max_group_element_number_list)



        self.bias_for_group_split = generate_current_sequence_for_bias(num_of_group_element_list, config.GLOBAL_SETTING.DATASET_CHOICE)

        self.num_groups = len(self.max_elemnt_list)
        self.rel_classifer_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()
        self.softtripletloss = SoftTriple(la=20, gamma=0.1, tau=0.2, margin=0.01, dim=4096, cN=51, K=10).cuda()
        # if self.use_bias:
        #     self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)
        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)
            self.freq_bias = FrequencyBias(config, statistics)

        if self.Knowledge_Transfer_Mode != 'None':
            self.NLL_Loss = nn.NLLLoss()
            self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
            self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()



        self.rel_criterion_loss = ReweightingCE()
        self.criterion_loss = nn.CrossEntropyLoss()
        self.xERMloss=xERMLoss()

        # f = open("VGclasses.txt", "r")
        # class_names = []
        # class_list = f.readlines()
        # for classes in class_list:
        #     classes = classes.strip('\n')
        #     class_names.append(classes)
        # self.class_names = class_names
        #
        # f_r = open("VGrelationclasses.txt", "r")
        # class_names_r = []
        # class_list_r = f_r.readlines()
        # for classes_r in class_list_r:
        #     classes_r = classes_r.strip('\n')
        #     class_names_r.append(classes_r)
        # self.class_names_r = class_names_r
        #
        # rel_cnt_dic = {}
        # path = "em_E.pk"
        # l = pickle.load(open(path, "rb"))
        # vocab = json.load(open("VG-SGG-dicts-with-attri.json", "r"))
        # idx2lb = {int(k): v for k, v in vocab["idx_to_label"].items()}
        # idx2pred = {int(k) - 1: v for k, v in vocab["idx_to_predicate"].items()}
        # for i, data in enumerate(l):
        #     labels = data["labels"]
        #     logits = data["logits"][:, 1:]
        #     relation_tuple = deepcopy(data["relations"])
        #     sub_idxs, obj_idxs, rels = relation_tuple[:, 0], relation_tuple[:, 1], relation_tuple[:, 2]
        #     sub_lbs, obj_lbs = labels[sub_idxs], labels[obj_idxs]
        #     # [[sub_lb1, obj_lb1], [sub_lb2, obj_lb2]......]
        #     pairs = np.stack([sub_lbs, obj_lbs], 1).tolist()
        #     pairs = [(idx2lb[p[0]], idx2lb[p[1]]) for p in pairs]
        #     # behave as indexes, so -=1
        #     rels -= 1
        #
        #     # fill in rel_dic
        #     # rel_dic: {rel_i: {pair_j: distribution} }
        #     for j, (pair, r, logit) in enumerate(zip(pairs, rels, logits)):
        #         r_name = idx2pred[int(r)]
        #         if r_name not in rel_cnt_dic:
        #             rel_cnt_dic[r_name] = {}
        #         if pair not in rel_cnt_dic[r_name]:
        #             rel_cnt_dic[r_name][pair] = 0
        #         rel_cnt_dic[r_name][pair] += 1
        # self.importance_dic = {}
        # for r, pair_cnt_dic in rel_cnt_dic.items():
        #     for pair in pair_cnt_dic:
        #         cnt = pair_cnt_dic[pair]
        #         triplet = (r, *pair)
        #         #self.importance_dic[triplet] = cnt / sum(pair_cnt_dic.values())
        #         self.importance_dic[triplet] = cnt
        #
        #
        # all_keys = list(self.importance_dic.keys())
        # self.importance_dic_weight={}
        # self.importance_dic_sample = {}
        # arr_value = np.stack([list(self.importance_dic.values())])
        # self.med = np.median(arr_value, 1)[0]
        # for dict in all_keys:
        #     if self.importance_dic[dict] > self.med:
        #         self.importance_dic_weight[dict] = self.med/self.importance_dic[dict]
        #     else:
        #         self.importance_dic_weight[dict] = 1.0
        #
        #     self.importance_dic_sample[dict] = self.med/self.importance_dic[dict]
        #
        #     # if self.importance_dic_weight[dict] < 0.01:
        #     #     self.importance_dic_weight[dict] = 0.01
        #     # if self.importance_dic_sample[dict] < 0.01:
        #     #     self.importance_dic_sample[dict] = 0.01
        #
        # weight_value = np.stack([list(self.importance_dic_weight.values())])
        # self.weight_mean = np.mean(weight_value, 1)[0]
        # sorted_dic = sorted(self.importance_dic_sample.items(), key=lambda k: k[-1], reverse=True)
        # self.med_weight = np.median(weight_value, 1)[0]
        # self.mean_weight = np.mean(weight_value, 1)[0]
        # self.min_weight = np.min(weight_value, 1)[0]
        # self.max_weight = np.max(weight_value, 1)[0]

        self.overweighted_weight = 1
        self.overweighted_weight_p = 1
        #self.norm_weight=0.1
        #self.norm_weight_p = 0.1
        self.norm_weight= 1
        self.norm_weight_p = 1

        self.weighted_weight = 1
        self.weighted_weight_p = 2

        self.iter = 0
        self.iter_2 = 0
        self.max_iter = 120000



        #self.binary_weight = 0.5
        #self.counter_binary=nn.Linear(self.pooling_dim, 2, bias=True)



        #deepBDC
        # self.dcov = BDC(is_vec=True, input_dim=4096, dimension_reduction=256)
        # self.feat_dim = int(256 * (256 + 1) / 2)
        # self.dropout = nn.Dropout(0.5)
        # self.bdcclshead = nn.Linear(self.feat_dim, self.num_rel_cls)
        # self.bdcclshead.bias.data.fill_(0)


        #Eqinv
        # self.mlp = nn.Sequential(nn.Linear(self.pooling_dim, 512, bias=False), nn.BatchNorm1d(512),
        #                          nn.ReLU(inplace=True), nn.Linear(512, 128, bias=True))
        # self.scaler = 10
        # mask_layer = torch.rand(self.pooling_dim, )
        # self.mask_layer = torch.nn.Parameter(mask_layer)



        #BBN
        self.BBN = False


        #TSNE
        self.TSNE= False
        if self.TSNE:
            self.num_groups_tsne = 2
            self.cur_chosen_label_tsne = []
            self.cur_chosen_tensor_tsne = []
            for i in range(self.num_groups_tsne):
                self.cur_chosen_label_tsne.append([])
                self.cur_chosen_tensor_tsne.append([])



        self.test_flag=0

        self.test_iter = 0






        '''
        torch.int64
        torch.float16
        '''

    def penalty(self,logits, y, weights=None):
        scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
        logits_f=logits*scale
        loss = self.CE_loss(logits_f, y)
        if weights != None:
            x = F.log_softmax(logits_f, 1)
            target_onehot = torch.FloatTensor(y.size(0), 51).cuda()
            #GQA200
            #target_onehot=torch.FloatTensor(y.size(0),101).cuda()
            target_onehot.zero_()
            target_onehot.scatter_(1,y.view(-1,1),1)
            celoss = torch.sum(- x * target_onehot, dim=1)*weights
            celoss=torch.mean(celoss)
            grad = torch.autograd.grad(celoss, [scale], create_graph=True)[0]
        else:
            grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]


        fin_loss=torch.sum(grad ** 2)
        return fin_loss

    def info_nce_loss_supervised(self,features, batch_size, temperature=0.07, base_temperature=0.07, labels=None,
                                 choose_pos=None):
        ### features    bs * 2 * dim
        forlabels = labels.nonzero()
        #forlabels =forlabels.contiguous().view(-1, 1)
        labels = labels.contiguous().view(-1, 1)

        # for i in range(len(idx)):
        #     weights[idx[i]] = self.weight_rate_matrix[-1][group_label[idx[i]]]
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        #mask = torch.eq(labels, labels.T).float().cuda()
        mask=torch.eq(labels[forlabels].squeeze(1), labels[forlabels].squeeze(1).T).float().cuda()

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        contrast_feature=contrast_feature[forlabels].squeeze(1)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # plot111=torch.arange(batch_size * anchor_count).view(-1, 1)
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
        #     0
        # )
        #mask = mask * logits_mask

        # compute log_prob
        #exp_logits = torch.exp(logits) * logits_mask
        exp_logits = torch.exp(logits)
        #neg=torch.log(exp_logits.sum(1, keepdim=True))

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        #fenmu=(mask * log_prob).sum(1)
        #fenzi=mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        #mean_log_prob_pos=mean_log_prob_pos[forlabels].squeeze(1)

        # loss
        loss = - (temperature / base_temperature) * mean_log_prob_pos

        if choose_pos is None:
            #loss = loss.view(anchor_count, batch_size).mean()
            loss = loss.view(anchor_count, len(forlabels)).mean()
        else:
            loss = loss.view(anchor_count, batch_size)[:, choose_pos].sum() / choose_pos.sum()

        return loss



    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):


    #TSNE
    #def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, proposals_1, rel_pair_idxs_1, rel_labels_1, rel_binarys_1,roi_features_1, union_features_1,logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """



        if self.iter_2==99:
            print("*******************************************************************************************")

        # reweight
        T=30000

        self.iter_2 += 1
        if self.iter_2 > T:
            self.iter += 1
        #
        alpha =  1-(self.iter / T)
        alpha = max(alpha, 0.1)
        alpha= alpha*0.9
        if self.iter_2 <= T:
            alpha = 0.9

        self.norm_weight=alpha
        self.norm_weight_p=alpha

        self.overweighted_weight=1-alpha
        self.overweighted_weight_p=1-alpha


        # encode context infomation
        # if not self.training:
        #     print("roi_features",roi_features)
        #     print("proposals", proposals)


        obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        # BBN
        if self.BBN:
            if self.training:
                num_groups = 2
                cur_chosen_matrix = []
                for i in range(num_groups):
                    cur_chosen_matrix.append([])

                rel_labels = cat(rel_labels, dim=0)

                for i in range(len(rel_labels)):
                    rel_tar = rel_labels[i].item()
                    if rel_tar == 0:
                        if self.zero_label_padding_mode == 'rand_insert':
                            random_idx = random.randint(0, num_groups - 1)
                            cur_chosen_matrix[random_idx].append(i)
                    else:
                        random_num = random.random()
                        for j in range(num_groups):
                            if j == 0:
                                if random_num <= 0.5:
                                    cur_chosen_matrix[j].append(i)
                            else:
                                if random_num <= self.sample_rate_matrix[-1][rel_tar]:
                                    cur_chosen_matrix[j].append(i)

                l = 1 - ((self.iter_2) / self.max_iter) ** 2

                group_input_a = prod_rep[cur_chosen_matrix[0]]
                group_label_a = rel_labels[cur_chosen_matrix[0]]
                group_pairs_a = pair_pred[cur_chosen_matrix[0]]
                union_feat_a = union_features[cur_chosen_matrix[0]]

                group_input_b = prod_rep[cur_chosen_matrix[1]]
                group_label_b = rel_labels[cur_chosen_matrix[1]]
                group_pairs_b = pair_pred[cur_chosen_matrix[1]]
                union_feat_b = union_features[cur_chosen_matrix[1]]

                group_len = min(len(group_input_a), len(group_input_b))

                group_input_a = self.post_cat(group_input_a)
                if self.use_vision:
                    if self.union_single_not_match:
                        group_input_a = group_input_a * self.up_dim(union_feat_a)
                    else:
                        group_input_a = group_input_a * union_feat_a
                group_input_b = self.post_cat_1(group_input_b)
                if self.use_vision:
                    if self.union_single_not_match:
                        group_input_b = group_input_b * self.up_dim(union_feat_b)
                    else:
                        group_input_b = group_input_b * union_feat_b
            else:
                num_groups = 2
                cur_chosen_matrix = []
        else:
            prod_rep = self.post_cat(prod_rep)
            if self.use_vision:
                if self.union_single_not_match:
                    prod_rep = prod_rep * self.up_dim(union_features)
                else:
                    prod_rep = prod_rep * union_features

        #TSNE
        if self.TSNE:
            if self.iter_2==5000:

                label_to_id_dict = {v: i for i, v in enumerate(np.unique(self.cur_chosen_label_tsne[1]))}  # 转成数字
                label_ids = np.array([label_to_id_dict[x] for x in self.cur_chosen_label_tsne[1]])
                # print(label_ids)

                fig = plt.figure(figsize=(10, 10))
                ax1 = fig.add_subplot(111)

                tsne_tensors=torch.stack(self.cur_chosen_tensor_tsne[1])
                tsne_tensors = np.array(tsne_tensors.cpu())
                np.save('/data/myk/SHA/TSNE/'+str(self.iter_2)+'reweight_'+'Motifs_'+'tensors'+'.npy', tsne_tensors)
                np.save('/data/myk/SHA/TSNE/' + str(self.iter_2) + 'reweight_' + 'Motifs_' +'labels'+ '.npy', label_ids)
                # 解释：Save an array to a binary file in NumPy .npy format。以“.npy”格式将数组保存到二进制文件中。
                # 参数：
                # file 要保存的文件名称，需指定文件保存路径，如果未设置，保存到默认路径。其文件拓展名为.npy
                # arr 为需要保存的数组，也即把数组arr保存至名称为file的文件中。

                plot_with_labels(visual(tsne_tensors), label_ids, '(a)')

                plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None,
                                    wspace=0.1, hspace=0.15)
                plt.legend(scatterpoints=1, labels=Label_Com, loc='best', labelspacing=0.2, columnspacing=0.4,
                           markerscale=0.2,
                           bbox_to_anchor=(0.9, 0), ncol=12, prop=font1, handletextpad=0.1)

                plt.savefig('/data/myk/SHA/TSNE/'+str(self.iter_2)+'reweight_'+'Motifs_'+'.png', format='png',dpi=300, bbox_inches='tight')
                plt.show()





            obj_dists_1, obj_preds_1, edge_ctx_1, _ = self.context_layer(roi_features_1, proposals_1, logger)

            # post decode
            edge_rep_1 = self.post_emb(edge_ctx_1)
            edge_rep_1 = edge_rep_1.view(edge_rep_1.size(0), 2, self.hidden_dim)
            head_rep_1 = edge_rep_1[:, 0].contiguous().view(-1, self.hidden_dim)
            tail_rep_1 = edge_rep_1[:, 1].contiguous().view(-1, self.hidden_dim)

            num_rels_1 = [r.shape[0] for r in rel_pair_idxs_1]
            num_objs_1 = [len(b) for b in proposals_1]
            assert len(num_rels_1) == len(num_objs_1)

            head_reps_1 = head_rep_1.split(num_objs, dim=0)
            tail_reps_1 = tail_rep_1.split(num_objs, dim=0)
            obj_preds_1 = obj_preds_1.split(num_objs, dim=0)

            prod_reps_1 = []
            pair_preds_1 = []
            for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs_1, head_reps_1, tail_reps_1, obj_preds_1):
                prod_reps_1.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
                pair_preds_1.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            prod_rep_1 = cat(prod_reps_1, dim=0)
            pair_pred_1 = cat(pair_preds_1, dim=0)

            prod_rep_1 = self.post_cat(prod_rep_1)
            if self.use_vision:
                if self.union_single_not_match:
                    prod_rep_1 = prod_rep_1 * self.up_dim(union_features_1)
                else:
                    prod_rep_1 = prod_rep_1 * union_features_1





            rel_labels_1 = cat(rel_labels_1, dim=0)
            for i in range(len(rel_labels_1)):
                rel_tar = rel_labels_1[i].item()
                if rel_tar == 0:
                    continue
                else:
                    random_num = random.random()
                    for j in range(self.num_groups_tsne):
                        if j == 0:
                            if rel_tar <= 15:
                                self.cur_chosen_label_tsne[j].append(rel_tar)
                                self.cur_chosen_tensor_tsne[j].append(prod_rep_1[i])
                        else:
                            if 15< rel_tar <= 30:
                                self.cur_chosen_label_tsne[j].append(rel_tar)
                                self.cur_chosen_tensor_tsne[j].append(prod_rep_1[i])






        #pred objects
        # random_num_subject = random.random()
        # prod_reps_subjects=[]
        # prod_reps_objects = []
        # prod_reps_object_labels = []
        # prod_reps_subject_labels = []
        # fg_labels_split = [proposal.get_field("labels") for proposal in proposals]
        # for pair_idx, head_rep, tail_rep, obj_pred,fglabels in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds,fg_labels_split):
        #         prod_reps_subjects.append(head_rep[pair_idx[:, 0]])
        #         prod_reps_objects.append(tail_rep[pair_idx[:, 1]])
        #         prod_reps_subject_labels.append(fglabels[pair_idx[:, 0]])
        #         prod_reps_object_labels.append(fglabels[pair_idx[:, 1]])
        # prod_rep_object = torch.cat((union_features,cat(prod_reps_subjects, dim=0)),dim=-1)
        # prod_rep_subject = torch.cat((union_features,cat(prod_reps_objects, dim=0)),dim=-1)
        # subject_labels = cat(prod_reps_subject_labels,dim=0)
        # object_labels = cat(prod_reps_object_labels, dim=0)

        '''begin to change'''
        # Eqinv

        # masked_feature = self.mlp(masked_feature_erm)
        # prod_rep = F.normalize(torch.sigmoid(self.mask_layer) * prod_rep,
        #                        dim=-1) * self.scaler

        add_losses = {}
        if self.training:
            # binary
            # random_num_subject = random.random()
            # prod_reps_false = []
            # for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            #     prod_reps_false.append(torch.cat((tail_rep[pair_idx[:, 1]], head_rep[pair_idx[:, 0]]), dim=-1))
            # prod_rep_false = cat(prod_reps_false, dim=0)
            # prod_rep_false = self.post_cat(prod_rep_false)
            # if self.use_vision:
            #     if self.union_single_not_match:
            #         prod_rep_false = prod_rep_false * self.up_dim(union_features)
            #
            #     else:
            #         prod_rep_false = prod_rep_false * union_features
            #
            # prod_rep_d=self.counter_binary(prod_rep)
            # prod_rep_false_d = self.counter_binary(prod_rep_false)
            # detect_labels_t=torch.zeros(len(prod_rep_d),2).cuda().float()
            # detect_labels_f = torch.zeros(len(prod_rep_d), 2).cuda().float()
            # detect_labels_t[:,1]=1.0
            # detect_labels_f[:, 0] = 1.0
            # prod_rep_d=F.softmax(prod_rep_d, 1)
            # prod_rep_false_d = F.softmax(prod_rep_false_d, 1)
            #
            # loss_b1 = F.binary_cross_entropy(prod_rep_d, detect_labels_t)
            # loss_b2 = F.binary_cross_entropy(prod_rep_false_d, detect_labels_f)
            # loss_b=loss_b1+loss_b2
            # add_losses['contb_loss'] = self.binary_weight*loss_b






            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj



            #eqinv
            # masked_feature_for_globalcont_norm = F.normalize(self.mlp(prod_rep), dim=-1)
            # loss_cont = 1.0 * self.info_nce_loss_supervised(masked_feature_for_globalcont_norm.unsqueeze(1),
            #                                                 masked_feature_for_globalcont_norm.size(0),
            #                                                 temperature=0.1, labels=rel_labels)
            # add_losses['cont_loss'] = loss_cont



            # pred objects
            # if random_num_subject<=0.5:
            # subjects_inf=self.subject_inference(prod_rep_subject)
            # loss_inf_subject = self.criterion_loss(subjects_inf, subject_labels.long())
            # add_losses['subject_loss'] = loss_inf_subject
            # else:
            # objects_inf=self.object_inference(prod_rep_object)
            # loss_inf_object = self.criterion_loss(objects_inf, object_labels.long())
            # add_losses['object_loss'] = loss_inf_object

            # split group
            # num_groups_1 = self.incre_idx_list[max_label.item()]
            # if num_groups_1 == 0:
            #     num_groups_1 = max(self.incre_idx_list)
            # cur_chosen_matrix_1 = []
            # for i in range(num_groups_1):
            #     cur_chosen_matrix_1.append([])


            #BBN
            if self.BBN:
                mixed_feature = 2 * torch.cat((l * group_input_a[:group_len], (1 - l) * group_input_b[:group_len]), dim=1)
                output_BBN = self.rel_compress_BBN(mixed_feature)

                group_label_a = group_label_a[:group_len]
                group_label_b = group_label_b[:group_len]

                if self.use_bias:
                    output_BBN = output_BBN + l * self.freq_bias.index_with_labels(
                        (group_pairs_a[:group_len].long())) + (1 - l) * self.freq_bias.index_with_labels(
                        group_pairs_b[:group_len].long())

                    add_losses['CE_loss'] = l * self.criterion_loss(output_BBN, group_label_a)

                    add_losses['CE_bias_loss'] = (1 - l) * self.criterion_loss(output_BBN, group_label_b)
                    return None, None, add_losses

            rel_labels = cat(rel_labels, dim=0)
            max_label = max(rel_labels)




            num_groups=3
            cur_chosen_matrix = []
            for i in range(num_groups):
                cur_chosen_matrix.append([])



            # object class
            # object_gts = []
            # #object_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            # for obj_pair_idx,proposal in zip(rel_pair_idxs,proposals):
            #     object_label=proposal.get_field("labels")
            #     object_gts.append(torch.stack((object_label[obj_pair_idx[:, 0]], object_label[obj_pair_idx[:, 1]]), dim=1))
            # object_gt = cat(pair_preds, dim=0)
            # for i in range(len(rel_labels)):
            #     rel_tar = rel_labels[i].item()
            #     sub_class = pair_pred[i][0].item()
            #     obj_class = pair_pred[i][1].item()
            #     #sum_rate=(self.class_stat[sub_class]+self.class_stat[obj_class])/2
            #     rel_tar_word=self.class_names_r[rel_tar]
            #     sub_class_word=self.class_names[sub_class]
            #     obj_class_word = self.class_names[obj_class]
            #     triplet=(rel_tar_word,sub_class_word,obj_class_word)
            #     if  triplet in self.importance_dic_weight.keys():
            #         samplerate = self.importance_dic_weight[triplet]
            #     else:
            #         samplerate = self.weight_mean
            #
            #     if rel_tar == 0:
            #         if self.zero_label_padding_mode == 'rand_insert':
            #             random_idx = random.randint(0, num_groups - 1)
            #             cur_chosen_matrix[random_idx].append(i)
            #         elif self.zero_label_padding_mode == 'rand_choose' or self.zero_label_padding_mode == 'all_include':
            #             if self.zero_label_padding_mode == 'rand_choose':
            #                 rand_zeros = random.random()
            #             else:
            #                 rand_zeros = 1.0
            #             if rand_zeros >= 0.4:
            #                 for zix in range(len(cur_chosen_matrix)):
            #                     cur_chosen_matrix[zix].append(i)
            #     else:
            #         random_num = random.random()
            #         for j in range(num_groups):
            #             if j==0:
            #                 if random_num<=0.5:
            #                     cur_chosen_matrix[j].append(i)
            #             else:
            #                 if random_num<=samplerate:
            #                     cur_chosen_matrix[j].append(i)
            #
            # for i in range(num_groups):
            #     if len(prod_rep) == 0:
            #         print("#############################feat==0########################################")
            #         break
            #
            #     if max_label == 0:
            #         group_input = prod_rep
            #         group_label = rel_labels
            #         group_pairs = pair_pred
            #     else:
            #         group_input = prod_rep[cur_chosen_matrix[i]]
            #         group_label = rel_labels[cur_chosen_matrix[i]]
            #         group_pairs = pair_pred[cur_chosen_matrix[i]]
            #
            #     '''count Cross Entropy loss'''
            #     jdx = i
            #     group_output_now= self.rel_compress(group_input)
            #     if self.use_bias:
            #         group_output_now = group_output_now + self.freq_bias.index_with_labels(group_pairs.long())
            #     if i==2:
            #         weights = torch.ones_like(group_label,dtype=torch.float16)
            #         for i in range(len(group_label)):
            #             rel_tar = group_label[i].item()
            #             sub_class = group_pairs[i][0].item()
            #             obj_class = group_pairs[i][1].item()
            #             rel_tar_word = self.class_names_r[rel_tar]
            #             sub_class_word = self.class_names[sub_class]
            #             obj_class_word = self.class_names[obj_class]
            #             triplet = (rel_tar_word, sub_class_word, obj_class_word)
            #             if triplet in self.importance_dic.keys():
            #                 weights[i] = self.importance_dic[triplet]
            #             else:
            #                 if group_label[i].item()==0:
            #                     weights[i] = self.min_weight
            #                 else:
            #                     weights[i] = self.mean_weight
            #
            #             #weights[i] = (self.class_stat[sub_class] + self.class_stat[obj_class]) / 2
            #
            #         add_losses['%d_CE_bias_loss' % (jdx + 1)] =self.overweighted_weight*self.rel_criterion_loss(group_output_now, group_label,weights)
            #         add_losses['%d_reg_bias_loss' % (jdx + 1)] =self.overweighted_weight_p*self.penalty(group_output_now, group_label,weights)
            #     else:
            #         if i == 1:
            #             add_losses['%d_CE_loss' % (jdx + 1)] = self.weighted_weight*self.criterion_loss(group_output_now, group_label)
            #             add_losses['%d_reg_loss' % (jdx + 1)] = self.weighted_weight_p*self.penalty(group_output_now, group_label)
            #         else:
            #             if i==0:
            #                 add_losses['%d_CE_loss' % (jdx + 1)] = self.norm_weight*self.criterion_loss(group_output_now, group_label)
            #                 add_losses['%d_reg_loss' % (jdx + 1)] = self.norm_weight_p*self.penalty(group_output_now, group_label)
            #
            #




            # relation class
            for i in range(len(rel_labels)):
                rel_tar = rel_labels[i].item()
                if rel_tar == 0:
                    if self.zero_label_padding_mode == 'rand_insert':
                        random_idx = random.randint(0, num_groups - 1)
                        cur_chosen_matrix[random_idx].append(i)
                        #random_idx1 = random.randint(0, num_groups_1 - 1)
                        #cur_chosen_matrix_1[random_idx1].append(i)
                    elif self.zero_label_padding_mode == 'rand_choose' or self.zero_label_padding_mode == 'all_include':
                        if self.zero_label_padding_mode == 'rand_choose':
                            rand_zeros = random.random()
                        else:
                            rand_zeros = 1.0
                        if rand_zeros >= 0.4:
                            for zix in range(len(cur_chosen_matrix)):
                                cur_chosen_matrix[zix].append(i)
                else:
                    # rel_idx = self.incre_idx_list[rel_tar]
                    # random_num = random.random()
                    # for j in range(num_groups):
                    #     act_idx = num_groups - j
                    #     threshold_cur = self.sample_rate_matrix[act_idx - 1][rel_tar]
                    #     if random_num <= threshold_cur or act_idx < rel_idx:
                    #         # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
                    #         for k in range(act_idx):
                    #             cur_chosen_matrix[k].append(i)
                    #         break

                    #resample
                    # rel_idx = self.incre_idx_list[rel_tar]
                    # cur_chosen_matrix_1[rel_idx-1].append(i)
                    random_num = random.random()
                    for j in range(num_groups):
                        if j==0:
                            if random_num<=0.5:
                                cur_chosen_matrix[j].append(i)
                        else:
                            if random_num<=self.sample_rate_matrix[-1][rel_tar]:
                                cur_chosen_matrix[j].append(i)

            for i in range(num_groups):
                if len(prod_rep) == 0:
                    print("#############################feat==0########################################")
                    break

                if max_label == 0:
                    group_input = prod_rep
                    #group_input = union_features
                    group_label = rel_labels
                    group_pairs = pair_pred
                else:
                    group_input = prod_rep[cur_chosen_matrix[i]]
                    #group_input=union_features[cur_chosen_matrix[i]]
                    group_label = rel_labels[cur_chosen_matrix[i]]
                    group_pairs = pair_pred[cur_chosen_matrix[i]]

                '''count Cross Entropy loss'''
                jdx = i
                group_output_now= self.rel_compress(group_input)
                #group_output_now_bias = self.rel_compress_biased(group_input)
                #group_output_now_overbias = self.rel_compress_overbiased(group_input)
                if self.use_bias:
                    group_output_now = group_output_now + self.freq_bias.index_with_labels(group_pairs.long())
                    #group_output_now_bias=group_output_now_bias + self.freq_bias.index_with_labels(group_pairs.long())
                    #group_output_now_overbias = group_output_now_overbias + self.freq_bias.index_with_labels(group_pairs.long())
                #add_losses['%d_CE_loss' % (jdx + 1)] = self.criterion_loss(group_output_now, group_label)
                if i==2:
                    weights = torch.ones_like(group_label,dtype=torch.float16)
                    for i in range(len(group_label)):
                        weights[i] = self.weight_rate_matrix[-1][group_label[i]]
                    # idx=group_label.nonzero()
                    # for i in range(len(idx)):
                    #     weights[idx[i]] = self.weight_rate_matrix[-1][group_label[idx[i]]]
                    add_losses['%d_CE_bias_loss' % (jdx + 1)] =self.overweighted_weight*self.rel_criterion_loss(group_output_now, group_label,weights)
                    add_losses['%d_reg_bias_loss' % (jdx + 1)] =self.overweighted_weight_p*self.penalty(group_output_now, group_label,weights)
                    #add_losses['%d_proxy_loss' % (jdx + 1)] = self.softtripletloss(group_input, group_label,weights)
                else:
                    if i == 1:
                        add_losses['%d_CE_loss' % (jdx + 1)] = self.weighted_weight*self.criterion_loss(group_output_now, group_label)
                        add_losses['%d_reg_loss' % (jdx + 1)] = self.weighted_weight_p*self.penalty(group_output_now, group_label)
                        #add_losses['%d_proxy_loss' % (jdx + 1)] = self.softtripletloss(group_input, group_label)
                    else:
                        add_losses['%d_CE_loss' % (jdx + 1)] = self.norm_weight*self.criterion_loss(group_output_now, group_label)
                        add_losses['%d_reg_loss' % (jdx + 1)] = self.norm_weight_p*self.penalty(group_output_now, group_label)
                        #add_losses['%d_proxy_loss' % (jdx + 1)] = self.softtripletloss(group_input, group_label)


                #add_losses['%d_CEoverbias_loss' % (jdx + 1)] = self.rel_criterion_loss(group_output_now_overbias, group_label, torch.pow(weights,2))
                #if max_label != 0 and jdx < 2:
                #add_losses['%d_CEbias_loss' % (jdx + 1)] = self.criterion_loss(group_output_now_bias, group_label)
                #add_losses['%d_xERM_loss' % (jdx + 1)] = self.xERMloss(group_output_now,group_output_now_bias, group_label)

                #deepBDC
                # if max_label != 0 and jdx>1 and jdx<4:
                #     #print("prod_rep",prod_rep)
                #     bdc_feat=self.dcov(group_input)
                #     bdc_feat=self.dropout(bdc_feat)
                #     bdc_out=self.bdcclshead(bdc_feat)
                #     add_losses['%d_bdcce_loss' % (jdx + 1)] = self.rel_criterion_loss(bdc_out, group_label, weights)


                #proxy
                #     add_losses['%d_proxy_loss' % (jdx + 1)] = self.softtripletloss(group_input, group_label,weights)


                # rel_classier_now = self.rel_classifer_all[jdx]
                # weights = torch.ones_like(group_label,dtype=torch.float16)
                # for i in range(len(group_label)):
                #     weights[i] = self.weight_rate_matrix[-1][group_label[i]]
                # group_output_now = rel_classier_now(group_input)
                # if self.use_bias:
                #     # rel_bias_now = self.freq_bias_all[jdx]
                #     # group_output_now = group_output_now + rel_bias_now.index_with_labels(group_pairs.long())
                #     group_output_now = group_output_now + self.freq_bias.index_with_labels(group_pairs.long())
                # # actual_label_piece: if label is out of range, then filter it to ensure the training can continue
                # actual_label_now = self.pre_group_matrix[jdx][group_label]
                # add_losses['%d_CE_loss' % (jdx + 1)] = self.rel_criterion_loss(group_output_now, group_label,weights)




                # if self.Knowledge_Transfer_Mode == 'KL_logit_Neighbor':
                #     if i > 0:
                #         '''count knowledge transfer loss'''
                #         jbef = i - 1
                #         rel_classier_bef = self.rel_classifer_all[jbef]
                #         group_output_bef = rel_classier_bef(group_input)
                #         if self.use_bias:
                #             rel_bias_bef = self.freq_bias_all[jbef]
                #             group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                #         max_vector = self.max_elemnt_list[jbef] + 1
                #
                #         if self.no_relation_restrain:
                #             kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                #             kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                #                                            reduce=False)
                #             kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                #             kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                #         else:
                #             kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                #                                            reduce=True)
                #             kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                #         add_losses['%d%d_kl_loss' % (jbef + 1, jdx + 1)] = kd_loss_final
                #
                # elif self.Knowledge_Transfer_Mode == 'KL_logit_TopDown':
                #     layer_total_loss = 0
                #     for jbef in range(i):
                #         rel_classier_bef = self.rel_classifer_all[jbef]
                #         group_output_bef = rel_classier_bef(group_input)
                #         if self.use_bias:
                #             rel_bias_bef = self.freq_bias_all[jbef]
                #             group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                #         # kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                #         max_vector = self.max_elemnt_list[jbef] + 1
                #
                #         if self.no_relation_restrain:
                #             kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                #             kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                #                                            reduce=False)
                #             kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                #             kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                #         else:
                #             kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                #                                            reduce=True)
                #             kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                #         layer_total_loss += kd_loss_final
                #
                #     if i > 0:
                #         add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss
                # elif self.Knowledge_Transfer_Mode == 'KL_logit_BiDirection':
                #     layer_total_loss = 0
                #     for jbef in range(i):
                #         rel_classier_bef = self.rel_classifer_all[jbef]
                #         group_output_bef = rel_classier_bef(group_input)
                #         if self.use_bias:
                #             rel_bias_bef = self.freq_bias_all[jbef]
                #             group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                #         # kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                #         max_vector = self.max_elemnt_list[jbef] + 1
                #
                #         if self.no_relation_restrain:
                #             kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                #             kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                #                                               group_output_now[:, 1:max_vector],
                #                                               reduce=False)
                #             kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                #                                               group_output_bef[:, 1:],
                #                                               reduce=False)
                #             kd_loss_vecify = (kd_loss_matrix_td + kd_loss_matrix_bu) * kd_choice_vector
                #             kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                #         else:
                #             kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                #                                               group_output_now[:, 1:max_vector],
                #                                               reduce=True)
                #             kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                #                                               group_output_bef[:, 1:],
                #                                               reduce=True)
                #             kd_loss_final = self.knowledge_loss_coefficient * (kd_loss_matrix_td + kd_loss_matrix_bu)
                #         layer_total_loss += kd_loss_final
                #
                #     if i > 0:
                #         add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss
            return None, None, add_losses
        else:
            if self.test_flag==0:
                self.test_flag+=1

            if self.BBN:
                prod_rep_a = self.post_cat(prod_rep)
                prod_rep_b = self.post_cat_1(prod_rep)
                if self.use_vision:
                    if self.union_single_not_match:
                        prod_rep_a = prod_rep_a * self.up_dim(union_features)
                        prod_rep_b = prod_rep_b * self.up_dim(union_features)
                    else:
                        prod_rep_a = prod_rep_a * union_features
                        prod_rep_b = prod_rep_b * union_features


                mixed_feature = torch.cat((prod_rep_a,prod_rep_b),dim=1)
                output_BBN = self.rel_compress_BBN(mixed_feature)
                if self.use_bias:
                    rel_dists = output_BBN + self.freq_bias.index_with_labels(pair_pred.long())
                    rel_dists = rel_dists.split(num_rels, dim=0)
                    obj_dists = obj_dists.split(num_objs, dim=0)

                    return obj_dists, rel_dists, add_losses
            else:
                    rel_dists = self.rel_compress(prod_rep)
                    #rel_classier_test = self.rel_classifer_all[-1]
                    # rel_dists = self.rel_classifer_all[0](prod_rep)+self.rel_classifer_all[1](prod_rep)+self.rel_classifer_all[2](prod_rep)+self.rel_classifer_all[3](prod_rep)+self.rel_classifer_all[4](prod_rep)
                    if self.use_bias:
                        rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())
                        #rel_bias_test = self.freq_bias_all[-1]
                        #rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())
                    rel_dists = rel_dists.split(num_rels, dim=0)
                    obj_dists = obj_dists.split(num_objs, dim=0)

                    return obj_dists, rel_dists, add_losses

    def generate_muti_networks(self, num_cls):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        # self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
        # self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
        # self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
        # self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)


        self.rel_classifer_1 = nn.Linear(self.pooling_dim, 50 + 1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, 50 + 1)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, 50 + 1)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, 50 + 1)



        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        layer_init(self.rel_classifer_4, xavier=True)
        if num_cls == 4:
            classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
        elif num_cls < 4:
            exit('wrong num in compress_all')
        else:
            self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_classifer_5, xavier=True)
            if num_cls == 5:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                 self.rel_classifer_4, self.rel_classifer_5]
            else:
                self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_classifer_6, xavier=True)
                if num_cls == 6:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                else:
                    self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_classifer_7, xavier=True)
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                     self.rel_classifer_7]
                    if num_cls > 7:
                        exit('wrong num in compress_all')
        return classifer_all

    def generate_multi_bias(self, config, statistics, num_cls):
        self.freq_bias_1 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[0])
        self.freq_bias_2 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[1])
        self.freq_bias_3 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[2])
        self.freq_bias_4 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[3])
        if num_cls < 4:
            exit('wrong num in multi_bias')
        elif num_cls == 4:
            freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4]
        else:
            self.freq_bias_5 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[4])
            if num_cls == 5:
                freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4,
                                 self.freq_bias_5]
            else:
                self.freq_bias_6 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                      predicate_all_list=self.bias_for_group_split[5])
                if num_cls == 6:
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6]
                else:
                    self.freq_bias_7 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                          predicate_all_list=self.bias_for_group_split[6])
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6, self.freq_bias_7]
                    if num_cls > 7:
                        exit('wrong num in multi_bias')
        return freq_bias_all

@registry.ROI_RELATION_PREDICTOR.register("VCTree_GCL")
class VCTree_GCL(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTree_GCL, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_bias = config.GLOBAL_SETTING.USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # get model configs
        self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        # generate the auxiliary lists
        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, config.GLOBAL_SETTING.DATASET_CHOICE)
        self.sample_rate_matrix = generate_sample_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE, self.max_group_element_number_list)
        self.weight_rate_matrix = generate_weight_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE,
                                                              self.max_group_element_number_list)
        self.bias_for_group_split = generate_current_sequence_for_bias(num_of_group_element_list, config.GLOBAL_SETTING.DATASET_CHOICE)

        self.num_groups = len(self.max_elemnt_list)
        self.rel_classifer_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()
        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)
        if self.Knowledge_Transfer_Mode != 'None':
            self.NLL_Loss = nn.NLLLoss()
            self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
            self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
        self.criterion_loss = nn.CrossEntropyLoss()

        #3samples
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.freq_bias = FrequencyBias(config, statistics)
        self.rel_criterion_loss = ReweightingCE()

        #sample-weighted
        self.norm_weight=0.1
        self.overweighted_weight=1


        self.iter = 0
        self.iter_2 = 0
        self.max_iter = 120000

    def penalty(self,logits, y, weights=None):
        scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
        logits_f=logits*scale
        loss = self.CE_loss(logits_f, y)
        if weights != None:
            x = F.log_softmax(logits_f, 1)
            target_onehot = torch.FloatTensor(y.size(0), 51).cuda()
            #GQA200
            #target_onehot=torch.FloatTensor(y.size(0),101).cuda()
            target_onehot.zero_()
            target_onehot.scatter_(1,y.view(-1,1),1)
            celoss = torch.sum(- x * target_onehot, dim=1)*weights
            celoss=torch.mean(celoss)
            grad = torch.autograd.grad(celoss, [scale], create_graph=True)[0]
        else:
            grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]


        fin_loss=torch.sum(grad ** 2)
        return fin_loss


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        self.iter_2 += 1
        if self.iter_2 > 30000:
            self.iter += 1

        alpha = 1 - (self.iter / 30000)
        alpha = max(alpha, 0.1)
        alpha = alpha * 0.9
        if self.iter_2 <= 30000:
            alpha = 0.9

        self.norm_weight = alpha


        self.overweighted_weight = 1 - alpha


        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        prod_rep = prod_rep * union_features

        add_losses = {}
        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj

            rel_labels = cat(rel_labels, dim=0)
            max_label = max(rel_labels)

            # num_groups = self.incre_idx_list[max_label.item()]
            # if num_groups == 0:
            #     num_groups = max(self.incre_idx_list)



            #3samples
            num_groups=3
            cur_chosen_matrix = []

            for i in range(num_groups):
                cur_chosen_matrix.append([])

            for i in range(len(rel_labels)):
                rel_tar = rel_labels[i].item()
                if rel_tar == 0:
                    if self.zero_label_padding_mode == 'rand_insert':
                        random_idx = random.randint(0, num_groups - 1)
                        cur_chosen_matrix[random_idx].append(i)
                    elif self.zero_label_padding_mode == 'rand_choose' or self.zero_label_padding_mode == 'all_include':
                        if self.zero_label_padding_mode == 'rand_choose':
                            rand_zeros = random.random()
                        else:
                            rand_zeros = 1.0
                        if rand_zeros >= 0.4:
                            for zix in range(len(cur_chosen_matrix)):
                                cur_chosen_matrix[zix].append(i)
                else:
                    # rel_idx = self.incre_idx_list[rel_tar]
                    # random_num = random.random()
                    # for j in range(num_groups):
                    #     act_idx = num_groups - j
                    #     threshold_cur = self.sample_rate_matrix[act_idx - 1][rel_tar]
                    #     if random_num <= threshold_cur or act_idx < rel_idx:
                    #         # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
                    #         for k in range(act_idx):
                    #             cur_chosen_matrix[k].append(i)
                    #         break
                    random_num = random.random()
                    for j in range(num_groups):
                        if j==0:
                            if random_num<=0.5:
                                cur_chosen_matrix[j].append(i)
                        else:
                            if random_num<=self.sample_rate_matrix[-1][rel_tar]:
                                cur_chosen_matrix[j].append(i)


            for i in range(num_groups):
                if max_label == 0:
                    group_input = prod_rep
                    group_label = rel_labels
                    group_pairs = pair_pred
                else:
                    group_input = prod_rep[cur_chosen_matrix[i]]
                    group_label = rel_labels[cur_chosen_matrix[i]]
                    group_pairs = pair_pred[cur_chosen_matrix[i]]
                jdx = i
                group_output_now= self.ctx_compress(group_input)
                #group_output_now_bias = self.rel_compress_biased(group_input)
                #group_output_now_overbias = self.rel_compress_overbiased(group_input)
                if self.use_bias:
                    group_output_now = group_output_now + self.freq_bias.index_with_labels(group_pairs.long())
                    #group_output_now_bias=group_output_now_bias + self.freq_bias.index_with_labels(group_pairs.long())
                    #group_output_now_overbias = group_output_now_overbias + self.freq_bias.index_with_labels(group_pairs.long())
                #add_losses['%d_CE_loss' % (jdx + 1)] = self.criterion_loss(group_output_now, group_label)
                if i==2:
                    weights = torch.ones_like(group_label,dtype=torch.float16)
                    for i in range(len(group_label)):
                        weights[i] = self.weight_rate_matrix[-1][group_label[i]]
                    #idx=group_label.nonzero()
                    # for i in range(len(idx)):
                    #     weights[idx[i]] = self.weight_rate_matrix[-1][group_label[idx[i]]]


                    add_losses['%d_CE_bias_loss' % (jdx + 1)] = self.overweighted_weight * self.rel_criterion_loss(
                        group_output_now, group_label, weights)
                    add_losses['%d_reg_bias_loss' % (jdx + 1)] = self.overweighted_weight * self.penalty(
                        group_output_now, group_label, weights)
                    # add_losses['%d_proxy_loss' % (jdx + 1)] = self.softtripletloss(group_input, group_label,weights)
                else:
                    if i == 1:
                        add_losses['%d_CE_loss' % (jdx + 1)] = self.criterion_loss(group_output_now, group_label)
                        add_losses['%d_reg_loss' % (jdx + 1)] = self.penalty(group_output_now, group_label)
                    # add_losses['%d_proxy_loss' % (jdx + 1)] = self.softtripletloss(group_input, group_label)
                    else:
                        add_losses['%d_CE_loss' % (jdx + 1)] = self.norm_weight * self.criterion_loss(group_output_now,
                                                                                                  group_label)
                        add_losses['%d_reg_loss' % (jdx + 1)] = self.norm_weight * self.penalty(group_output_now,
                                                                                            group_label)
                    # add_losses['%d_proxy_loss' % (jdx + 1)] = self.softtripletloss(group_input, group_label)
            #
            #     '''count Cross Entropy loss'''
            #     jdx = i
            #     rel_classier_now = self.rel_classifer_all[jdx]
            #     group_output_now = rel_classier_now(group_input)
            #     if self.use_bias:
            #         rel_bias_now = self.freq_bias_all[jdx]
            #         group_output_now = group_output_now + rel_bias_now.index_with_labels(group_pairs.long())
            #     # actual_label_piece: if label is out of range, then filter it to ensure the training can continue
            #     actual_label_now = self.pre_group_matrix[jdx][group_label]
            #     add_losses['%d_CE_loss' % (jdx + 1)] = self.CE_loss(group_output_now, actual_label_now)
            #
            #     if self.Knowledge_Transfer_Mode == 'KL_logit_Neighbor':
            #         if i > 0:
            #             '''count knowledge transfer loss'''
            #             jbef = i - 1
            #             rel_classier_bef = self.rel_classifer_all[jbef]
            #             group_output_bef = rel_classier_bef(group_input)
            #             if self.use_bias:
            #                 rel_bias_bef = self.freq_bias_all[jbef]
            #                 group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
            #             max_vector = self.max_elemnt_list[jbef] + 1
            #
            #             if self.no_relation_restrain:
            #                 kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
            #                 kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
            #                                                reduce=False)
            #                 kd_loss_vecify = kd_loss_matrix * kd_choice_vector
            #                 kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
            #             else:
            #                 kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
            #                                                reduce=True)
            #                 kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
            #             add_losses['%d%d_kl_loss' % (jbef + 1, jdx + 1)] = kd_loss_final
            #
            #     elif self.Knowledge_Transfer_Mode == 'KL_logit_TopDown':
            #         layer_total_loss = 0
            #         for jbef in range(i):
            #             rel_classier_bef = self.rel_classifer_all[jbef]
            #             group_output_bef = rel_classier_bef(group_input)
            #             if self.use_bias:
            #                 rel_bias_bef = self.freq_bias_all[jbef]
            #                 group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
            #             # kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
            #             max_vector = self.max_elemnt_list[jbef] + 1
            #
            #             if self.no_relation_restrain:
            #                 kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
            #                 kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
            #                                                reduce=False)
            #                 kd_loss_vecify = kd_loss_matrix * kd_choice_vector
            #                 kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
            #             else:
            #                 kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
            #                                                reduce=True)
            #                 kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
            #             layer_total_loss += kd_loss_final
            #
            #         if i > 0:
            #             add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss
            #     elif self.Knowledge_Transfer_Mode == 'KL_logit_BiDirection':
            #         layer_total_loss = 0
            #         for jbef in range(i):
            #             rel_classier_bef = self.rel_classifer_all[jbef]
            #             group_output_bef = rel_classier_bef(group_input)
            #             if self.use_bias:
            #                 rel_bias_bef = self.freq_bias_all[jbef]
            #                 group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
            #             # kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
            #             max_vector = self.max_elemnt_list[jbef] + 1
            #
            #             if self.no_relation_restrain:
            #                 kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
            #                 kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
            #                                                   group_output_now[:, 1:max_vector],
            #                                                   reduce=False)
            #                 kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
            #                                                   group_output_bef[:, 1:],
            #                                                   reduce=False)
            #                 kd_loss_vecify = (kd_loss_matrix_td + kd_loss_matrix_bu) * kd_choice_vector
            #                 kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
            #             else:
            #                 kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
            #                                                   group_output_now[:, 1:max_vector],
            #                                                   reduce=True)
            #                 kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
            #                                                   group_output_bef[:, 1:],
            #                                                   reduce=True)
            #                 kd_loss_final = self.knowledge_loss_coefficient * (kd_loss_matrix_td + kd_loss_matrix_bu)
            #             layer_total_loss += kd_loss_final
            #
            #         if i > 0:
            #             add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss
            return None, None, add_losses
        else:
            rel_dists = self.ctx_compress(prod_rep)
            #rel_classier_test = self.rel_classifer_all[-1]
            # rel_dists = self.rel_classifer_all[0](prod_rep)+self.rel_classifer_all[1](prod_rep)+self.rel_classifer_all[2](prod_rep)+self.rel_classifer_all[3](prod_rep)+self.rel_classifer_all[4](prod_rep)
            if self.use_bias:
                rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())
            # rel_classier_test = self.rel_classifer_all[-1]
            # rel_dists = rel_classier_test(prod_rep)
            # if self.use_bias:
            #     rel_bias_test = self.freq_bias_all[-1]
            #     rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())
            rel_dists = rel_dists.split(num_rels, dim=0)
            obj_dists = obj_dists.split(num_objs, dim=0)

            return obj_dists, rel_dists, add_losses


        ctx_dists = self.ctx_compress(prod_rep * union_features)
        # uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists

        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)
            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj
            rel_labels = cat(rel_labels, dim=0)
            add_losses['rel_loss'] = self.criterion_loss(rel_dists, rel_labels)
            return None, None, add_losses
        else:
            obj_dists = obj_dists.split(num_objs, dim=0)
            rel_dists = rel_dists.split(num_rels, dim=0)
            return obj_dists, rel_dists, add_losses

    def generate_muti_networks(self, num_cls):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)
        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        layer_init(self.rel_classifer_4, xavier=True)
        if num_cls == 4:
            classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
        elif num_cls < 4:
            exit('wrong num in compress_all')
        else:
            self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_classifer_5, xavier=True)
            if num_cls == 5:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                 self.rel_classifer_4, self.rel_classifer_5]
            else:
                self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_classifer_6, xavier=True)
                if num_cls == 6:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                else:
                    self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_classifer_7, xavier=True)
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                     self.rel_classifer_7]
                    if num_cls > 7:
                        exit('wrong num in compress_all')
        return classifer_all

    def generate_multi_bias(self, config, statistics, num_cls):
        self.freq_bias_1 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[0])
        self.freq_bias_2 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[1])
        self.freq_bias_3 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[2])
        self.freq_bias_4 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[3])
        if num_cls < 4:
            exit('wrong num in multi_bias')
        elif num_cls == 4:
            freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4]
        else:
            self.freq_bias_5 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[4])
            if num_cls == 5:
                freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4,
                                 self.freq_bias_5]
            else:
                self.freq_bias_6 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                      predicate_all_list=self.bias_for_group_split[5])
                if num_cls == 6:
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6]
                else:
                    self.freq_bias_7 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                          predicate_all_list=self.bias_for_group_split[6])
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6, self.freq_bias_7]
                    if num_cls > 7:
                        exit('wrong num in multi_bias')
        return freq_bias_all

@registry.ROI_RELATION_PREDICTOR.register("TransLikePredictor")
class TransLikePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransLikePredictor, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False
        self.use_bias = True
        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)
        self.criterion_loss = nn.CrossEntropyLoss()


        self.use_causal= True
        self.causal_weight = 2

        if self.use_causal==True:
            if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
                self.qhat = self.initial_qhat(class_num=51)
            else:
                self.qhat = self.initial_qhat(class_num=101)

    def initial_qhat(self,class_num=1000):
            # initialize qhat of predictions (probability)
            qhat = (torch.ones([1, class_num], dtype=torch.float) / class_num).cuda()
            print("qhat size: ".format(qhat.size()))
            return qhat

    def update_qhat(self,probs, qhat, momentum, qhat_mask=None):
        if qhat_mask is not None:
            mean_prob = probs.detach() * qhat_mask.detach().unsqueeze(dim=-1)
        else:
            mean_prob = probs.detach().mean(dim=0)
        qhat = momentum * qhat + (1 - momentum) * mean_prob
        return qhat



    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())


        rel_dists = rel_dists + frq_dists
        add_losses = {}

        if self.training:
            if self.use_causal:
                self.qhat = self.update_qhat(torch.softmax(rel_dists.detach(), dim=-1), self.qhat,
                                             momentum=0.99)
                delta_logits = torch.log(self.qhat)
                rel_dists = rel_dists + self.causal_weight * delta_logits

            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj
            rel_labels = cat(rel_labels, dim=0)
            add_losses['rel_loss'] = self.criterion_loss(rel_dists, rel_labels)
            return None, None, add_losses
        else:
            obj_dists = obj_dists.split(num_objs, dim=0)
            rel_dists = rel_dists.split(num_rels, dim=0)
            return obj_dists, rel_dists, add_losses


def make_roi_relation_predictor(cfg, in_channels):
    import time
    result_str = '---'*20
    result_str += ('\n\nthe dataset we use is [ %s ]' % cfg.GLOBAL_SETTING.DATASET_CHOICE)
    if cfg.GLOBAL_SETTING.USE_BIAS:
        result_str += ('\nwe use [ bias ]!')
    else:
        result_str += ('\nwe do [ not ] use bias!')
    result_str += ('\nthe model we use is [ %s ]' % cfg.GLOBAL_SETTING.RELATION_PREDICTOR)
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == True and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == True:
        result_str += ('\ntraining mode is [ predcls ]')
    elif cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == False and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == True:
        result_str += ('\ntraining mode is [ sgcls ]')
    elif cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == False and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == False:
        result_str += ('\ntraining mode is [ sgdet ]')
    else:
        exit('wrong training mode!')
    result_str += ('\nlearning rate is [ %.5f ]' % cfg.SOLVER.BASE_LR)
    result_str += ('\nthe knowledge distillation strategy is [ %s ]' % cfg.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE)
    assert cfg.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE in ['None', 'KL_logit_Neighbor', 'KL_logit_None',
                                               'KL_logit_TopDown', 'KL_logit_BottomUp', 'KL_logit_BiDirection']
    if cfg.GLOBAL_SETTING.RELATION_PREDICTOR in ['TransLike_GCL', 'TransLikePredictor']:
        result_str += ('\nrel labels=0 is use [ %s ] to process' % cfg.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE)
        assert cfg.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE in ['rand_insert', 'rand_choose', 'all_include']
        assert cfg.GLOBAL_SETTING.BASIC_ENCODER in ['Self-Attention', 'Cross-Attention', 'Hybrid-Attention']
        result_str += ('\n-----Transformer layer is [ %d ] in obj and [ %d ] in rel' %
                       (cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER,
                        cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER))
        result_str += ('\n-----Transformer mode is [ %s ]' % cfg.GLOBAL_SETTING.BASIC_ENCODER)
    if cfg.GLOBAL_SETTING.RELATION_PREDICTOR in ['MotifsLike_GCL', 'MotifsLikePredictor']:
        assert cfg.GLOBAL_SETTING.BASIC_ENCODER in ['Motifs', 'VTransE']
        result_str += ('\n-----Model mode is [ %s ]' % cfg.GLOBAL_SETTING.BASIC_ENCODER)

    num_of_group_element_list, predicate_stage_count = get_group_splits(cfg.GLOBAL_SETTING.DATASET_CHOICE, cfg.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE)
    max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
    incre_idx_list, max_elemnt_list, group_matrix, kd_matrix = get_current_predicate_idx(
        num_of_group_element_list, cfg.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_PENALTY, cfg.GLOBAL_SETTING.DATASET_CHOICE)
    result_str += ('\n   the number of elements in each group is {}'.format(incre_idx_list))
    result_str += ('\n   incremental stage list is {}'.format(num_of_group_element_list))
    result_str += ('\n   the length of each line in group is {}'.format(predicate_stage_count))
    result_str += ('\n   the max number of elements in each group is {}'.format(max_group_element_number_list))
    result_str += ('\n   the knowledge distillation strategy is [ %s ]' % cfg.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE)
    result_str += ('\n   the penalty for whole distillation loss is [ %.2f ]' % cfg.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT)
    with open(os.path.join(cfg.OUTPUT_DIR, 'control_info.txt'), 'w') as outfile:
        outfile.write(result_str)
    result_str += '\n\n'
    result_str += '---'*20
    print(result_str)
    time.sleep(2)
    func = registry.ROI_RELATION_PREDICTOR[cfg.GLOBAL_SETTING.RELATION_PREDICTOR]
    return func(cfg, in_channels)
