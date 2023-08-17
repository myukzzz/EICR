# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import obj_edge_vectors, center_x, sort_by_score, to_onehot, get_dropout_mask, nms_overlaps, encode_box_info

#DenseCLIP
#from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
#from denseclip.untils import tokenize


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
        if N>1:
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

            x = x.permute(1, 2, 0)
            # global_feat = x[:, :, 0]
            feature_map = x[:, :, :].reshape(B, -1, N)
        else:
            v = self.v_proj(x)
            x = self.c_proj(v)
            feature_map = x.permute(0, 2, 1)
        # return global_feat, feature_map
        return feature_map
class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, cfg, statistics, eps=1e-3):
        super(FrequencyBias, self).__init__()
        pred_dist = statistics['pred_dist'].float()
        assert pred_dist.size(0) == pred_dist.size(1)

        self.num_objs = pred_dist.size(0)
        self.num_rels = pred_dist.size(2)
        pred_dist = pred_dist.view(-1, self.num_rels)

        self.obj_baseline = nn.Embedding(self.num_objs*self.num_objs, self.num_rels)
        with torch.no_grad():
            self.obj_baseline.weight.copy_(pred_dist, non_blocking=True)

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2] 
        :return: 
        """
        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def index_with_probability(self, pair_prob):
        """
        :param labels: [batch_size, num_obj, 2] 
        :return: 
        """
        batch_size, num_obj, _ = pair_prob.shape

        joint_prob = pair_prob[:,:,0].contiguous().view(batch_size, num_obj, 1) * pair_prob[:,:,1].contiguous().view(batch_size, 1, num_obj)

        return joint_prob.view(batch_size, num_obj*num_obj)  @ self.obj_baseline.weight

    def forward(self, labels):
        # implement through index_with_labels
        return self.index_with_labels(labels)


class DecoderRNN(nn.Module):
    def __init__(self, config, obj_classes, embed_dim, inputs_dim, hidden_dim, rnn_drop):
        super(DecoderRNN, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.embed_dim = embed_dim

        obj_embed_vecs = obj_edge_vectors(['start'] + self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=embed_dim)
        self.obj_embed = nn.Embedding(len(self.obj_classes)+1, embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.hidden_size = hidden_dim
        self.inputs_dim = inputs_dim
        self.input_size = self.inputs_dim + self.embed_dim
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES
        self.rnn_drop=rnn_drop

        self.input_linearity = torch.nn.Linear(self.input_size, 6 * self.hidden_size, bias=True)
        self.state_linearity = torch.nn.Linear(self.hidden_size, 5 * self.hidden_size, bias=True)
        self.out_obj = nn.Linear(self.hidden_size, len(self.obj_classes))
        
        self.init_parameters()

    def init_parameters(self):
        # Use sensible default initializations for parameters.
        with torch.no_grad():
            torch.nn.init.constant_(self.state_linearity.bias, 0.0)
            torch.nn.init.constant_(self.input_linearity.bias, 0.0)

    def lstm_equations(self, timestep_input, previous_state, previous_memory, dropout_mask=None):
        """
        Does the hairy LSTM math
        :param timestep_input:
        :param previous_state:
        :param previous_memory:
        :param dropout_mask:
        :return:
        """
        # Do the projections for all the gates all at once.
        projected_input = self.input_linearity(timestep_input)
        projected_state = self.state_linearity(previous_state)

        # Main LSTM equations using relevant chunks of the big linear
        # projections of the hidden state and inputs.
        input_gate = torch.sigmoid(projected_input[:, 0 * self.hidden_size:1 * self.hidden_size] +
                                   projected_state[:, 0 * self.hidden_size:1 * self.hidden_size])
        forget_gate = torch.sigmoid(projected_input[:, 1 * self.hidden_size:2 * self.hidden_size] +
                                    projected_state[:, 1 * self.hidden_size:2 * self.hidden_size])
        memory_init = torch.tanh(projected_input[:, 2 * self.hidden_size:3 * self.hidden_size] +
                                 projected_state[:, 2 * self.hidden_size:3 * self.hidden_size])
        output_gate = torch.sigmoid(projected_input[:, 3 * self.hidden_size:4 * self.hidden_size] +
                                    projected_state[:, 3 * self.hidden_size:4 * self.hidden_size])
        memory = input_gate * memory_init + forget_gate * previous_memory
        timestep_output = output_gate * torch.tanh(memory)

        highway_gate = torch.sigmoid(projected_input[:, 4 * self.hidden_size:5 * self.hidden_size] +
                                         projected_state[:, 4 * self.hidden_size:5 * self.hidden_size])
        highway_input_projection = projected_input[:, 5 * self.hidden_size:6 * self.hidden_size]
        timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection

        # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
        if dropout_mask is not None and self.training:
            timestep_output = timestep_output * dropout_mask
        return timestep_output, memory

    def forward(self, inputs, initial_state=None, labels=None, boxes_for_nms=None):
        if not isinstance(inputs, PackedSequence):
            raise ValueError('inputs must be PackedSequence but got %s' % (type(inputs)))

        assert isinstance(inputs, PackedSequence)
        sequence_tensor, batch_lengths, _, _ = inputs
        batch_size = batch_lengths[0]

        # We're just doing an LSTM decoder here so ignore states, etc
        if initial_state is None:
            previous_memory = sequence_tensor.new().resize_(batch_size, self.hidden_size).fill_(0)
            previous_state = sequence_tensor.new().resize_(batch_size, self.hidden_size).fill_(0)
        else:
            assert len(initial_state) == 2
            previous_memory = initial_state[1].squeeze(0)
            previous_state = initial_state[0].squeeze(0)

        previous_obj_embed = self.obj_embed.weight[0, None].expand(batch_size, self.embed_dim)

        if self.rnn_drop > 0.0:
            dropout_mask = get_dropout_mask(self.rnn_drop, previous_memory.size(), previous_memory.device)
        else:
            dropout_mask = None

        # Only accumulating label predictions here, discarding everything else
        out_dists = []
        out_commitments = []

        end_ind = 0
        for i, l_batch in enumerate(batch_lengths):
            start_ind = end_ind
            end_ind = end_ind + l_batch

            if previous_memory.size(0) != l_batch:
                previous_memory = previous_memory[:l_batch]
                previous_state = previous_state[:l_batch]
                previous_obj_embed = previous_obj_embed[:l_batch]
                if dropout_mask is not None:
                    dropout_mask = dropout_mask[:l_batch]

            timestep_input = torch.cat((sequence_tensor[start_ind:end_ind], previous_obj_embed), 1)

            previous_state, previous_memory = self.lstm_equations(timestep_input, previous_state,
                                                                  previous_memory, dropout_mask=dropout_mask)

            pred_dist = self.out_obj(previous_state)
            out_dists.append(pred_dist)

            if self.training:
                labels_to_embed = labels[start_ind:end_ind].clone()
                # Whenever labels are 0 set input to be our max prediction
                nonzero_pred = pred_dist[:, 1:].max(1)[1] + 1
                is_bg = (labels_to_embed == 0).nonzero()
                if is_bg.dim() > 0:
                    labels_to_embed[is_bg.squeeze(1)] = nonzero_pred[is_bg.squeeze(1)]
                out_commitments.append(labels_to_embed)
                previous_obj_embed = self.obj_embed(labels_to_embed+1)
            else:
                assert l_batch == 1
                out_dist_sample = F.softmax(pred_dist, dim=1)
                best_ind = out_dist_sample[:, 1:].max(1)[1] + 1
                out_commitments.append(best_ind)
                previous_obj_embed = self.obj_embed(best_ind+1)

        # Do NMS here as a post-processing step
        if boxes_for_nms is not None and not self.training:
            is_overlap = nms_overlaps(boxes_for_nms).view(
                boxes_for_nms.size(0), boxes_for_nms.size(0), boxes_for_nms.size(1)
            ).cpu().numpy() >= self.nms_thresh

            out_dists_sampled = F.softmax(torch.cat(out_dists,0), 1).cpu().numpy()
            out_dists_sampled[:,0] = 0

            out_commitments = out_commitments[0].new(len(out_commitments)).fill_(0)

            for i in range(out_commitments.size(0)):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_commitments[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            out_commitments = out_commitments
        else:
            out_commitments = torch.cat(out_commitments, 0)

        return torch.cat(out_dists, 0), out_commitments


class LSTMContext(nn.Module):
    """
    Modified from neural-motifs to encode contexts for each objects
    """
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super(LSTMContext, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nl_obj = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER
        self.nl_edge = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER
        assert self.nl_obj > 0 and self.nl_edge > 0

        # TODO Kaihua Tang
        # AlternatingHighwayLSTM is invalid for pytorch 1.0
        self.obj_ctx_rnn = torch.nn.LSTM(
                input_size=self.obj_dim+self.embed_dim + 128,
                hidden_size=self.hidden_dim,
                num_layers=self.nl_obj,
                dropout=self.dropout_rate if self.nl_obj > 1 else 0,
                bidirectional=True)
        self.decoder_rnn = DecoderRNN(self.cfg, self.obj_classes, embed_dim=self.embed_dim,
                inputs_dim=self.hidden_dim + self.obj_dim + self.embed_dim + 128,
                hidden_dim=self.hidden_dim,
                rnn_drop=self.dropout_rate)
        self.edge_ctx_rnn = torch.nn.LSTM(
                input_size=self.embed_dim + self.hidden_dim + self.obj_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.nl_edge,
                dropout=self.dropout_rate if self.nl_edge > 1 else 0,
                bidirectional=True)
        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.lin_obj_h = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.lin_edge_h = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        
        # untreated average features
        self.average_ratio = 0.0005
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS

        if self.effect_analysis:
            self.register_buffer("untreated_dcd_feat", torch.zeros(self.hidden_dim + self.obj_dim + self.embed_dim + 128))
            self.register_buffer("untreated_obj_feat", torch.zeros(self.obj_dim+self.embed_dim + 128))
            self.register_buffer("untreated_edg_feat", torch.zeros(self.embed_dim + self.obj_dim))
        use_CLP=False
        self.use_CLIP=use_CLP
        if self.use_CLIP:
            self.proj_CLIP = nn.Linear(4096 + 151, 4096)
            self.attnpool = AttentionPool2d(80, 4096, 32, 512)
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
            self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])

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

    def sort_rois(self, proposals):
        c_x = center_x(proposals)#框中心x坐标
        # leftright order
        scores = c_x / (c_x.max() + 1)
        return sort_by_score(proposals, scores)

    def obj_ctx(self, obj_feats, proposals, obj_labels=None, boxes_per_cls=None, ctx_average=False):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_labels: [num_obj] the GT labels of the image
        :param box_priors: [num_obj, 4] boxes. We'll use this for NMS
        :param boxes_per_cls
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        # Sort by the confidence of the maximum detection.
        perm, inv_perm, ls_transposed = self.sort_rois(proposals)
        # Pass object features, sorted by score, into the encoder LSTM
        obj_inp_rep = obj_feats[perm].contiguous()
        input_packed = PackedSequence(obj_inp_rep, ls_transposed)
        encoder_rep = self.obj_ctx_rnn(input_packed)[0][0]
        encoder_rep = self.lin_obj_h(encoder_rep) # map to hidden_dim

        # untreated decoder input
        batch_size = encoder_rep.shape[0]
        
        if (not self.training) and self.effect_analysis and ctx_average:##########没用
            decoder_inp = self.untreated_dcd_feat.view(1, -1).expand(batch_size, -1)
        else:###############
            decoder_inp = torch.cat((obj_inp_rep, encoder_rep), 1)

        if self.training and self.effect_analysis:#########没用
            self.untreated_dcd_feat = self.moving_average(self.untreated_dcd_feat, decoder_inp)
        
        # Decode in order
        if self.mode != 'predcls':
            decoder_inp = PackedSequence(decoder_inp, ls_transposed)
            obj_dists, obj_preds = self.decoder_rnn(
                decoder_inp, #obj_dists[perm],
                labels=obj_labels[perm] if obj_labels is not None else None,
                boxes_for_nms=boxes_per_cls[perm] if boxes_per_cls is not None else None,
                )
            obj_preds = obj_preds[inv_perm]
            obj_dists = obj_dists[inv_perm]
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        encoder_rep = encoder_rep[inv_perm]

        return obj_dists, obj_preds, encoder_rep, perm, inv_perm, ls_transposed

    def edge_ctx(self, inp_feats, perm, inv_perm, ls_transposed):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :return: edge_ctx: [num_obj, #feats] For later!
        """
        edge_input_packed = PackedSequence(inp_feats[perm], ls_transposed)
        edge_reps = self.edge_ctx_rnn(edge_input_packed)[0][0]
        edge_reps = self.lin_edge_h(edge_reps) # map to hidden_dim

        edge_ctx = edge_reps[inv_perm]
        return edge_ctx

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

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




    def forward(self, x, proposals, rel_pair_idxs, logger=None, all_average=False, ctx_average=False):
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        else:
            obj_labels = None

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels.long())
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight#label-embedding
        
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(proposals))

        batch_size = x.shape[0]#x-ROIfeat
        if self.use_CLIP:
            num_objs = [len(p) for p in proposals]
            new_obj_feats = []
            scoremaps = []
            CLIP_obj_feats = list(x.split(num_objs, dim=0))
            for obj_feat in CLIP_obj_feats:
                obj_feat = obj_feat.unsqueeze(0)
                # x_global, x_local = self.attnpool(obj_feat)
                x_local = self.attnpool(obj_feat)
                # obj_feat, score_map = self.extract_feat(obj_feat, x_global, x_local, use_seg_loss=True)
                obj_feat, score_map = self.extract_feat(obj_feat, x_local, use_seg_loss=True)
                new_obj_feats.append(obj_feat)
                scoremaps.append(score_map)
            x = cat(new_obj_feats, dim=0)
            scoremaps = cat(scoremaps, dim=0)

        if all_average and self.effect_analysis and (not self.training):
            obj_pre_rep = self.untreated_obj_feat.view(1, -1).expand(batch_size, -1)
        else:#all_average= False
            obj_pre_rep = cat((x, obj_embed, pos_embed), -1)#4096+200+128

        boxes_per_cls = None
        if self.mode == 'sgdet' and not self.training:
            boxes_per_cls = cat([proposal.get_field('boxes_per_cls') for proposal in proposals], dim=0) # comes from post process of box_head

        # object level contextual feature
        obj_dists, obj_preds, obj_ctx, perm, inv_perm, ls_transposed = self.obj_ctx(obj_pre_rep, proposals, obj_labels, boxes_per_cls, ctx_average=ctx_average)
        # edge level contextual feature
        obj_embed2 = self.obj_embed2(obj_preds.long())

        if (all_average or ctx_average) and self.effect_analysis and (not self.training):
            obj_rel_rep = cat((self.untreated_edg_feat.view(1, -1).expand(batch_size, -1), obj_ctx), dim=-1)
        else:####################
            obj_rel_rep = cat((obj_embed2, x, obj_ctx), -1)
            
        edge_ctx = self.edge_ctx(obj_rel_rep, perm=perm, inv_perm=inv_perm, ls_transposed=ls_transposed)#也是过LSTM

        # memorize average feature
        if self.training and self.effect_analysis:###############没用
            self.untreated_obj_feat = self.moving_average(self.untreated_obj_feat, obj_pre_rep)
            self.untreated_edg_feat = self.moving_average(self.untreated_edg_feat, cat((obj_embed2, x), -1))

        if self.use_CLIP:
            return obj_dists, obj_preds, edge_ctx, scoremaps
        #target,labels,edges
        return obj_dists, obj_preds, edge_ctx, None


