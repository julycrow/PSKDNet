# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

import functools
import numpy as np
import torch
import torch.nn as nn
import math
import copy
import warnings
import random
from mmcv.cnn import xavier_init, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence, TransformerLayerSequence, \
    BaseTransformerLayer, build_transformer_layer
from mmcv.runner.base_module import BaseModule, ModuleList
from typing import Optional
from collections import namedtuple

import torch.nn.functional as F
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import inverse_sigmoid, Transformer
from torch.nn.init import normal_
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)

from .CustomMSDeformableAttention import CustomMSDeformableAttention

from detectron2.layers import batched_nms, batched_nms_rotated
from detectron2.structures import Boxes, BoxMode, ImageList, Instances, RotatedBoxes

from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from plugin.models.utils.position_encoding import get_sine_pos_embed
from plugin.models.utils.misc import Conv2dNormActivation
from mmcv.cnn import Conv2d

AnnotObject = namedtuple("AnnotObject", ["gt_boxes", "gt_classes"])
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate t index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


@TRANSFORMER_LAYER.register_module()
class ParticleTransformerDecoderLayer(BaseTransformerLayer):
    """Particle Transformer Layer using similar structure to MapTransformerLayer."""

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):

        super().__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            batch_first=batch_first,
            **kwargs
        )

    def forward(self,
                query,
                key=None,
                value=None,
                memory_query=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                time_embedding=None,
                **kwargs):
        """Forward function for ParticleTransformerLayer."""

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        # if attn_masks is None:
        #     attn_masks = [None for _ in range(self.num_attn)]
        # elif isinstance(attn_masks, torch.Tensor):
        #     attn_masks = [
        #         copy.deepcopy(attn_masks) for _ in range(self.num_attn)
        #     ]
        #     warnings.warn(f'Use same attn_mask in all attentions in '
        #                   f'{self.__class__.__name__} ')
        # else:
        #     assert len(attn_masks) == self.num_attn, f'The length of ' \
        #                                              f'attn_masks {len(attn_masks)} must be equal ' \
        #                                              f'to the number of attention in ' \
        #                                              f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                if memory_query is None:
                    temp_key = temp_value = query
                else:
                    temp_key = temp_value = torch.cat([memory_query, query], dim=0)

                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_mask,
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    # attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                # 如果有时间嵌入，可以在这里处理条件
                # if time_embedding is not None:
                #     # 这里可以添加时间条件处理逻辑
                #     pass

                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class ParticleTransformerDecoder(BaseModule):
    """Implements the decoder for ParticleTransformer, keeping structure similar to MapTransformerDecoder_new."""

    def __init__(self,
                 transformerlayers=None,
                 num_layers=None,
                 prop_add_stage=0,
                 return_intermediate=True,
                 init_cfg=None):

        super().__init__(init_cfg)
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(build_transformer_layer(transformerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm
        self.return_intermediate = return_intermediate
        self.prop_add_stage = prop_add_stage
        self.look_forward_twice = True
        self.position_relation_embedding = PositionRelationEmbedding(16, 8)

        assert prop_add_stage >= 0 and prop_add_stage < num_layers

    def transform_box(self, pts, num_vec=5, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], num_vec, 20, 2).contiguous()
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        xmin = pts_x.min(dim=2, keepdim=True)[0]
        xmax = pts_x.max(dim=2, keepdim=True)[0]
        ymin = pts_y.min(dim=2, keepdim=True)[0]
        ymax = pts_y.max(dim=2, keepdim=True)[0]
        bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
        bbox = bbox_xyxy_to_cxcywh(bbox)

        return bbox, pts_reshape

    def forward(self,
                query,
                prop_query=None,
                key=None,
                value=None,
                query_pos=None,
                key_padding_mask=None,
                query_key_padding_mask=None,
                reference_points=None,
                prop_reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                reg_branches=None,
                cls_branches=None,
                is_first_frame_list=None,
                predict_refine=False,
                time_embedding=None,
                num_bevquery=None,
                attn_mask=None,
                **kwargs):
        """Forward function for ParticleTransformerDecoder."""

        if is_first_frame_list is None:
            # 如果未提供is_first_frame_list，则默认为全部是第一帧
            is_first_frame_list = [True] * query.shape[1]
        num_queries, bs, embed_dims = query.shape
        output = query
        intermediate = []
        intermediate_reference_points = []
        pos_relation = attn_mask

        for lid, layer in enumerate(self.layers):
            # 支持传播查询的合并阶段 - 与MapTransformerDecoder_new保持一致
            if lid == self.prop_add_stage and prop_query is not None and prop_reference_points is not None:  # prop_query:torch.Size([4, 33, 512]), prop_reference_points:torch.Size([4, 33, 20, 2])
                bs, topk, embed_dims = prop_query.shape  # torch.Size([4, 33, 512]) 修改之后应该是torch.Size([4, 66, 512])
                output = output.permute(1, 0, 2)
                # 划分BEV和diffusion查询的边界
                half_queries = num_bevquery  # 一半是BEV查询，一半是diffusion查询
                half_topk = topk // 2  # prop_query中一半是BEV，一半是diffusion

                with torch.no_grad():
                    tmp_scores, _ = cls_branches[lid](output).max(-1)  # (bs, num_q)

                new_query = []
                new_refpts = []
                for i in range(bs):
                    if is_first_frame_list[i]:
                        new_query.append(output[i])
                        new_refpts.append(reference_points[i])
                    else:
                        # 分别对BEV和diffusion部分取topk
                        bev_scores = tmp_scores[i][:half_queries]
                        diff_scores = tmp_scores[i][half_queries:]

                        # 分别为BEV和diffusion取topk
                        _, bev_valid_idx = torch.topk(bev_scores, k=half_queries - half_topk, dim=-1)
                        _, diff_valid_idx = torch.topk(diff_scores, k=half_queries - half_topk, dim=-1)

                        # 调整diffusion索引以匹配原始张量中的位置
                        diff_valid_idx = diff_valid_idx + half_queries

                        # 从prop_query中获取BEV和diffusion部分
                        prop_query_bev = prop_query[i][:half_topk]
                        prop_query_diff = prop_query[i][half_topk:]

                        # 从prop_reference_points中获取BEV和diffusion部分
                        prop_ref_bev = prop_reference_points[i][:half_topk]
                        prop_ref_diff = prop_reference_points[i][half_topk:]

                        # 按照BEV-diffusion的顺序拼接结果
                        combined_query = torch.cat([
                            prop_query_bev,
                            output[i][bev_valid_idx],
                            prop_query_diff,
                            output[i][diff_valid_idx]
                        ], dim=0)

                        combined_refpts = torch.cat([
                            prop_ref_bev,
                            reference_points[i][bev_valid_idx],
                            prop_ref_diff,
                            reference_points[i][diff_valid_idx]
                        ], dim=0)

                        new_query.append(combined_query)
                        new_refpts.append(combined_refpts)

                output = torch.stack(new_query).permute(1, 0, 2)
                reference_points = torch.stack(new_refpts)
                assert list(output.shape) == [num_queries, bs, embed_dims]

            # 准备参考点
            tmp = reference_points.clone()
            # if tmp.dim() == 4:  # 如果是多个点而不是单个参考点
            tmp[..., 1:2] = 1.0 - reference_points[..., 1:2]  # 反转y轴

            # 通过transformer层
            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                reference_points=tmp,  # torch.Size([4, 200, 20, 2])
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                # query_key_padding_mask=query_key_padding_mask,
                query_key_padding_mask=None,
                time_embedding=time_embedding,
                attn_mask=pos_relation,
                **kwargs)

            # 应用回归分支
            if reg_branches is not None:
                reg_points = reg_branches[lid](output.permute(1, 0, 2))  # (bs, num_q, 2*num_points)
                bs, num_queries, num_points2 = reg_points.shape
                reg_points = reg_points.view(bs, num_queries, num_points2 // 2, 2)  # range (0, 1)

                if self.training:
                    if lid >= 1:
                        src_points = reg_points
                    else:
                        src_points = tmp.view(bs, num_queries, num_points2 // 2, 2).contiguous()
                    tgt_points = reg_points
                    pos_relation = self.position_relation_embedding(src_points, tgt_points).flatten(0, 1).contiguous()
                    if attn_mask is not None:
                        pos_relation.masked_fill_(attn_mask, float("-inf"))

                if predict_refine:  # 与particle_detr相同
                    new_reference_points = reg_points + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.clone().detach()
                else:  # 与原版相同
                    new_reference_points = reg_points.sigmoid()  # (bs, num_q, num_points, 2)
                    reference_points = new_reference_points.clone().detach()

            if self.return_intermediate:
                intermediate.append(output.permute(1, 0, 2))  # [(bs, num_q, embed_dims)]
                intermediate_reference_points.append(new_reference_points)  # (bs, num_q, num_points, 2)

                # if predict_refine:
                #     intermediate_reference_points.append(new_reference_points)  # (bs, num_q, num_points, 2)
                # else:
                #     intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return intermediate, intermediate_reference_points

        return output, reference_points


@TRANSFORMER.register_module()
class ParticleTransformer(Transformer):
    """ParticleTransformer with diffusion model capabilities.

    Args:
        num_feature_levels (int): Number of feature maps from FPN.
        num_points (int): Number of sampling points for each instance.
        coord_dim (int): Dimension of coordinates (2D or 3D).
        diffusion_cfg (dict): Configuration for diffusion model.
    """

    def __init__(self,
                 num_feature_levels=1,
                 num_cams=6,
                 # two_stage_num_proposals=300,
                 decoder=None,
                 embed_dims=512,
                 num_points=20,
                 coord_dim=2,
                 # rotate_prev_bev=True,
                 # use_shift=True,
                 # use_can_bus=True,
                 # can_bus_norm=True,
                 # use_cams_embeds=True,
                 # rotate_center=[100, 100],
                 diffusion_cfg=None,
                 scale=3.0,
                 **kwargs):
        super().__init__(decoder=decoder, **kwargs)
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.num_points = num_points
        self.coord_dim = coord_dim
        # self.embed_dims = embed_dims
        self.embed_dims = self.encoder.embed_dims
        # self.rotate_prev_bev = rotate_prev_bev
        # self.use_shift = use_shift
        # self.use_can_bus = use_can_bus
        # self.can_bus_norm = can_bus_norm
        # self.use_cams_embeds = use_cams_embeds
        # self.rotate_center = rotate_center
        self.fp16_enabled = False

        # 配置扩散模型参数
        self.num_classes = diffusion_cfg.get('NUM_CLASSES', 1)
        self.num_proposals = diffusion_cfg.get('NUM_PROPOSALS', 900)
        self.hidden_dim = diffusion_cfg.get('HIDDEN_DIM', 256)
        self.num_heads = diffusion_cfg.get('NUM_HEADS', 8)
        self.radial_suppression_radius = diffusion_cfg.get('RADIAL_SUPPRESSION_RADIUS', 1.0)
        self.ddim_query_type = diffusion_cfg.get('DDIM_QUERY_TYPE', 'both')
        self.box_renewal_threshold = diffusion_cfg.get('BOX_RENEWAL_THRESHOLD', 0.5)
        self.NMS_THRESHOLD = float(diffusion_cfg.get('NMS_THRESHOLD', 0.5))

        if isinstance(self.radial_suppression_radius, list):
            self.radial_suppression_radius = self.radial_suppression_radius[0]

        # 设置扩散模型时间步
        timesteps = 1000
        sampling_timesteps = diffusion_cfg.get('SAMPLE_STEP', 50)
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 0.
        self.self_condition = False
        self.scale = scale
        self.box_renewal = True
        self.use_ensemble = True
        self.use_nms = True

        # 注册扩散模型buffer
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 计算扩散过程参数
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # 计算后验分布参数
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.init_layers()
        self.diffusion_loss = torch.tensor(0.0)  # 初始化扩散损失为0

    def init_layers(self):
        """初始化ParticleTransformer的层。"""
        # self.level_embeds = nn.Parameter(torch.Tensor(
        #     self.num_feature_levels, self.embed_dims))
        # self.cams_embeds = nn.Parameter(
        #     torch.Tensor(self.num_cams, self.embed_dims))
        # self.can_bus_mlp = nn.Sequential(
        #     nn.Linear(18, self.embed_dims // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.embed_dims // 2, self.embed_dims),
        #     nn.ReLU(inplace=True),
        # )
        # if self.can_bus_norm:
        #     self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

        # 扩散模型参考点到位置编码
        self.query_pos_net = nn.Linear(self.num_points * 2, self.embed_dims)

        # 时间嵌入
        self.time_dim = 4 * self.embed_dims
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.embed_dims),
            nn.Linear(self.embed_dims, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.block_time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embed_dims * 4, self.embed_dims * 2)
        )

        # BEVFormer位置编码到参考点
        self.positional_encoding_to_reference_point = nn.Linear(self.embed_dims, self.num_points * 2)

    def init_weights(self):
        """初始化变换器权重。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            # if hasattr(m, 'init_weights'):
            #     m.init_weights()
            # elif isinstance(m, CustomMSDeformableAttention):
            #     m.init_weights()
            if isinstance(m, CustomMSDeformableAttention):
                m.init_weights()

        # normal_(self.level_embeds)
        # normal_(self.cams_embeds)
        # xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
        xavier_init(self.query_pos_net, distribution='uniform', bias=0.)
        xavier_init(self.positional_encoding_to_reference_point, distribution='uniform', bias=0.)
        for m in self.time_mlp.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform', bias=0.)

        for m in self.block_time_mlp.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('bev_embed', 'object_query_embeds_bevformer', 'object_query_embeds_diffusion'))
    def forward(self,
                mlvl_feats=None,
                mlvl_masks=None,
                query_embed=None,
                prop_query=None,
                mlvl_pos_embeds=None,
                memory_query=None,
                init_reference_points=None,
                prop_reference_points=None,
                reg_branches=None,
                cls_branches=None,
                predict_refine=False,
                is_first_frame_list=None,
                query_key_padding_mask=None,
                # 新增参数，专门用于ParticleTransformer
                bev_embed=None,
                object_query_embeds_bevformer=None,
                object_query_embeds_diffusion=None,
                bev_h=None,
                bev_w=None,
                grid_length=[0.512, 0.512],
                prev_bev=None,
                gt_points=None,
                gt_labels=None,
                pc_range=None,
                num_bevquery=None,
                attn_mask=None,
                **kwargs):
        """ParticleTransformer的前向函数，保持与MapTransformer接口一致。

        同时支持MapTransformer和ParticleTransformer的调用方式。
        """
        feat_flatten = []
        mask_flatten = []
        # lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            # pos_embed = pos_embed.flatten(2).transpose(1, 2)
            # lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            # lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)  # torch.Size([4, 5000])
        # lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)

        # 如果提供了bev_embed，使用ParticleTransformer逻辑
        # 测试/推理逻辑
        if (gt_points is None) and (gt_labels is None):
            # 测试逻辑 - 采样随机参考点
            # outputs_class, outputs_coord = self.ddim_sample(
            bev_embed, inter_states, init_reference_out, inter_references_out, outputs_classes, outputs_coords = self.ddim_sample(
                backbone_feats=bev_embed,
                object_query_embeds_bevformer=object_query_embeds_bevformer,
                object_query_embeds_diffusion=object_query_embeds_diffusion,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                init_reference_points=init_reference_points,
                pc_range=pc_range,
                bev_h=bev_h,
                bev_w=bev_w,
                num_bevquery=num_bevquery,
                feat_flatten=feat_flatten,
                mask_flatten=mask_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                memory_query=memory_query,
                predict_refine=predict_refine,
                prop_query=prop_query,
                prop_reference_points=prop_reference_points,
                is_first_frame_list=is_first_frame_list,
                attn_mask=attn_mask,
                **kwargs
            )

            # return bev_embed, outputs_class, outputs_coord
            return bev_embed, inter_states, init_reference_out, inter_references_out, outputs_classes, outputs_coords

        # 训练逻辑 - 准备targets
        targets, x_points, noises, t = self.prep_targets(  # x_points是gt第一中展开补全到num_query并加了噪声的torch.Size([4, 100, 40])
            gt_points, gt_labels,
            bev_params=(bev_h, bev_w, grid_length, pc_range)
        )
        bs = x_points.size(0)
        num_query = object_query_embeds_bevformer.size(1)  # 200
        # 扩散查询 - 将坐标转换到[0, 1]范围内，(0,0)为左上角，(1,1)为右下角
        # 第一个数字为车辆右侧为正，第二个数字为车辆前方为正
        device = x_points.device

        # 需要将y轴翻转，使其符合图像坐标系
        x_points[..., 1::2] = 1 - x_points[..., 1::2]

        # 处理BEVFormer查询
        query_bevformer, query_pos_bevformer = torch.split(
            object_query_embeds_bevformer, num_bevquery,
            dim=1)  # object_query_embeds_bevformer:torch.Size([4, 200, 512]),self.embed_dims:512
        # query_pos_bevformer = query_pos_bevformer[None].expand(bs, -1, -1)
        # query_bevformer = query_bevformer[None].expand(bs, -1, -1)
        # reference_points_bevformer = self.positional_encoding_to_reference_point(
        #     query_pos_bevformer).view(bs, -1, self.num_points, 2).contiguous().sigmoid()  # torch.Size([4, 100, 20, 2])
        reference_points_bevformer, reference_points_bevformer_pos = torch.split(init_reference_points, num_bevquery,
                                                                                 dim=1)
        # 编码时间和位置
        time_embedding = self.time_mlp(t)

        # 在有噪声点位置获取BEV特征作为diffusion query
        if object_query_embeds_diffusion is None:
            # 使用grid_sample在BEV特征上采样
            query_diffusion = torch.nn.functional.grid_sample(
                bev_embed.view(bs, bev_h, bev_w, self.embed_dims).permute(0, 3, 1, 2),
                2 * x_points[:, None, :, :2].float() - 1,
                mode='bilinear',
                align_corners=False
            )
            query_diffusion = query_diffusion.squeeze(2).permute(0, 2, 1)
        else:
            # 使用预定义的查询嵌入
            num_query = object_query_embeds_diffusion.shape[1]
            query_diffusion = torch.nn.functional.grid_sample(  # torch.Size([4, 512, 1, 100])
                object_query_embeds_diffusion.permute(0, 2, 1).view(
                    bs, self.embed_dims, int(num_query ** 0.5), int(num_query ** 0.5)),
                2 * x_points[:, None, :, :2].float() - 1,
                mode='bilinear',
                align_corners=False
            )
            query_diffusion = query_diffusion.squeeze(2).permute(0, 2, 1)  # torch.Size([4, 100, 512])

        # 计算diffusion点的位置编码
        reference_points_diffusion = x_points.to(query_diffusion.dtype).view(bs, -1, self.num_points,
                                                                             2).contiguous()  # torch.Size([4, 100, 20, 2])
        x_points_type = x_points.to(query_diffusion.dtype)
        query_pos_diffusion = self.query_pos_net(x_points_type)

        # 格式化查询和位置
        query = torch.cat((query_bevformer, query_diffusion), dim=1)
        query_pos = torch.cat((query_pos_bevformer, query_pos_diffusion), dim=1)
        reference_points = torch.cat((reference_points_bevformer, reference_points_diffusion), dim=1)

        # 符合decoder期望的格式调整
        init_reference_out = reference_points
        query = query.permute(1, 0, 2)  # (num_q * 2, bs, embed_dims)
        query_pos = query_pos.permute(1, 0, 2)  # (num_q * 2, bs, embed_dims)
        value = bev_embed.permute(1, 0, 2)  # (H*W, bs, embed_dims)

        # 创建mask
        mask = torch.zeros((bs, bev_h * bev_w), device=query.device).bool()

        # 通过解码器
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=feat_flatten,
            # query_pos=query_pos,
            query_pos=None,
            key_padding_mask=mask_flatten,
            query_key_padding_mask=None,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,  # tensor([[ 50, 100]], device='cuda:0')
            level_start_index=level_start_index,  # tensor([0], device='cuda:0')
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            predict_refine=predict_refine,
            is_first_frame_list=is_first_frame_list,
            time_embedding=time_embedding,
            memory_query=memory_query,
            prop_query=prop_query,
            prop_reference_points=prop_reference_points,
            num_bevquery=num_bevquery,
            attn_mask=attn_mask,
            **kwargs
        )

        # 处理时间嵌入
        n_diffusion_queries = query_diffusion.size(1)  # 100
        n_bevformer_queries = query_bevformer.size(1)  # 100

        # 处理输出 - 应用时间条件
        if isinstance(inter_states, list):
            # 多个中间输出
            for i in range(len(inter_states)):
                inter_state = inter_states[i].permute(1, 0, 2)  # torch.Size([200, 4, 512])

                # 从时间嵌入中获取scale和shift
                scale_shift = self.block_time_mlp(time_embedding)  # torch.Size([4, 1, 1024])
                scale_shift = torch.repeat_interleave(scale_shift, n_diffusion_queries,
                                                      dim=1)  # torch.Size([4, 100, 1024])
                scale, shift = scale_shift.permute(1, 0, 2).chunk(2, dim=-1)  # torch.Size([100, 4, 512])
                # scale = scale.permute(1, 0, 2)  # torch.Size([100, 4, 412])

                # 只对diffusion部分应用时间变换
                inter_state2 = torch.empty_like(inter_state)
                inter_state2[:n_bevformer_queries, ...] = inter_state[:n_bevformer_queries, ...]
                inter_state2[n_bevformer_queries:, ...] = inter_state[n_bevformer_queries:, ...] * (
                            scale + 1) + shift  # todo 最小值会变化,调试particle是否scale + 1) + shift之后范围一样
                inter_states[i] = inter_state2.permute(1, 0, 2)  # 恢复原始格式

        # # 计算扩散损失 - 计算噪声预测损失
        # if self.training and targets is not None:
        #     # 获取预测的boxes - 使用最后一层的输出
        #     if isinstance(inter_references, list):
        #         pred_boxes = inter_references[-1][:, n_bevformer_queries:, :, :2].view(bs, n_bevformer_queries, self.num_points * 2).contiguous().clone()  # torch.Size([4, 100, 40])
        #         # 预测噪声
        #         pred_noise = self.predict_noise_from_start(x_points, t, pred_boxes)
        #         # 计算噪声预测损失
        #         self.diffusion_loss = F.mse_loss(pred_noise, noises.to(pred_noise.device))
        diffusion_outputs = {
            'x_points': x_points,
            'noises': noises,
            't': t,
            'targets': targets,
            'inter_references_out': inter_references,
            'n_diffusion_queries': n_diffusion_queries,
            'n_bevformer_queries': n_bevformer_queries
        }

        return bev_embed, inter_states, init_reference_out, inter_references, diffusion_outputs
        # return bev_embed, inter_states, init_reference_out, inter_references

    def prep_targets(self, gt_points, gt_labels, bev_params=None):
        """
        准备训练目标

        Args:
            gt_points: 列表，每个元素包含'lines'，形状为(n_element, 38, n_points(20)*xy(2))
            gt_labels: 列表，每个元素包含'labels'，形状为(n_element,)
            bev_params: BEV参数元组(bev_h, bev_w, grid_length, pc_range)

        Returns:
            targets: 目标列表
            diffused_points: 扩散后的点
            noises: 添加的噪声
            ts: 时间步
        """
        bev_h, bev_w, grid_length, pc_range = bev_params

        # 准备目标
        new_targets = []
        diffused_points = []
        noises = []
        ts = []

        device = gt_points[0].device
        coord_minimum = torch.Tensor([pc_range[0], pc_range[1]]).to(device)
        coord_range = torch.Tensor([pc_range[3] - pc_range[0], pc_range[4] - pc_range[1]]).to(device)

        # 处理每个target
        for i, (points_per_image, labels_per_image) in enumerate(zip(gt_points, gt_labels)):
            target = {}

            # 扁平化点，只保留关键点坐标
            # gt_centers = points_per_image.reshape(-1, 38, self.num_points, 2)
            gt_centers = points_per_image[:, 0, :]
            # gt_centers = gt_centers.squeeze(1)
            # gt_centers = gt_centers.reshape(-1, self.num_points, 2)
            # 归一化到[0, 1]
            # gt_centers_normalized = (gt_centers - coord_minimum) / coord_range

            # 添加扩散
            # d_points, d_noise, d_t = self.prepare_diffusion_concat(gt_centers_normalized)
            d_points, d_noise, d_t = self.prepare_diffusion_concat(gt_centers)
            diffused_points.append(d_points)
            noises.append(d_noise)
            ts.append(d_t)

            target["labels"] = labels_per_image.to(device)
            target["centers"] = gt_centers.to(device)
            target['centers_normalized'] = gt_centers.to(device)

            new_targets.append(target)

        return new_targets, torch.stack(diffused_points), torch.stack(noises), torch.stack(ts)

    def q_sample(self, x_start, t, noise=None):
        """
        向起始输入添加噪声
        """
        if noise is None:
            noise = torch.rand_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t, t, x0):
        """
        从起始点和当前点预测噪声
        """
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    @torch.no_grad()
    def ddim_sample(self, backbone_feats, object_query_embeds_bevformer, object_query_embeds_diffusion,
                    reg_branches=None, cls_branches=None, init_reference_points=None, clip_denoised=True,
                    num_bevquery=None, predict_refine=False, feat_flatten=None, mask_flatten=None, spatial_shapes=None,
                    level_start_index=None, memory_query=None, prop_query=None,
                    prop_reference_points=None, is_first_frame_list=None, attn_mask=None, **kwargs):
        """
        DDIM采样过程
        """
        batch = backbone_feats.size(0)
        shape = (batch, self.num_proposals, self.num_points * 2)  # (1, 100, 40)
        total_timesteps, sampling_timesteps = self.num_timesteps, self.sampling_timesteps  # 1000, 8

        # 加速采样
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)  # torch.Size([9])
        times = list(reversed(times.int().tolist()))  # list:9,[999, 874, 749, 624, 499, 374, 249, 124, -1]
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)] list:8

        device = backbone_feats.device
        img = torch.rand(shape, device=device)  # img是随机噪声点torch.Size([1, 100, 40]),用来迭代去噪得到最终预测点,从N(0, 1)开始

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        outputs_coords_all, outputs_class_all = [], []

        x_start = None
        eta = self.ddim_sampling_eta
        outputs_classes_all, outputs_coords_all = [], []

        # 迭代采样
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None  # None

            # 获取来自解码器的预测
            output = self.model_predictions(
                backbone_feats, img,
                object_query_embeds_bevformer,
                object_query_embeds_diffusion,
                time_cond,
                reg_branches,
                cls_branches,
                init_reference_points,
                self_cond,
                clip_x_start=clip_denoised,
                num_bevquery=num_bevquery,
                feat_flatten=feat_flatten,
                mask_flatten=mask_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                memory_query=memory_query,
                prop_query=prop_query,
                predict_refine=predict_refine,
                prop_reference_points=prop_reference_points,
                is_first_frame_list=is_first_frame_list,
                attn_mask=attn_mask,
                **kwargs
            )
            # preds, outputs_class, outputs_coord = output
            bev_embed, inter_states, init_reference_out, inter_references_out, outputs_class, outputs_coord, preds = output
            # outputs_classes_all.append(outputs_class)
            # outputs_coords_all.append(outputs_coord)
            # 提取预测结果，从model_predictions函数获取ModelPrediction
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            # 添加box_renewal逻辑,对diff特征进行框更新
            if self.box_renewal:
                score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][
                    0]  # torch.Size([200, 3]), torch.Size([200, 20, 2])

                if self.ddim_query_type == 'both':
                    n_bevformer_queries = num_bevquery
                else:
                    n_bevformer_queries = 0

                score_per_image = score_per_image[n_bevformer_queries:]
                box_per_image = box_per_image[n_bevformer_queries:]

                # 评分阈值筛选
                threshold = self.box_renewal_threshold
                score_per_image = torch.sigmoid(score_per_image)  # torch.Size([100, 3])
                value, _ = torch.max(score_per_image, -1, keepdim=False)  # torch.Size([100])

                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                # 只保留置信度大于threshold
                pred_noise = pred_noise[:, keep_idx, :] if pred_noise is not None else None
                x_start = x_start[:, keep_idx, :] if x_start is not None else None
                img = img[:, keep_idx, :]

            # # 在最后一步直接使用预测的x_start
            # if time_next < 0:
            #     img = x_start
            #     if self.use_ensemble and self.sampling_timesteps > 1:
            #         outputs_coords_all.append(outputs_coord)
            #         outputs_class_all.append(outputs_class)
            #         # bev_boxes = self.prediction_to_box(outputs_coord)[-1] if hasattr(self,
            #         #                                                                  'prediction_to_box') else None
            #         _, b, c, d, e = outputs_coord.size()
            #         bev_boxes = outputs_coord[-1].view(b, c, d * e).contiguous()  # torch.Size([1, 200, 40])
            #         sigmoid_cls = torch.sigmoid(outputs_class[-1])
            #         labels = torch.arange(self.num_classes, device=bev_boxes.device). \
            #             unsqueeze(0).repeat(sigmoid_cls.size(1), 1).flatten(0, 1)
            #         for i, (scores_per_image, box_pred_per_image) in enumerate(zip(sigmoid_cls, bev_boxes)):
            #             # scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals,
            #             #                                                                      sorted=False)
            #             # labels_per_image = labels[topk_indices]
            #             # box_pred_per_image = box_pred_per_image.view(-1, 1, d * e).repeat(1, sigmoid_cls.size(1), 1).view(
            #             #     -1, d * e)
            #             scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(sigmoid_cls.size(1),
            #                                                                                  sorted=False)  # torch.Size([900])
            #             labels_per_image = labels[topk_indices]  # torch.Size([200])
            #             box_pred_per_image = box_pred_per_image.view(-1, 1, d * e).repeat(1, self.num_classes, 1).view(
            #                 -1, d * e)  # torch.Size([600, 40])
            #
            #             box_pred_per_image = box_pred_per_image[topk_indices]
            #             ensemble_coord.append(box_pred_per_image)
            #             ensemble_label.append(labels_per_image)
            #             ensemble_score.append(scores_per_image)
            #     continue

            # 计算下一步
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            # 根据需要添加随机噪声补齐proposal数量,意义是:置信度大于0.5的保留,其他的全部随机化
            if self.box_renewal and num_remain < self.num_proposals:
                img = torch.cat((
                    img, torch.randn(batch, self.num_proposals - num_remain, self.num_points * 2, device=img.device)
                ), dim=1)

        #     # Store DDIM outputs
        #     if self.use_ensemble and self.sampling_timesteps > 1:
        #         outputs_coords_all.append(outputs_coord)  # 每个时间步的outputs_coord都保存
        #         outputs_class_all.append(outputs_class)
        #         # bev_boxes = self.prediction_to_box(outputs_coord)[
        #         #     -1]  # outputs_coord:torch.Size([6, 1, 200, 20, 2]), bev_boxes:torch.Size([1, 900, 5])
        #         _, b, c, d, e = outputs_coord.size()
        #         bev_boxes = outputs_coord[-1].view(b, c, d * e).contiguous()  # torch.Size([1, 200, 40])
        #         sigmoid_cls = torch.sigmoid(outputs_class[-1])  # torch.Size([1, 900, 10])
        #         labels = torch.arange(self.num_classes, device=bev_boxes.device). \
        #             unsqueeze(0).repeat(sigmoid_cls.size(1), 1).flatten(0, 1)  # torch.Size([9000])
        #         for i, (scores_per_image, box_pred_per_image) in enumerate(zip(sigmoid_cls, bev_boxes)):
        #             scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(sigmoid_cls.size(1),
        #                                                                                  sorted=False)  # torch.Size([900])
        #             labels_per_image = labels[topk_indices]  # torch.Size([200])
        #             box_pred_per_image = box_pred_per_image.view(-1, 1, d * e).repeat(1, self.num_classes, 1).view(-1, d * e)  # torch.Size([600, 40])
        #             ensemble_coord.append(box_pred_per_image)
        #             ensemble_label.append(labels_per_image)
        #             ensemble_score.append(scores_per_image)
        # # Ensemble NMS
        # if self.use_ensemble and self.sampling_timesteps > 1:
        #     box_pred_per_image = torch.cat(ensemble_coord, dim=0)  # torch.Size([4300, 40])
        #     scores_per_image = torch.cat(ensemble_score, dim=0)  # torch.Size([1500])
        #     labels_per_image = torch.cat(ensemble_label, dim=0)  # torch.Size([1500])
        #     outputs_coords_all = torch.cat(outputs_coords_all, dim=2)  # torch.Size([6, 1, 2700, 10])
        #     outputs_class_all = torch.cat(outputs_class_all, dim=2)  # torch.Size([6, 1, 2700, 10])
        #
        #     if self.use_nms:
        #         keep = batched_nms_rotated(box_pred_per_image, scores_per_image, labels_per_image,
        #                                    0.5)  # torch.Size([959])
        #         outputs_coord = outputs_coords_all[:, :, keep, :]  # torch.Size([6, 1, 959, 10])
        #         outputs_class = outputs_class_all[:, :, keep, :]  # torch.Size([6, 1, 959, 10])
        #
        # # NMS - Non-maximum suppression
        # boxes = self.prediction_to_box(outputs_coord)[-1, 0]  # torch.Size([959, 5])
        # scores, idxs = outputs_class[-1, 0, :, :].max(dim=1)  # torch.Size([959])
        # keep = batched_nms_rotated(boxes=boxes, scores=scores, idxs=idxs,
        #                            iou_threshold=self.NMS_THRESHOLD)  # torch.Size([389])
        #
        # # Finally, do radial suppression
        # selected_logits, selected_classes = outputs_class[-1, 0, keep, :].max(1)  # (N, C)torch.Size([389])
        # outputs_coord, outputs_class = self.average_boxes_in_radius(outputs_coord[-1, 0, keep, :],
        #                                                             selected_logits,
        #                                                             outputs_class[-1, 0, keep, :],
        #                                                             selected_classes,
        #                                                             radius=self.radial_suppression_radius)  # torch.Size([6, 1, 959, 10])->torch.Size([388, 10]), torch.Size([6, 1, 959, 10])->torch.Size([388, 10])

        return bev_embed, inter_states, init_reference_out, inter_references_out, outputs_class, outputs_coord
        # if self.ddim_query_type == 'bevformer':
        #     # 直接使用BEVFormer的结果
        #     return outputs_class, outputs_coord
        #
        # # 从预测中提取结果
        # pred_noise, x_start = preds.pred_noise, preds.pred_x_start
        #
        # # 过滤掉低分数的结果
        # if self.box_renewal:
        #     # 实现框更新逻辑
        #     score_per_image, coord_per_image = outputs_class[-1][0], outputs_coord[-1][0]
        #
        #     if self.ddim_query_type == 'both':
        #         n_bevformer_queries = object_query_embeds_bevformer.size(0) // 2
        #     else:
        #         n_bevformer_queries = 0
        #
        #     score_per_image = score_per_image[n_bevformer_queries:]
        #     coord_per_image = coord_per_image[n_bevformer_queries:]
        #     threshold = self.box_renewal_threshold
        #     score_per_image = torch.sigmoid(score_per_image)
        #     value, _ = torch.max(score_per_image, -1, keepdim=False)
        #
        #     keep_idx = value > threshold
        #     num_remain = torch.sum(keep_idx)
        #     pred_noise = pred_noise[:, keep_idx, :]
        #     x_start = x_start[:, keep_idx, :]
        #     img = img[:, keep_idx, :]
        #
        # # 在最后一步直接使用预测的x_start
        # if time_next < 0:
        #     img = x_start
        #     continue
        #
        # # 根据需要添加随机噪声补齐proposal数量
        # if self.box_renewal:
        #     img = torch.cat((
        #         img,
        #         torch.randn(1, self.num_proposals - num_remain, self.num_points * 2, device=img.device)
        #     ), dim=1)
        #
        # # 计算下一步
        # alpha = self.alphas_cumprod[time]
        # alpha_next = self.alphas_cumprod[time_next]
        # sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        # c = (1 - alpha_next - sigma ** 2).sqrt()
        # noise = torch.randn_like(img)
        #
        # img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        # 后处理
        # return outputs_class, outputs_coord

    def model_predictions(self, backbone_feats, x, object_query_embeds_bevformer, object_query_embeds_diffusion, t,
                          reg_branches=None, cls_branches=None, init_reference_points=None, x_self_cond=None,
                          clip_x_start=False, num_bevquery=None, predict_refine=False, feat_flatten=None,
                          mask_flatten=None, spatial_shapes=None,
                          level_start_index=None, memory_query=None, prop_query=None,
                          prop_reference_points=None, is_first_frame_list=None, attn_mask=None, **kwargs):
        """
        准备并调用解码器(扩散头)，以维持与MapTransformer接口的一致性
        """
        pc_range = kwargs['pc_range']
        bev_h = kwargs['bev_h']
        bev_w = kwargs['bev_w']

        # 归一化点坐标 todo x_points既然作为reference_points,是否同样用sigmoid来归一化,还有统一y轴翻转,x是gt中的点,已经是归一化到 [0, 1]了
        # x_points = torch.clamp(x, min=-1 * self.scale, max=self.scale)  # torch.Size([1, 100, 40])
        # x_points = ((x_points / self.scale) + 1) / 2  # 映射到[0, 1]
        x_points = x

        # y轴翻转以匹配图像坐标系
        x_points[..., 1::2] = 1 - x_points[..., 1::2]  # 1::2表示每隔一个元素取一次，翻转y轴
        bs = x_points.size(0)

        # 处理BEVFormer查询
        if self.ddim_query_type in ['both', 'bevformer']:
            # query_pos_bevformer, query_bevformer = torch.split(
            #     object_query_embeds_bevformer, self.embed_dims, dim=2)
            # query_pos_bevformer = query_pos_bevformer[None].expand(bs, -1, -1)
            # query_bevformer = query_bevformer[None].expand(bs, -1, -1)
            # reference_points_bevformer = self.positional_encoding_to_reference_point(
            #     query_pos_bevformer).sigmoid()
            num_query = object_query_embeds_bevformer.shape[1]  # 200
            query_bevformer, query_pos_bevformer = torch.split(
                object_query_embeds_bevformer, num_bevquery,
                dim=1)  # object_query_embeds_bevformer:torch.Size([1, 200, 512]),self.embed_dims:512
            # query_pos_bevformer = query_pos_bevformer[None].expand(bs, -1, -1)
            # query_bevformer = query_bevformer[None].expand(bs, -1, -1)
            # reference_points_bevformer = self.positional_encoding_to_reference_point(
            #     query_pos_bevformer).view(bs, -1, self.num_points,
            #                               2).contiguous().sigmoid()  # torch.Size([1, 100, 20, 2])
            reference_points_bevformer, reference_points_bevformer_pos = torch.split(init_reference_points,
                                                                                     num_bevquery, dim=1)

        # 处理扩散查询
        if object_query_embeds_diffusion is None:
            query_diffusion = torch.nn.functional.grid_sample(
                backbone_feats.view(bs, bev_h, bev_w, self.embed_dims).permute(0, 3, 1, 2),
                2 * x_points[:, None, :, :2].float() - 1,
                mode='bilinear',
                align_corners=False
            )
            query_diffusion = query_diffusion.squeeze(2).permute(0, 2, 1)
        else:
            num_query = object_query_embeds_diffusion.shape[1]
            query_diffusion = torch.nn.functional.grid_sample(  # torch.Size([4, 512, 1, 100])
                object_query_embeds_diffusion.permute(0, 2, 1).view(
                    bs, self.embed_dims, int(num_query ** 0.5), int(num_query ** 0.5)),
                2 * x_points[:, None, :, :2].float() - 1,
                mode='bilinear',
                align_corners=False
            )
            query_diffusion = query_diffusion.squeeze(2).permute(0, 2, 1)

        # 编码时间和位置
        if len(t.size()) == 1:
            t = t.view(1, 1)
        time_embedding = self.time_mlp(t)  # torch.Size([1, 1, 2048])
        reference_points_diffusion = x_points.to(query_diffusion.dtype).view(bs, -1, self.num_points,
                                                                             2).contiguous()  # torch.Size([1, 100, 20, 2])
        x_points_type = x_points.to(query_diffusion.dtype)
        query_pos_diffusion = self.query_pos_net(x_points_type)

        # 格式化查询和位置
        if self.ddim_query_type == 'both':
            query = torch.cat((query_bevformer, query_diffusion), dim=1)
            query_pos = torch.cat((query_pos_bevformer, query_pos_diffusion), dim=1)
            reference_points = torch.cat((reference_points_bevformer, reference_points_diffusion), dim=1)
        elif self.ddim_query_type == 'diffusion':
            query = query_diffusion
            query_pos = query_pos_diffusion
            reference_points = reference_points_diffusion
        else:
            query = query_bevformer
            query_pos = query_pos_bevformer
            reference_points = reference_points_bevformer

        # 准备符合decoder期望的格式
        init_reference_out = reference_points
        query = query.permute(1, 0, 2)  # (num_q, bs, embed_dims)
        query_pos = query_pos.permute(1, 0, 2)  # (num_q, bs, embed_dims)
        value = backbone_feats.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        mask = torch.zeros((bs, bev_h * bev_w), device=query.device).bool()

        # 通过解码器
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=feat_flatten,
            # query_pos=query_pos,
            query_pos=None,
            key_padding_mask=mask_flatten,
            query_key_padding_mask=None,
            reference_points=reference_points,
            # spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            # level_start_index=torch.tensor([0], device=query.device),
            spatial_shapes=spatial_shapes,  # tensor([[ 50, 100]], device='cuda:0')
            level_start_index=level_start_index,  # tensor([0], device='cuda:0')
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            predict_refine=predict_refine,
            # is_first_frame_list=[True] * bs,
            is_first_frame_list=is_first_frame_list,
            time_embedding=time_embedding,
            memory_query=memory_query,
            prop_query=prop_query,
            prop_reference_points=prop_reference_points,
            num_bevquery=num_bevquery,
            attn_mask=attn_mask,
            **kwargs
        )

        # 处理输出
        if self.ddim_query_type == 'both':
            n_diffusion_queries = query_diffusion.size(1)
            n_bevformer_queries = query_bevformer.size(1)

            # 调整输出格式
            if isinstance(inter_states, list):
                for i in range(len(inter_states)):
                    inter_state = inter_states[i].permute(1, 0, 2)  # (bs, num_q, embed_dims)

                    # 从时间嵌入中获取scale和shift
                    scale_shift = self.block_time_mlp(time_embedding)
                    scale_shift = torch.repeat_interleave(scale_shift, n_diffusion_queries, dim=1)
                    scale, shift = scale_shift.permute(1, 0, 2).chunk(2, dim=-1)

                    # 只对diffusion部分应用时间变换
                    inter_state2 = torch.empty_like(inter_state)
                    inter_state2[:n_bevformer_queries, ...] = inter_state[:n_bevformer_queries, ...]
                    inter_state2[n_bevformer_queries:, ...] = inter_state[n_bevformer_queries:, ...] * (
                            scale + 1) + shift
                    inter_states[i] = inter_state2.permute(1, 0, 2)  # 恢复原始格式

        # 处理输出格式
        bev_embed, hs, init_reference, inter_references = backbone_feats, inter_states, reference_points, inter_references
        # 处理输出 - 应用时间条件# 这里增加outputs_class和outputs_coord的计算
        # hs = inter_states.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl, (queries) in enumerate(hs):
            reference = inter_references[lvl]
            outputs_class = cls_branches[lvl](hs[lvl])
            # tmp = reg_branches[lvl](hs[lvl])
            # 参考diffusion_transformer的坐标后处理
            # tmp[..., 0:2] = (tmp[..., 0:2] + inverse_sigmoid(reference)).sigmoid()
            # 这里根据你的box格式做缩放
            outputs_classes.append(outputs_class)
            outputs_coords.append(reference)
        outputs_classes = torch.stack(outputs_classes)  # torch.Size([6, 1, 200, 3])
        outputs_coords = torch.stack(outputs_coords)  # torch.Size([6, 1, 200, 20, 2])

        #         outputs_class = outputs_class.unsqueeze(0)  # 添加层维度
        #         tmp = tmp.unsqueeze(0)
        #         outputs_classes.append(outputs_class)
        #         outputs_coords.append(tmp)
        #
        #     outputs_classes = torch.cat(outputs_classes, dim=0)
        #     outputs_coords = torch.cat(outputs_coords, dim=0)

        # 如果使用的是bevformer查询类型，直接返回
        if self.ddim_query_type == 'bevformer':
            return None, inter_states, init_reference_out, inter_references

        n_diffusion_queries = query_diffusion.size(1)
        x_start = outputs_coords[-1, :, -n_diffusion_queries:, :, :]  # torch.Size([1, 100, 20, 2])预测中心点: 绝对坐标 (cx, cy)
        # 将绝对坐标转换为[0,1]范围
        device = x_start.device
        # coord_minimum = torch.Tensor([pc_range[0], pc_range[1]]).to(device)
        # coord_range = torch.Tensor([pc_range[3] - pc_range[0], pc_range[4] - pc_range[1]]).to(device)
        # x_start = (x_start - coord_minimum) / coord_range  # (cx, cy) 在 [0, 1] 范围内

        # 转换为扩散模型使用的范围 (-scale, scale)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)

        # 计算预测的噪声
        t = t.squeeze(-1)
        x_start = x_start.view(-1, n_diffusion_queries,
                               self.num_points * 2).contiguous()  # x_start: torch.Size([1, 100, 40])
        pred_noise = self.predict_noise_from_start(x, t,
                                                   x_start)  # x:torch.Size([1, 100, 40]), pred_noise:torch.Size([1, 100, 40])
        return bev_embed, inter_states, init_reference_out, inter_references, outputs_classes, outputs_coords, ModelPrediction(
            pred_noise, x_start)

    def prepare_diffusion_concat(self, gt_boxes):
        """
        准备扩散连接

        Args:
            gt_boxes: Tensor (N, num_points*2). 每个元素是 (x1, y1, x2, y2, ...), 在 [0, 1] 范围内
                已经归一化到[0, 1]
        Returns:
            diff_boxes: 加入噪声后的点
        """
        device = gt_boxes.device
        t = torch.randint(0, self.num_timesteps, (1,), device=device).long()

        # 为每个关键点创建噪声
        noise = torch.rand(self.num_proposals, self.num_points * 2, device=device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # 如果没有gt点，生成假的gt点
            gt_boxes = torch.as_tensor(
                [[0.5, 0.5] * self.num_points],
                dtype=torch.float,
                device=device
            )
            num_gt = 1

        if num_gt < self.num_proposals:
            # 生成随机点作为填充
            box_placeholder = torch.rand(
                self.num_proposals - num_gt, self.num_points * 2,
                device=device
            )  # / 6. + 0.5  # N(mu=1/2, sigma=1/6)torch.Size([86, 40])
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            # 随机选择部分gt点
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        # 添加扩散
        x_start = (x_start * 2. - 1.) * self.scale
        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # torch.Size([100, 40])
        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.  # 映射回[0, 1]范围
        diff_boxes = x

        return diff_boxes, noise, t

    def average_boxes_in_radius(self, boxes, logits, logits_full, classes, radius=1.0):
        """
        类似于NMS但只考虑中心距离（而不是IOU），根据置信度对框进行平均。

        boxes: torch.Tensor (B, 10)
        logits: torch.Tensor (B,)
        logits_full: torch.Tensor (B, K)
        classes = torch.Tensor (B,)
        """
        new_boxes = []
        new_logits_full = []
        for class_id in classes.unique():
            # 获取对应这个类的框
            ids_for_that_class = torch.where(classes == class_id)[0]
            boxes_for_that_class = boxes[ids_for_that_class]
            logits_for_that_class = logits[ids_for_that_class].sigmoid()
            logits_full_for_that_class = logits_full[ids_for_that_class]

            # 按置信度降序排序
            order = torch.argsort(logits_for_that_class, descending=True)
            keep = torch.ones_like(order, dtype=torch.bool)

            # 计算成对距离
            pairwise_dist = torch.cdist(boxes_for_that_class[..., :2], boxes_for_that_class[..., :2], p=1)

            for i in order:
                if keep[i] == True:
                    indices = torch.nonzero(pairwise_dist[i] < radius)
                    if len(indices.size()) > 1:
                        indices = indices.squeeze()
                    if len(indices.size()) < 1:
                        continue
                    average_box = (logits_for_that_class[indices][:, None] * boxes_for_that_class[indices]).sum(0) / \
                                  logits_for_that_class[indices].sum()
                    average_conf = (logits_for_that_class[indices][:, None] * logits_full_for_that_class[indices]).sum(
                        0) / logits_for_that_class[indices].sum()

                    keep[indices] = False
                    keep[i] = True
                    boxes_for_that_class[i] = average_box
                    logits_full_for_that_class[i] = average_conf

            new_boxes.append(boxes_for_that_class[keep])
            new_logits_full.append(logits_full_for_that_class[keep])
        return torch.cat(new_boxes, dim=0), torch.cat(new_logits_full, dim=0)


def box_rel_encoding(src_boxes, tgt_boxes, eps=1e-5):
    # construct position relation
    xy1, wh1 = src_boxes.split([2, 2], -1)
    xy2, wh2 = tgt_boxes.split([2, 2], -1)
    delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
    delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
    delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
    pos_embed = torch.cat([delta_xy, delta_wh], -1)

    return pos_embed


class PositionRelationEmbedding(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            num_heads=8,
            temperature=10000.0,
            scale=100.0,
            activation_layer=nn.ReLU,
            inplace=True,
    ):
        super().__init__()
        self.pos_proj = Conv2dNormActivation(
            embed_dim * 4,
            num_heads,
            kernel_size=1,
            inplace=inplace,
            norm_layer=None,
            activation_layer=activation_layer,
        )
        self.pos_func = functools.partial(
            get_sine_pos_embed,
            num_pos_feats=embed_dim,
            temperature=temperature,
            scale=scale,
            exchange_xy=False,
        )
        self.point_similarity_net = PointSetSimilarityNetwork(num_points=20)

    def forward(self, src_boxes, tgt_boxes=None):
        if tgt_boxes is None:
            tgt_boxes = src_boxes

        with torch.no_grad():
            if src_boxes.dim() == 4 and src_boxes.shape[-2] == 20:
                pos_embed = self.point_similarity_net(src_boxes, tgt_boxes)
            else:
                pos_embed = box_rel_encoding(src_boxes, tgt_boxes)

            pos_embed = self.pos_func(pos_embed).permute(0, 3, 1, 2).contiguous()

        pos_embed = self.pos_proj(pos_embed)

        return pos_embed.clone().contiguous()


class PointSetSimilarityNetwork(nn.Module):
    def __init__(self, num_points=20):
        super().__init__()
        self.num_points = num_points

        self.point_encoder = nn.Sequential(
            nn.Conv1d(2, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1),
        )

        self.similarity = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 1)
        )

    def forward(self, src_points, tgt_points=None):
        if tgt_points is None:
            tgt_points = src_points

        bs, num_q, num_points, _ = src_points.shape

        src_flat = src_points.reshape(bs * num_q, num_points, 2).permute(0, 2, 1)
        tgt_flat = tgt_points.reshape(bs * num_q, num_points, 2).permute(0, 2, 1)

        src_features = self.point_encoder(src_flat)
        tgt_features = self.point_encoder(tgt_flat)

        src_global = torch.max(src_features, dim=2)[0].reshape(bs, num_q, 64)
        tgt_global = torch.max(tgt_features, dim=2)[0].reshape(bs, num_q, 64)

        src_global = src_global.unsqueeze(2)
        tgt_global = tgt_global.unsqueeze(1)

        combined = torch.cat([
            src_global.expand(-1, -1, num_q, -1),
            tgt_global.expand(-1, num_q, -1, -1)
        ], dim=-1)

        combined = combined.permute(0, 3, 1, 2)
        similarity = self.similarity(combined)

        return similarity.permute(0, 2, 3, 1)
