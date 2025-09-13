# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
import copy
import random
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer, xavier_init
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer, build_transformer_layer)
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.runner import force_fp32, auto_fp16
from torch.nn.init import normal_

from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.utils.transformer import Transformer
from .CustomMSDeformableAttention import CustomMSDeformableAttention
from collections import namedtuple

# 定义命名元组用于扩散模型
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
AnnotObject = namedtuple("AnnotObject", ["gt_boxes", "gt_classes"])

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """从张量a中提取时间步t对应的值，并重塑为与x形状兼容的张量"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    余弦调度
    论文：https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码，用于时间嵌入"""
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

# 定义Particle转换器解码器层
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class ParticleTransformerDecoder(BaseModule):
    """Particle transformer解码器实现
    
    类似于DETR解码器，但添加了对扩散模型的支持
    
    Args:
        transformerlayers (list[dict] | dict): 解码器层配置
        num_layers (int): 解码器层数
        return_intermediate (bool): 是否返回中间输出
        prop_add_stage (int): 添加propagation query的阶段
    """

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
        assert prop_add_stage >= 0 and prop_add_stage < num_layers

    def forward(self,
                query,
                key,
                value,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                reg_branches=None,
                cls_branches=None,
                memory_query=None,
                prop_query=None,
                prop_reference_points=None,
                is_first_frame_list=None,
                predict_refine=True,
                time_embedding=None,
                **kwargs):
        """解码器前向传播函数
        
        Args:
            query: 输入查询张量，形状为 [num_query, bs, embed_dims]
            key, value: 注意力机制的键值
            query_pos: 查询位置编码
            key_padding_mask: 键的填充掩码
            reference_points: 参考点，形状为 [bs, num_query, num_points, 2]
            spatial_shapes: 空间形状
            level_start_index: 层级起始索引
            reg_branches: 回归分支
            cls_branches: 分类分支
            memory_query: 记忆查询
            prop_query: 传播查询
            prop_reference_points: 传播参考点
            is_first_frame_list: 是否是第一帧的列表
            predict_refine: 是否细化预测
            time_embedding: 时间嵌入，用于扩散模型
            
        Returns:
            intermediate 或 output: 解码器输出
            intermediate_reference_points 或 reference_points: 参考点
        """
        num_queries, bs, embed_dims = query.shape
        output = query
        intermediate = []
        intermediate_reference_points = []
        
        # 遍历所有解码器层
        for lid, layer in enumerate(self.layers):
            # 在特定阶段添加传播查询
            if lid == self.prop_add_stage and prop_query is not None and prop_reference_points is not None:
                bs, topk, embed_dims = prop_query.shape
                output = output.permute(1, 0, 2)
                
                # 获取当前查询的得分
                with torch.no_grad():
                    if cls_branches and lid < len(cls_branches):
                        tmp_scores, _ = cls_branches[lid](output).max(-1) # (bs, num_q)
                    else:
                        tmp_scores = torch.ones(bs, output.size(1), device=output.device)
                
                # 为每个样本创建新的查询和参考点
                new_query = []
                new_refpts = []
                for i in range(bs):
                    if is_first_frame_list is None or is_first_frame_list[i]:
                        new_query.append(output[i])
                        new_refpts.append(reference_points[i])
                    else:
                        # 选择得分最高的查询
                        _, valid_idx = torch.topk(tmp_scores[i], k=num_queries-topk, dim=-1)
                        new_query.append(torch.cat([prop_query[i], output[i][valid_idx]], dim=0))
                        new_refpts.append(torch.cat([prop_reference_points[i], reference_points[i][valid_idx]], dim=0))
                
                output = torch.stack(new_query).permute(1, 0, 2)
                reference_points = torch.stack(new_refpts)
                assert list(output.shape) == [num_queries, bs, embed_dims]

            # 复制参考点并调整y轴
            tmp = reference_points.clone()
            if reference_points.dim() > 3:  # [bs, num_query, num_points, 2]
                tmp[..., 1:2] = 1.0 - reference_points[..., 1:2]  # 反转y轴
            elif reference_points.dim() == 3:  # [bs, num_query, 2]
                tmp[..., 1:2] = 1.0 - reference_points[..., 1:2]  # 反转y轴
            
            # 传递时间嵌入到解码器层（如果有的话）
            layer_kwargs = kwargs.copy()
            if time_embedding is not None and hasattr(layer, 'with_time_embedding'):
                layer_kwargs['time_embedding'] = time_embedding
            
            # 应用transformer层
            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                reference_points=tmp,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                memory_query=memory_query,
                **layer_kwargs)
            
            # 应用回归分支预测参考点
            if reg_branches is not None and lid < len(reg_branches):
                # 获取回归分支的预测
                reg_points = reg_branches[lid](output.permute(1, 0, 2))  # (bs, num_q, 2*num_points)
                bs, num_queries, num_points2 = reg_points.shape
                
                # 处理一点或多点情况
                if num_points2 > 2:
                    reg_points = reg_points.view(bs, num_queries, num_points2//2, 2)  # 范围 (0, 1)
                else:
                    reg_points = reg_points.unsqueeze(2)  # (bs, num_q, 1, 2)
                
                # 更新参考点
                if predict_refine:
                    new_reference_points = reg_points + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    new_reference_points = reg_points.sigmoid()  # (bs, num_q, num_points, 2)
                
                reference_points = new_reference_points.clone().detach()
            
            # 存储中间结果
            if self.return_intermediate:
                intermediate.append(output.permute(1, 0, 2))  # [(bs, num_q, embed_dims)]
                intermediate_reference_points.append(new_reference_points)  # (bs, num_q, num_points, 2)

        # 返回中间结果或最终结果
        if self.return_intermediate:
            return intermediate, intermediate_reference_points

        return output, reference_points

@TRANSFORMER_LAYER.register_module()
class ParticleTransformerLayer(BaseTransformerLayer):
    """Particle Transformer层实现
    
    类似于基本Transformer层，但为扩散模型添加了功能
    
    Args:
        attn_cfgs (list[dict] | dict | None): 注意力模块配置
        ffn_cfgs (dict): 前馈网络配置
        operation_order (tuple[str]): 操作执行顺序
        norm_cfg (dict): 归一化层配置
        with_time_embedding (bool): 是否使用时间嵌入
    """

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
                 with_time_embedding=False,
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
        self.with_time_embedding = with_time_embedding

    def forward(self,
                query,
                key=None,
                value=None,
                memory_query=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                time_embedding=None,
                **kwargs):
        """Transformer层前向传播函数
        
        Args:
            query: 输入查询张量
            key, value: 注意力机制的键值
            memory_query: 记忆查询
            query_pos: 查询位置编码
            key_pos: 键位置编码
            attn_masks: 注意力掩码
            query_key_padding_mask: 查询键填充掩码
            key_padding_mask: 键填充掩码
            time_embedding: 时间嵌入，用于扩散模型
            
        Returns:
            query: 转换后的查询
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        
        # 处理注意力掩码
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in {self.__class__.__name__}')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of attn_masks {len(attn_masks)} must be equal to the number of attention in operation_order {self.num_attn}'

        # 按操作顺序执行层
        for layer in self.operation_order:
            if layer == 'self_attn':
                if memory_query is None:
                    temp_key = temp_value = query
                else:
                    temp_key = temp_value = torch.cat([memory_query, query], dim=0)
                
                # 自注意力
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                # 归一化
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                # 交叉注意力
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                # 前馈网络
                # 如果有时间嵌入且层支持时间条件，则使用时间嵌入
                if self.with_time_embedding and time_embedding is not None:
                    # 将时间嵌入应用到FFN中
                    # 这里可以根据具体需求实现时间条件
                    query = self.ffns[ffn_index](
                        query, identity if self.pre_norm else None)
                else:
                    query = self.ffns[ffn_index](
                        query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

@TRANSFORMER.register_module()
class ParticleTransformer(Transformer):
    """实现了融合扩散模型思想的Particle Transformer
    
    Args:
        encoder (dict): 编码器配置
        decoder (dict): 解码器配置
        num_feature_levels (int): 特征层级数量
        num_points (int): 每个实例的关键点数量
        coord_dim (int): 坐标维度
        diffusion_cfg (dict): 扩散模型的配置
    """

    def __init__(self,
                 num_feature_levels=1,
                 num_points=20,
                 coord_dim=2,
                 diffusion_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_feature_levels = num_feature_levels
        self.embed_dims = self.decoder.embed_dims if self.encoder is None else self.encoder.embed_dims
        self.coord_dim = coord_dim
        self.num_points = num_points
        
        # 扩散模型配置
        self.diffusion_cfg = diffusion_cfg
        if diffusion_cfg:
            self.num_classes = diffusion_cfg.get('NUM_CLASSES', 1)
            self.num_proposals = diffusion_cfg.get('NUM_PROPOSALS', 100)
            self.hidden_dim = diffusion_cfg.get('HIDDEN_DIM', 256)
            self.radial_suppression_radius = diffusion_cfg.get('RADIAL_SUPPRESSION_RADIUS', 1.0)
            self.ddim_query_type = diffusion_cfg.get('DDIM_QUERY_TYPE', 'both')
            self.box_renewal_threshold = diffusion_cfg.get('BOX_RENEWAL_THRESHOLD', 0.3)
            self.use_diffusion = diffusion_cfg.get('USE_DIFFUSION', True)
            self.nms_threshold = float(diffusion_cfg.get('NMS_THRESHOLD', 0.5))
            self.scale = diffusion_cfg.get('SNR_SCALE', 1.0)
            
            # 扩散过程参数
            self.timesteps = diffusion_cfg.get('TIMESTEPS', 1000)
            self.sampling_timesteps = diffusion_cfg.get('SAMPLE_STEP', 50)
            self.objective = 'pred_x0'
            self.box_renewal = diffusion_cfg.get('BOX_RENEWAL', True)
            self.use_ensemble = diffusion_cfg.get('USE_ENSEMBLE', True)
            self.use_nms = diffusion_cfg.get('USE_NMS', True)
            self.self_condition = False
            self.ddim_sampling_eta = 0.
            
            # 设置扩散噪声调度
            self.setup_diffusion_scheduling()
        else:
            self.use_diffusion = False
        
        self.init_layers()
            
    def setup_diffusion_scheduling(self):
        """设置扩散过程的噪声调度参数"""
        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        
        # 注册缓冲区用于扩散计算
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # 前向扩散过程q(x_t|x_{t-1})的计算参数
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # 后验计算q(x_{t-1}|x_t,x_0)的参数
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def init_layers(self):
        """初始化Transformer的层"""
        if self.num_feature_levels > 1:
            self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        
        # 查询位置网络
        self.query_pos_net = nn.Linear(self.coord_dim, self.embed_dims)
        
        # 扩散模型相关层
        if self.use_diffusion:
            # 时间嵌入
            self.time_dim = 4 * self.embed_dims
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(self.embed_dims),
                nn.Linear(self.embed_dims, self.time_dim),
                nn.GELU(),
                nn.Linear(self.time_dim, self.time_dim)
            )
            
            # 时间块MLP
            self.block_time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.embed_dims * 4, self.embed_dims * 2)
            )
            
            # 位置编码到参考点
            self.positional_encoding_to_reference_point = nn.Linear(self.embed_dims, 2)

    def init_weights(self):
        """初始化Transformer权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # 初始化特定模块
        for m in self.modules():
            if isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        
        # 初始化位置编码和查询位置网络
        if hasattr(self, 'level_embeds'):
            normal_(self.level_embeds)
        
        xavier_init(self.query_pos_net)
        
        # 初始化时间嵌入和位置编码到参考点
        if self.use_diffusion:
            xavier_init(self.block_time_mlp[1])
            xavier_init(self.positional_encoding_to_reference_point)

    @auto_fp16(apply_to=('mlvl_feats',))
    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                init_reference_points,
                reg_branches=None,
                cls_branches=None,
                memory_query=None,
                prop_query=None,
                prop_reference_points=None,
                gt_bboxes_3d=None,
                gt_labels_3d=None,
                pc_range=None,
                is_first_frame_list=None,
                predict_refine=True,
                **kwargs):
        """ParticleTransformer的前向传播函数
        
        Args:
            mlvl_feats: 多层级特征列表，每个元素形状为[bs, embed_dims, h, w]
            mlvl_masks: 多层级掩码列表，每个元素形状为[bs, h, w]
            query_embed: 解码器的查询嵌入，形状为[num_query, c]
            mlvl_pos_embeds: 多层级位置编码列表
            init_reference_points: 初始参考点，形状为[bs, num_query, num_points, 2]
            reg_branches: 回归分支
            cls_branches: 分类分支
            memory_query: 记忆查询
            prop_query: 传播查询
            prop_reference_points: 传播参考点
            gt_bboxes_3d: 3D边界框真值
            gt_labels_3d: 3D标签真值
            pc_range: 点云范围
            is_first_frame_list: 是否是第一帧的列表
            predict_refine: 是否细化预测
            
        Returns:
            tuple: 解码器输出和参考点
        """
        # 处理多层级特征
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        
        # 展平多层级特征
        for lvl, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            
            # 展平特征和掩码
            feat = feat.flatten(2).transpose(1, 2)  # [bs, h*w, c]
            mask = mask.flatten(1)  # [bs, h*w]
            
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        
        # 连接展平的特征和掩码
        feat_flatten = torch.cat(feat_flatten, 1)  # [bs, sum(h*w), c]
        mask_flatten = torch.cat(mask_flatten, 1)  # [bs, sum(h*w)]
        
        # 转换空间形状为张量
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        # 转置特征以适应transformer
        feat_flatten = feat_flatten.permute(1, 0, 2)  # [sum(h*w), bs, c]
        
        # 获取解码器的输入
        query = query_embed.permute(1, 0, 2)  # [num_q, bs, c]
        
        # 处理记忆查询
        if memory_query is not None:
            memory_query = memory_query.permute(1, 0, 2)
        
        # 判断是否使用扩散模型
        if self.use_diffusion and (gt_bboxes_3d is not None) and (gt_labels_3d is not None):
            # 训练阶段 - 使用扩散模型
            device = mlvl_feats[0].device
            bs = mlvl_feats[0].size(0)
            
            # 准备目标与扩散样本
            targets, x_boxes, noises, t = self.prep_targets(gt_bboxes_3d, gt_labels_3d, pc_range=pc_range)
            
            # 调整boxes坐标到[0, 1]，(0,0)为左上角，(1,1)为右下角
            x_boxes[..., 1] = 1 - x_boxes[..., 1]
            
            # 获取时间嵌入
            time_embedding = self.time_mlp(t)
            
            # 生成参考点
            reference_points_diffusion = x_boxes.to(device)
            query_pos_diffusion = self.query_pos_net(reference_points_diffusion)
            
            # 整合传播与当前查询
            query_integration = query.clone()
            query_pos_integration = None
            if query_pos_diffusion is not None:
                query_pos_integration = query_pos_diffusion.permute(1, 0, 2)
            
            reference_points = reference_points_diffusion
            
            # 运行解码器
            inter_states, inter_references = self.decoder(
                query=query_integration,
                key=None,
                value=feat_flatten,
                query_pos=query_pos_integration,
                key_padding_mask=mask_flatten,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                memory_query=memory_query,
                prop_query=prop_query,
                prop_reference_points=prop_reference_points,
                is_first_frame_list=is_first_frame_list,
                predict_refine=predict_refine,
                time_embedding=time_embedding,
                **kwargs)
            
            # 对扩散查询应用时间条件缩放和偏移
            n_diffusion_queries = x_boxes.size(1)
            scale_shift = self.block_time_mlp(time_embedding)
            scale_shift = torch.repeat_interleave(scale_shift, n_diffusion_queries, dim=0)[None]
            scale, shift = scale_shift.chunk(2, dim=-1)
            
            # 调整解码器输出
            inter_states2 = torch.empty_like(inter_states)
            inter_states2 = inter_states * (scale + 1) + shift
            inter_states = inter_states2
            
            return feat_flatten, inter_states, init_reference_points, inter_references
            
        elif self.use_diffusion and (gt_bboxes_3d is None) and (gt_labels_3d is None):
            # 推理阶段 - 使用扩散采样
            outputs_class, outputs_coord = self.ddim_sample(
                backbone_feats=feat_flatten,
                spatial_shapes=spatial_shapes, 
                level_start_index=level_start_index,
                mask_flatten=mask_flatten,
                query_embed=query_embed,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                predict_refine=predict_refine,
                pc_range=pc_range,
                **kwargs)
            
            return feat_flatten, outputs_class, outputs_coord
            
        else:
            # 不使用扩散模型 - 标准transformer流程
            inter_states, inter_references = self.decoder(
                query=query,
                key=None,
                value=feat_flatten,
                query_pos=None,
                key_padding_mask=mask_flatten,
                reference_points=init_reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                memory_query=memory_query,
                prop_query=prop_query,
                prop_reference_points=prop_reference_points,
                is_first_frame_list=is_first_frame_list,
                predict_refine=predict_refine,
                **kwargs)
            
            return inter_states, init_reference_points, inter_references

    # 扩散模型的方法
    def prep_targets(self, gt_bboxes_3d, gt_labels_3d, pc_range=None):
        """准备扩散模型的训练目标
        
        Args:
            gt_bboxes_3d: 3D边界框真值
            gt_labels_3d: 3D标签真值
            pc_range: 点云范围
            
        Returns:
            tuple: 目标，扩散后的框，噪声，时间步
        """
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        
        # 准备设备和坐标范围
        device = gt_bboxes_3d[0].tensor.device
        coord_minimum = torch.Tensor([pc_range[0], pc_range[1]]).to(device)
        coord_range = torch.Tensor([pc_range[3] - pc_range[0], pc_range[4] - pc_range[1]]).to(device)
        
        # 处理每个gt框
        gt_bboxes_3d = [i.tensor.to(device) for i in gt_bboxes_3d]
        gt_labels_3d = [i.to(device) for i in gt_labels_3d]
        targets = [AnnotObject(gt_boxes=i, gt_classes=j) for (i, j) in zip(gt_bboxes_3d, gt_labels_3d)]
        
        for targets_per_image in targets:
            target = {}
            
            # 提取中心点
            gt_centers = targets_per_image.gt_boxes[:, [0, 1]]
            
            # 归一化到[0, 1]
            gt_centers_normalized = (gt_centers - coord_minimum) / coord_range
            
            # 添加扩散
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_centers_normalized)
            
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            target["labels"] = targets_per_image.gt_classes
            target["centers"] = gt_centers
            target['centers_normalized'] = gt_centers_normalized
            new_targets.append(target)
            
        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)

    def prepare_diffusion_concat(self, gt_boxes):
        """准备用于扩散模型的训练数据
        
        Args:
            gt_boxes: 真值框，形状为[N, 2]，元素为(cx, cy)，范围在[0, 1]
            
        Returns:
            tuple: 扩散后的框，噪声，时间步
        """
        device = gt_boxes.device
        t = torch.randint(0, self.timesteps, (1,), device=device).long()
        noise = torch.randn(self.num_proposals, 2, device=device)
        
        num_gt = gt_boxes.shape[0]
        if not num_gt:  # 如果没有gt框则生成假的
            gt_boxes = torch.as_tensor([[0.5, 0.5]], dtype=torch.float, device=device)
            num_gt = 1
        
        # 处理GT框数量与proposal数量不匹配的情况
        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 2, device=device) / 6. + 0.5  # N(mu=1/2, sigma=1/6)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes
        
        # 缩放并添加噪声
        x_start = (x_start * 2. - 1.) * self.scale
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.
        diff_boxes = x
        
        return diff_boxes, noise, t

    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程 - 添加噪声
        
        Args:
            x_start: 起始状态，形状为[N, 2]
            t: 时间步
            noise: 噪声，如果为None则生成随机噪声
            
        Returns:
            x_t: t时刻的状态
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t, t, x0):
        """从起始状态预测噪声
        
        Args:
            x_t: t时刻的状态
            t: 时间步
            x0: 预测的初始状态
            
        Returns:
            predicted_noise: 预测的噪声
        """
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    @torch.no_grad()
    def ddim_sample(self, backbone_feats, spatial_shapes, level_start_index, mask_flatten, 
                   query_embed, reg_branches=None, cls_branches=None, clip_denoised=True, 
                   predict_refine=True, pc_range=None, **kwargs):
        """使用DDIM算法进行采样
        
        Args:
            backbone_feats: 骨干网络特征
            spatial_shapes: 空间形状
            level_start_index: 层级起始索引
            mask_flatten: 展平的掩码
            query_embed: 查询嵌入
            reg_branches: 回归分支
            cls_branches: 分类分支
            clip_denoised: 是否裁剪去噪后的输出
            predict_refine: 是否细化预测
            pc_range: 点云范围
            
        Returns:
            tuple: 分类输出和坐标输出
        """
        batch = backbone_feats.size(1)  # [H*W, bs, c]形状的backbone_feats
        shape = (batch, self.num_proposals, 2)
        total_timesteps, sampling_timesteps = self.timesteps, self.sampling_timesteps
        
        # 时间步列表
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        # 从标准正态分布初始化起始噪声
        img = torch.randn(shape, device=backbone_feats.device)
        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        outputs_coords_all, outputs_class_all = [], []
        x_start = None
        
        # 逐步去噪过程
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=backbone_feats.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            
            # 获取当前状态下的模型预测
            preds, outputs_class, outputs_coord = self.model_predictions(
                backbone_feats=backbone_feats,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                mask_flatten=mask_flatten,
                x=img,
                query_embed=query_embed,
                time_cond=time_cond,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                self_cond=self_cond,
                clip_x_start=clip_denoised,
                predict_refine=predict_refine,
                pc_range=pc_range,
                **kwargs
            )
            
            # 从预测中提取噪声和起始状态
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            
            # 盒子更新逻辑
            if self.box_renewal:
                score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > self.box_renewal_threshold
                num_remain = torch.sum(keep_idx)
                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]
            
            # 最后一步直接使用预测的x_start
            if time_next < 0:
                img = x_start
                if self.use_ensemble and self.sampling_timesteps > 1:
                    # 存储所有输出用于集成
                    outputs_coords_all.append(outputs_coord)
                    outputs_class_all.append(outputs_class)
                continue
            
            # 计算下一步
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            
            # 更新img
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            
            # 如果使用盒子更新，需要补充剩余的位置
            if self.box_renewal and num_remain < self.num_proposals:
                img = torch.cat((img, torch.randn(batch, self.num_proposals - num_remain, 2, device=img.device)), dim=1)
            
            # 存储DDIM输出
            if self.use_ensemble and self.sampling_timesteps > 1:
                outputs_coords_all.append(outputs_coord)
                outputs_class_all.append(outputs_class)
                
        # 后处理 - 可以添加NMS和集成等操作
        
        return outputs_class, outputs_coord

    def model_predictions(self, backbone_feats, spatial_shapes, level_start_index, mask_flatten,
                         x, query_embed, time_cond, reg_branches=None, cls_branches=None, 
                         self_cond=None, clip_x_start=False, predict_refine=True, pc_range=None, **kwargs):
        """生成模型预测
        
        Args:
            backbone_feats: 骨干网络特征
            spatial_shapes: 空间形状
            level_start_index: 层级起始索引
            mask_flatten: 展平的掩码
            x: 当前状态
            query_embed: 查询嵌入
            time_cond: 时间条件
            reg_branches: 回归分支
            cls_branches: 分类分支
            self_cond: 自条件
            clip_x_start: 是否裁剪x_start
            predict_refine: 是否细化预测
            pc_range: 点云范围
            
        Returns:
            tuple: 模型预测，分类输出，坐标输出
        """
        # 处理x的范围
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2  # 映射到[0, 1]
        
        # 调整y轴
        x_boxes[..., 1] = 1 - x_boxes[..., 1]
        bs = x_boxes.size(0)
        
        # 创建reference points
        reference_points = x_boxes.to(backbone_feats.dtype)
        
        # 获取时间嵌入
        if len(time_cond.size()) == 1:
            time_cond = time_cond.view(1, 1)
        time_embedding = self.time_mlp(time_cond)
        
        # 准备查询
        query_pos = self.query_pos_net(reference_points)
        
        # 整合查询和位置编码
        query = query_embed.permute(1, 0, 2)  # [num_q, bs, c]
        query_pos = query_pos.permute(1, 0, 2)  # [N, bs, c]
        
        # 运行解码器
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=backbone_feats,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            time_embedding=time_embedding,
            predict_refine=predict_refine,
            **kwargs
        )
        
        # 准备scale和shift
        scale_shift = self.block_time_mlp(time_embedding)
        scale_shift = torch.repeat_interleave(scale_shift, self.num_proposals, dim=0)[None]
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        # 调整中间状态
        inter_states = inter_states * (scale + 1) + shift
        
        # 获取分类和坐标预测
        hs = inter_states.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = reference_points
            else:
                reference = inter_references[lvl - 1]
            
            outputs_class = cls_branches[lvl](hs[lvl])
            tmp = reg_branches[lvl](hs[lvl])
            
            # 调整坐标
            if reference.shape[-1] == 2:
                tmp[..., 0:2] = (tmp[..., 0:2] + inverse_sigmoid(reference)).sigmoid()
            
            # 缩放到输出大小
            if tmp.size(-1) > 2:
                tmp[..., [0]] = (tmp[..., [0]] * (pc_range[3] - pc_range[0]) + pc_range[0])
                tmp[..., [1]] = (tmp[..., [1]] * (pc_range[4] - pc_range[1]) + pc_range[1])
            else:
                tmp[..., [0]] = (tmp[..., [0]] * (pc_range[3] - pc_range[0]) + pc_range[0])
                tmp[..., [1]] = (tmp[..., [1]] * (pc_range[4] - pc_range[1]) + pc_range[1])
            
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        
        # 堆叠输出
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        
        # 获取扩散预测
        x_start_pred = outputs_coords[-1, :, :, [0, 1]]
        
        # 将坐标归一化到[-scale, scale]
        device = x_start_pred.device
        coord_minimum = torch.Tensor([pc_range[0], pc_range[1]]).to(device)
        coord_range = torch.Tensor([pc_range[3] - pc_range[0], pc_range[4] - pc_range[1]]).to(device)
        x_start_norm = (x_start_pred - coord_minimum) / coord_range  # (cx, cy) in [0, 1]
        
        x_start = (x_start_norm * 2 - 1.) * self.scale  # [-scale, scale]
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        
        # 预测噪声
        pred_noise = self.predict_noise_from_start(x, time_cond.squeeze(-1), x_start)
        
        return ModelPrediction(pred_noise, x_start), outputs_classes, outputs_coords

    def average_boxes_in_radius(self, boxes, logits, logits_full, classes, radius=1.0):
        """
        类似于NMS，但只考虑中心距离（而不是IOU），并根据置信度平均框。
        
        Args:
            boxes: 框坐标，形状为[N, C]
            logits: 置信度logits，形状为[N]
            logits_full: 完整的logits，形状为[N, K]
            classes: 类别，形状为[N]
            radius: 聚类半径
            
        Returns:
            tuple: 平均后的框和logits
        """
        new_boxes = []
        new_logits_full = []
        
        # 按类别处理
        for class_id in classes.unique():
            # 获取该类别的框
            ids_for_that_class = torch.where(classes == class_id)[0]
            boxes_for_that_class = boxes[ids_for_that_class]
            logits_for_that_class = logits[ids_for_that_class].sigmoid()
            logits_full_for_that_class = logits_full[ids_for_that_class]
            
            # 按置信度降序排序
            order = torch.argsort(logits_for_that_class, descending=True)
            keep = torch.ones_like(order, dtype=torch.bool)
            
            # 计算成对距离
            pairwise_dist = torch.cdist(boxes_for_that_class[..., :2], boxes_for_that_class[..., :2], p=1)
            
            # 平均半径内的框
            for i in order:
                if keep[i]:
                    indices = torch.nonzero(pairwise_dist[i] < radius)
                    if len(indices.size()) > 1:
                        indices = indices.squeeze()
                    if len(indices.size()) < 1:
                        continue
                    
                    # 计算加权平均
                    average_box = (logits_for_that_class[indices][:, None] * boxes_for_that_class[indices]).sum(0) / logits_for_that_class[indices].sum()
                    average_conf = (logits_for_that_class[indices][:, None] * logits_full_for_that_class[indices]).sum(0) / logits_for_that_class[indices].sum()
                    
                    # 更新keep标志
                    keep[indices] = False
                    keep[i] = True
                    boxes_for_that_class[i] = average_box
                    logits_full_for_that_class[i] = average_conf
            
            # 添加保留的框
            new_boxes.append(boxes_for_that_class[keep])
            new_logits_full.append(logits_full_for_that_class[keep])
        
        # 连接所有类别的结果
        if new_boxes:
            return torch.cat(new_boxes, dim=0), torch.cat(new_logits_full, dim=0)
        else:
            return boxes, logits_full