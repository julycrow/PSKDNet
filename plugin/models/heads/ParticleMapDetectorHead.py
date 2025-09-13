import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob, xavier_init
from mmcv.runner import force_fp32
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils import build_transformer
from mmdet.models import build_loss

from mmdet.core import multi_apply, reduce_mean, build_assigner, build_sampler
from mmdet.models import HEADS
from mmdet.models.utils.transformer import inverse_sigmoid
from ..utils.memory_buffer import StreamTensorMemory
from ..utils.query_update import MotionMLP

@HEADS.register_module(force=True)
class ParticleMapDetectorHead(nn.Module):

    def __init__(self,
                 num_queries,
                 num_classes=3,
                 in_channels=128,
                 embed_dims=256,
                 score_thr=0.1,
                 num_points=20,
                 coord_dim=2,
                 roi_size=(60, 30),
                 different_heads=True,
                 predict_refine=False,
                 bev_pos=None,
                 sync_cls_avg_factor=True,
                 bg_cls_weight=0.,
                 streaming_cfg=dict(),
                 transformer=dict(),
                 loss_cls=dict(),
                 loss_reg=dict(),
                 assigner=dict(),
                 # 新增参数，支持ParticleTransformer
                 use_particle_transformer=False,
                 num_bevformer_queries=None,
                 num_diffusion_queries=None,
                 diffusion_loss_weight=1.0,
                 diffusion_cfg=None,
                 ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.different_heads = different_heads
        self.predict_refine = predict_refine
        self.bev_pos = bev_pos
        self.num_points = num_points
        self.coord_dim = coord_dim

        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.bg_cls_weight = bg_cls_weight

        # 新增：ParticleTransformer相关配置
        self.use_particle_transformer = use_particle_transformer
        if self.use_particle_transformer:
            self.num_bevformer_queries = num_bevformer_queries if num_bevformer_queries is not None else num_queries
            self.num_diffusion_queries = num_diffusion_queries if num_diffusion_queries is not None else num_queries
            self.diffusion_loss_weight = diffusion_loss_weight
            self.diffusion_cfg = diffusion_cfg

        if streaming_cfg:
            self.streaming_query = streaming_cfg['streaming']
        else:
            self.streaming_query = False
        if self.streaming_query:
            self.batch_size = streaming_cfg['batch_size']
            self.topk_query = streaming_cfg['topk']
            self.trans_loss_weight = streaming_cfg.get('trans_loss_weight', 0.0)
            self.query_memory = StreamTensorMemory(
                self.batch_size,
            )
            self.reference_points_memory = StreamTensorMemory(
                self.batch_size,
            )
            c_dim = 12

            self.query_update = MotionMLP(c_dim=c_dim, f_dim=self.embed_dims, identity=True)
            self.target_memory = StreamTensorMemory(self.batch_size)

        self.register_buffer('roi_size', torch.tensor(roi_size, dtype=torch.float32))
        origin = (-roi_size[0] / 2, -roi_size[1] / 2)
        self.register_buffer('origin', torch.tensor(origin, dtype=torch.float32))

        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg, context=self)

        self.transformer = build_transformer(transformer)

        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)
        self.assigner = build_assigner(assigner)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self._init_embedding()
        self._init_branch()
        self.init_weights()

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""

        for p in self.input_proj.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        xavier_init(self.reference_points_embed, distribution='uniform', bias=0.)

        self.transformer.init_weights()

        # init prediction branch
        for m in self.reg_branches:
            for param in m.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

        # focal loss init
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            if isinstance(self.cls_branches, nn.ModuleList):
                for m in self.cls_branches:
                    if hasattr(m, 'bias'):
                        nn.init.constant_(m.bias, bias_init)
            else:
                m = self.cls_branches
                nn.init.constant_(m.bias, bias_init)

        if self.streaming_query:
            if isinstance(self.query_update, MotionMLP):
                self.query_update.init_weights()
            if hasattr(self, 'query_alpha'):
                for m in self.query_alpha:
                    for param in m.parameters():
                        if param.dim() > 1:
                            nn.init.zeros_(param)

    def _init_embedding(self):
        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=self.embed_dims // 2,
            normalize=True
        )
        self.bev_pos_embed = build_positional_encoding(positional_encoding)

        # query_pos_embed & query_embed
        self.query_embedding = nn.Embedding(self.num_queries,
                                            self.embed_dims)

        self.reference_points_embed = nn.Linear(self.embed_dims, self.num_points * 2)

        # 新增：为ParticleTransformer添加query embedding
        if self.use_particle_transformer:
            self.query_embedding_bevformer = nn.Embedding(self.num_bevformer_queries, 2 * self.embed_dims)
            self.query_embedding_diffusion = nn.Embedding(self.num_diffusion_queries, self.embed_dims)

    def _init_branch(self, ):
        """Initialize classification branch and regression branch of head."""
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = Linear(self.embed_dims, self.cls_out_channels)

        reg_branch = [
            Linear(self.embed_dims, 2 * self.embed_dims),
            nn.LayerNorm(2 * self.embed_dims),
            nn.ReLU(),
            Linear(2 * self.embed_dims, 2 * self.embed_dims),
            nn.LayerNorm(2 * self.embed_dims),
            nn.ReLU(),
            Linear(2 * self.embed_dims, self.num_points * self.coord_dim),
        ]
        reg_branch = nn.Sequential(*reg_branch)

        num_layers = self.transformer.decoder.num_layers
        if self.different_heads:
            cls_branches = nn.ModuleList(
                [copy.deepcopy(cls_branch) for _ in range(num_layers)])
            reg_branches = nn.ModuleList(
                [copy.deepcopy(reg_branch) for _ in range(num_layers)])
        else:
            cls_branches = nn.ModuleList(
                [cls_branch for _ in range(num_layers)])
            reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_layers)])

        self.reg_branches = reg_branches
        self.cls_branches = cls_branches

    def _prepare_context(self, bev_features):
        """Prepare class label and vertex context."""
        device = bev_features.device

        # Add 2D coordinate grid embedding
        B, C, H, W = bev_features.shape
        bev_mask = bev_features.new_zeros(B, H, W)
        bev_pos_embeddings = self.bev_pos_embed(bev_mask)  # (bs, embed_dims, H, W)
        bev_features = self.input_proj(bev_features) + bev_pos_embeddings  # (bs, embed_dims, H, W)

        assert list(bev_features.shape) == [B, self.embed_dims, H, W]
        return bev_features

    def propagate(self, query_embedding, img_metas, return_loss=True):
        bs = query_embedding.shape[0]
        propagated_query_list = []
        prop_reference_points_list = []

        tmp = self.query_memory.get(img_metas)
        query_memory, pose_memory = tmp['tensor'], tmp['img_metas']

        tmp = self.reference_points_memory.get(img_metas)
        ref_pts_memory, pose_memory = tmp['tensor'], tmp['img_metas']

        if return_loss and hasattr(self, 'target_memory'):
            target_memory = self.target_memory.get(img_metas)['tensor']
            trans_loss = query_embedding.new_zeros((1,))
            num_pos = 0

        is_first_frame_list = tmp['is_first_frame']

        # 分开处理BEV和diffusion特征
        half_topk = self.topk_query  #  // 2

        for i in range(bs):
            is_first_frame = is_first_frame_list[i]
            if is_first_frame:
                # 为BEV和diffusion分别创建填充
                bev_padding = query_embedding.new_zeros((half_topk, self.embed_dims))
                diff_padding = query_embedding.new_zeros((half_topk, self.embed_dims))
                padding = torch.cat([bev_padding, diff_padding], dim=0)
                propagated_query_list.append(padding)

                bev_ref_padding = query_embedding.new_zeros((half_topk, self.num_points, 2))
                diff_ref_padding = query_embedding.new_zeros((half_topk, self.num_points, 2))
                ref_padding = torch.cat([bev_ref_padding, diff_ref_padding], dim=0)
                prop_reference_points_list.append(ref_padding)
            else:
                # 坐标变换矩阵计算
                prev_e2g_trans = self.roi_size.new_tensor(pose_memory[i]['ego2global_translation'], dtype=torch.float64)
                prev_e2g_rot = self.roi_size.new_tensor(pose_memory[i]['ego2global_rotation'], dtype=torch.float64)
                curr_e2g_trans = self.roi_size.new_tensor(img_metas[i]['ego2global_translation'], dtype=torch.float64)
                curr_e2g_rot = self.roi_size.new_tensor(img_metas[i]['ego2global_rotation'], dtype=torch.float64)

                prev_e2g_matrix = torch.eye(4, dtype=torch.float64).to(query_embedding.device)
                prev_e2g_matrix[:3, :3] = prev_e2g_rot
                prev_e2g_matrix[:3, 3] = prev_e2g_trans

                curr_g2e_matrix = torch.eye(4, dtype=torch.float64).to(query_embedding.device)
                curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
                curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

                prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
                pos_encoding = prev2curr_matrix.float()[:3].view(-1)

                # 分别处理BEV和diffusion查询
                prop_q_bev = query_memory[i][:half_topk]
                prop_q_diff = query_memory[i][half_topk:]

                # 分别更新查询
                query_bev_updated = self.query_update(
                    prop_q_bev,
                    pos_encoding.view(1, -1).repeat(len(prop_q_bev), 1)
                )

                query_diff_updated = self.query_update(
                    prop_q_diff,
                    pos_encoding.view(1, -1).repeat(len(prop_q_diff), 1)
                )

                # 合并更新后的查询
                query_memory_updated = torch.cat([query_bev_updated, query_diff_updated], dim=0)
                propagated_query_list.append(query_memory_updated.clone())

                # 分别处理参考点
                ref_pts_bev = ref_pts_memory[i][:half_topk]
                ref_pts_diff = ref_pts_memory[i][half_topk:]

                # 对BEV参考点进行变换
                denormed_ref_pts_bev = ref_pts_bev * self.roi_size + self.origin
                denormed_ref_pts_bev = torch.cat([
                    denormed_ref_pts_bev,
                    denormed_ref_pts_bev.new_zeros((half_topk, self.num_points, 1)),
                    denormed_ref_pts_bev.new_ones((half_topk, self.num_points, 1))
                ], dim=-1)
                curr_ref_pts_bev = torch.einsum('lk,ijk->ijl', prev2curr_matrix, denormed_ref_pts_bev.double()).float()
                normed_ref_pts_bev = (curr_ref_pts_bev[..., :2] - self.origin) / self.roi_size
                normed_ref_pts_bev = torch.clip(normed_ref_pts_bev, min=0., max=1.)

                # 对diffusion参考点进行变换
                denormed_ref_pts_diff = ref_pts_diff * self.roi_size + self.origin
                denormed_ref_pts_diff = torch.cat([
                    denormed_ref_pts_diff,
                    denormed_ref_pts_diff.new_zeros((half_topk, self.num_points, 1)),
                    denormed_ref_pts_diff.new_ones((half_topk, self.num_points, 1))
                ], dim=-1)
                curr_ref_pts_diff = torch.einsum('lk,ijk->ijl', prev2curr_matrix,
                                                 denormed_ref_pts_diff.double()).float()
                normed_ref_pts_diff = (curr_ref_pts_diff[..., :2] - self.origin) / self.roi_size
                normed_ref_pts_diff = torch.clip(normed_ref_pts_diff, min=0., max=1.)

                # 合并参考点
                prop_ref_pts = torch.cat([normed_ref_pts_bev, normed_ref_pts_diff], dim=0)
                prop_reference_points_list.append(prop_ref_pts)

                # 处理损失计算
                if return_loss:
                    targets = target_memory[i]
                    target_bev = targets[:half_topk]
                    target_diff = targets[half_topk:]

                    # 处理BEV部分的损失
                    weights_bev = target_bev.new_ones((half_topk, 2 * self.num_points))
                    bg_idx_bev = torch.all(target_bev.view(half_topk, -1) == 0.0, dim=1)
                    weights_bev[bg_idx_bev, :] = 0.0

                    # 处理diffusion部分的损失
                    weights_diff = target_diff.new_ones((half_topk, 2 * self.num_points))
                    bg_idx_diff = torch.all(target_diff.view(half_topk, -1) == 0.0, dim=1)
                    weights_diff[bg_idx_diff, :] = 0.0

                    num_pos = num_pos + ((half_topk - bg_idx_bev.sum()) + (half_topk - bg_idx_diff.sum()))

                    # 处理BEV目标
                    denormed_targets_bev = target_bev * self.roi_size + self.origin
                    denormed_targets_bev = torch.cat([
                        denormed_targets_bev,
                        denormed_targets_bev.new_zeros((half_topk, self.num_points, 1)),
                        denormed_targets_bev.new_ones((half_topk, self.num_points, 1))
                    ], dim=-1)
                    curr_targets_bev = torch.einsum('lk,ijk->ijl', prev2curr_matrix.float(), denormed_targets_bev)
                    normed_targets_bev = (curr_targets_bev[..., :2] - self.origin) / self.roi_size
                    normed_targets_bev = torch.clip(normed_targets_bev, min=0., max=1.).reshape(-1, 2 * self.num_points)

                    # 处理diffusion目标
                    denormed_targets_diff = target_diff * self.roi_size + self.origin
                    denormed_targets_diff = torch.cat([
                        denormed_targets_diff,
                        denormed_targets_diff.new_zeros((half_topk, self.num_points, 1)),
                        denormed_targets_diff.new_ones((half_topk, self.num_points, 1))
                    ], dim=-1)
                    curr_targets_diff = torch.einsum('lk,ijk->ijl', prev2curr_matrix.float(), denormed_targets_diff)
                    normed_targets_diff = (curr_targets_diff[..., :2] - self.origin) / self.roi_size
                    normed_targets_diff = torch.clip(normed_targets_diff, min=0., max=1.).reshape(-1,
                                                                                                  2 * self.num_points)

                    # 计算BEV部分损失
                    pred_bev = self.reg_branches[-1](query_bev_updated).sigmoid()
                    trans_loss += self.loss_reg(pred_bev, normed_targets_bev, weights_bev, avg_factor=1.0)

                    # 计算diffusion部分损失
                    pred_diff = self.reg_branches[-1](query_diff_updated).sigmoid()
                    trans_loss += self.loss_reg(pred_diff, normed_targets_diff, weights_diff, avg_factor=1.0)

        # 合并结果
        prop_query_embedding = torch.stack(propagated_query_list)
        prop_ref_pts = torch.stack(prop_reference_points_list)

        # 初始化其他参数
        # init_reference_points = self.reference_points_embed(query_embedding).sigmoid()
        init_reference_points = self.reference_points_embed(query_embedding).sigmoid()
        init_reference_points = init_reference_points.view(bs, query_embedding.shape[1], self.num_points, 2)  # torch.Size([4, 200, 20, 2])
        memory_query_embedding = None

        if return_loss:
            trans_loss = self.trans_loss_weight * trans_loss / (num_pos + 1e-10)
            return query_embedding, prop_query_embedding, init_reference_points, prop_ref_pts, memory_query_embedding, is_first_frame_list, trans_loss
        else:
            return query_embedding, prop_query_embedding, init_reference_points, prop_ref_pts, memory_query_embedding, is_first_frame_list

    def forward_train(self, bev_features, img_metas, gts):
        '''
        Args:
            bev_feature (List[Tensor]): shape [B, C, H, W]
                feature in bev view
        Outs:
            preds_dict (list[dict]):
                lines (Tensor): Classification score of all
                    decoder layers, has shape
                    [bs, num_query, 2*num_points]
                scores (Tensor):
                    [bs, num_query,]
        '''

        bev_features = self._prepare_context(bev_features)
        attn_mask = (torch.zeros([self.num_queries * 2, self.num_queries * 2,]).bool().to(bev_features.device))

        bs, C, H, W = bev_features.shape
        img_masks = bev_features.new_zeros((bs, H, W))
        # pos_embed = self.positional_encoding(img_masks)
        pos_embed = None

        # 为ParticleTransformer准备查询嵌入
        object_query_embeds_bevformer = self.query_embedding_bevformer.weight.to(bev_features.dtype)  # torch.Size([100, 1024])
        object_query_embeds_diffusion = self.query_embedding_diffusion.weight.to(bev_features.dtype)  # torch.Size([100, 512])
        object_query_embeds_bevformer = object_query_embeds_bevformer.repeat(bs, 1, 1)  # torch.Size([4, 100, 1024])
        object_query_embeds_diffusion = object_query_embeds_diffusion.repeat(bs, 1, 1)  # torch.Size([4, 100, 512])
        # object_query_embeds_diffusion = torch.empty((bs, 0, 512), device=object_query_embeds_bevformer.device, dtype=object_query_embeds_diffusion.dtype)
        query_embedding_bevformer, query_pos_bevformer = torch.split(
            object_query_embeds_bevformer, self.embed_dims, dim=2)  # 各自形状: [4, 100, 512]
        object_query_embeds_bevformer = torch.cat([query_embedding_bevformer, query_pos_bevformer], dim=1)  # [4, 200, 512]
        query_embedding_combined = torch.cat([query_embedding_bevformer, object_query_embeds_diffusion],
                                             dim=1)  # query_embedding_combined 形状: [4, 200, 512]

        if self.streaming_query:
            query_embedding_combined, prop_query_embedding, init_reference_points, prop_ref_pts, memory_query, is_first_frame_list, trans_loss = \
                self.propagate(query_embedding_combined, img_metas, return_loss=True)
        else:
            init_reference_points = self.reference_points_embed(query_embedding_combined).sigmoid()  # (bs, num_q, 2*num_pts)
            init_reference_points = init_reference_points.view(-1, self.num_queries * 2, self.num_points,
                                                               2)  # (bs, num_q * 2, num_pts, 2)
            prop_query_embedding = None
            prop_ref_pts = None
            is_first_frame_list = [True for i in range(bs)]
        # assert list(init_reference_points.shape) == [bs, self.num_queries * 2, self.num_points, 2]
        # assert list(object_query_embeds_bevformer.shape) == [bs, self.num_queries * 2, self.embed_dims]
        # 使用ParticleTransformer处理BEV特征
        outputs = self.transformer(
            mlvl_feats=[bev_features, ],
            mlvl_masks=[img_masks.type(torch.bool)],
            mlvl_pos_embeds=[pos_embed],
            bev_embed=bev_features.flatten(2).permute(0, 2, 1),  # 转换为(B, H*W, C)格式
            object_query_embeds_bevformer=object_query_embeds_bevformer,
            object_query_embeds_diffusion=object_query_embeds_diffusion,
            # query_embed=object_query_embeds_bevformer,
            prop_query=prop_query_embedding,
            prop_reference_points=prop_ref_pts,
            init_reference_points=init_reference_points,  # torch.Size([4, 100, 20, 2])
            bev_h=H,
            bev_w=W,
            grid_length=[self.roi_size[1] / H, self.roi_size[0] / W],
            reg_branches=self.reg_branches,
            cls_branches=self.cls_branches,
            prev_bev=None,
            gt_points=gts[0]['lines'],
            gt_labels=gts[0]['labels'],
            pc_range=[self.origin[0], self.origin[1], -10,
                      self.origin[0] + self.roi_size[0],
                      self.origin[1] + self.roi_size[1], 10],
            predict_refine=self.predict_refine,
            is_first_frame_list=is_first_frame_list,
            query_key_padding_mask=prop_query_embedding.new_zeros((bs, self.num_queries), dtype=torch.bool),
            num_bevquery=self.num_queries,
            attn_mask=attn_mask,
        )

        bev_embed, inter_states, init_reference_out, inter_references_out, diffusion_outputs = outputs
        outputs = []
        for i, (queries) in enumerate(inter_states):
            reg_points = inter_references_out[i]  # (bs, num_q, num_points, 2)
            bs = reg_points.shape[0]
            reg_points = reg_points.view(bs, -1, 2 * self.num_points)  # (bs, num_q, 2*num_points)

            scores = self.cls_branches[i](queries)  # (bs, num_q, num_classes)

            # if self.predict_refine:
            #     tmp = self.reg_branches[i](queries)
            #     reg_points = reg_points + inverse_sigmoid(tmp)  # (新加入的,可能影响初始损失和梯度)reg_points已经是归一化过的了
            reg_points_list = []
            scores_list = []
            for i in range(len(scores)):
                # padding queries should not be output
                reg_points_list.append(reg_points[i])
                scores_list.append(scores[i])

            pred_dict = {
                'lines': reg_points_list,
                'scores': scores_list
            }
            outputs.append(pred_dict)
        # 计算损失
        loss_dict, det_match_idxs, det_match_gt_idxs, gt_lines_list = self.loss(gts=gts, preds=outputs, diffusion_outputs=diffusion_outputs)
        if self.streaming_query:
            query_list = []
            ref_pts_list = []
            gt_targets_list = []
            lines, scores = outputs[-1]['lines'], outputs[-1]['scores']
            gt_lines = gt_lines_list[-1]  # take results from the last layer

            for i in range(bs):
                _lines = lines[i]
                _queries = inter_states[-1][i]
                _scores = scores[i]
                _gt_targets = gt_lines[i]
                assert len(_lines) == len(_queries)
                assert len(_lines) == len(_gt_targets)

                # 拆分为 bevformer 和 diffusion 两部分
                n_bevformer_queries = self.num_queries  # 假设是原始的查询数量
                n_diffusion_queries = len(_queries) - n_bevformer_queries

                # 分别处理 bevformer 部分
                _bev_queries = _queries[:n_bevformer_queries]
                _bev_scores = _scores[:n_bevformer_queries]
                _bev_lines = _lines[:n_bevformer_queries]
                _bev_gt_targets = _gt_targets[:n_bevformer_queries]

                # 对 bevformer 部分取 topk
                _bev_scores_max, _ = _bev_scores.max(-1)
                topk_bev = self.topk_query
                bev_topk_score, bev_topk_idx = _bev_scores_max.topk(k=topk_bev, dim=-1)

                # 获取 bevformer 的 topk 结果
                _bev_queries_topk = _bev_queries[bev_topk_idx]
                _bev_lines_topk = _bev_lines[bev_topk_idx]
                _bev_gt_targets_topk = _bev_gt_targets[bev_topk_idx]

                # 分别处理 diffusion 部分
                _diff_queries = _queries[n_bevformer_queries:]
                _diff_scores = _scores[n_bevformer_queries:]
                _diff_lines = _lines[n_bevformer_queries:]
                _diff_gt_targets = _gt_targets[n_bevformer_queries:]

                # 对 diffusion 部分取 topk
                _diff_scores_max, _ = _diff_scores.max(-1)
                topk_diff = self.topk_query
                diff_topk_score, diff_topk_idx = _diff_scores_max.topk(k=topk_diff, dim=-1)

                # 获取 diffusion 的 topk 结果
                _diff_queries_topk = _diff_queries[diff_topk_idx]
                _diff_lines_topk = _diff_lines[diff_topk_idx]
                _diff_gt_targets_topk = _diff_gt_targets[diff_topk_idx]

                # 将两部分结果连接起来
                _queries_combined = torch.cat([_bev_queries_topk, _diff_queries_topk], dim=0)
                _lines_combined = torch.cat([_bev_lines_topk, _diff_lines_topk], dim=0)
                _gt_targets_combined = torch.cat([_bev_gt_targets_topk, _diff_gt_targets_topk], dim=0)

                # 将合并结果添加到列表
                query_list.append(_queries_combined)
                ref_pts_list.append(_lines_combined.view(-1, self.num_points, 2))
                gt_targets_list.append(_gt_targets_combined.view(-1, self.num_points, 2))

            # 更新内存
            self.query_memory.update(query_list, img_metas)  # query_memory.memory_list更新为query_list
            self.reference_points_memory.update(ref_pts_list, img_metas)
            self.target_memory.update(gt_targets_list, img_metas)

            loss_dict['trans_loss'] = trans_loss
        return outputs, loss_dict, det_match_idxs, det_match_gt_idxs

    def forward_test(self, bev_features, img_metas):
        '''
        Args:
            bev_feature (List[Tensor]): shape [B, C, H, W]
                feature in bev view
        Outs:
            preds_dict (list[dict]):
                lines (Tensor): Classification score of all
                    decoder layers, has shape
                    [bs, num_query, 2*num_points]
                scores (Tensor):
                    [bs, num_query,]
        '''

        bev_features = self._prepare_context(bev_features)

        bs, C, H, W = bev_features.shape
        img_masks = bev_features.new_zeros((bs, H, W))
        # pos_embed = self.positional_encoding(img_masks)
        pos_embed = None
        attn_mask = (torch.zeros([self.num_queries * 2, self.num_queries * 2,]).bool().to(bev_features.device))

        if self.use_particle_transformer:
            # 为ParticleTransformer准备查询嵌入
            object_query_embeds_bevformer = self.query_embedding_bevformer.weight.to(bev_features.dtype)
            object_query_embeds_diffusion = self.query_embedding_diffusion.weight.to(bev_features.dtype)
            object_query_embeds_bevformer = object_query_embeds_bevformer.repeat(bs, 1, 1)  # torch.Size([4, 100, 1024])
            object_query_embeds_diffusion = object_query_embeds_diffusion.repeat(bs, 1, 1)  # torch.Size([4, 100, 512])

            query_embedding_bevformer, query_pos_bevformer = torch.split(
                object_query_embeds_bevformer, self.embed_dims, dim=2)  # 各自形状: [4, 100, 512]
            object_query_embeds_bevformer = torch.cat([query_embedding_bevformer, query_pos_bevformer],
                                                      dim=1)  # [4, 200, 512]
            query_embedding_combined = torch.cat([query_embedding_bevformer, object_query_embeds_diffusion],
                                                 dim=1)  # query_embedding_combined 形状: [4, 200, 512]
            # 使用ParticleTransformer处理BEV特征
            if self.streaming_query:
                query_embedding_combined, prop_query_embedding, init_reference_points, prop_ref_pts, memory_query, is_first_frame_list = \
                    self.propagate(query_embedding_combined, img_metas, return_loss=False)

            else:
                init_reference_points = self.reference_points_embed(query_embedding_combined).sigmoid()  # (bs, num_q, 2*num_pts)
                init_reference_points = init_reference_points.view(-1, self.num_queries * 2, self.num_points,
                                                                   2)  # (bs, num_q * 2, num_pts, 2)
                prop_query_embedding = None
                prop_ref_pts = None
                is_first_frame_list = [True for i in range(bs)]
            # assert list(init_reference_points.shape) == [bs, self.num_queries * 2, self.num_points, 2]
            # assert list(query_embedding_combined.shape) == [bs, self.num_queries * 2, self.embed_dims]

            # 使用ParticleTransformer处理BEV特征
            outputs = self.transformer(
                mlvl_feats=[bev_features, ],
                mlvl_masks=[img_masks.type(torch.bool)],
                mlvl_pos_embeds=[pos_embed],
                bev_embed=bev_features.flatten(2).permute(0, 2, 1),  # 转换为(B, H*W, C)格式
                object_query_embeds_bevformer=object_query_embeds_bevformer,
                object_query_embeds_diffusion=object_query_embeds_diffusion,
                # query_embed=object_query_embeds_bevformer,
                prop_query=prop_query_embedding,
                prop_reference_points=prop_ref_pts,
                init_reference_points=init_reference_points,  # torch.Size([4, 100, 20, 2])
                bev_h=H,
                bev_w=W,
                grid_length=[self.roi_size[1] / H, self.roi_size[0] / W],
                reg_branches=self.reg_branches,
                cls_branches=self.cls_branches,
                prev_bev=None,
                gt_points=None,
                gt_labels=None,
                pc_range=[self.origin[0], self.origin[1], -10,
                          self.origin[0] + self.roi_size[0],
                          self.origin[1] + self.roi_size[1], 10],
                predict_refine=self.predict_refine,
                is_first_frame_list=is_first_frame_list,
                query_key_padding_mask=prop_query_embedding.new_zeros((bs, self.num_queries), dtype=torch.bool),
                num_bevquery=self.num_queries,
                attn_mask=attn_mask,
            )

            bev_embed, inter_states, init_reference_out, inter_references_out, outputs_classes, outputs_coords = outputs
            # 这里将outputs_class和outputs_coord转换为pred_dict
            outputs = []
            for i, (queries) in enumerate(inter_states):
                reg_points = outputs_coords[i]  # (bs, num_q, num_points, 2)
                bs = reg_points.shape[0]
                reg_points = reg_points.view(bs, -1, 2 * self.num_points)  # (bs, num_q, 2*num_points)

                scores = outputs_classes[i]  # (bs, num_q, num_classes)

                # if self.predict_refine:
                #     tmp = self.reg_branches[i](queries)
                #     reg_points = reg_points + inverse_sigmoid(tmp)  # (新加入的,可能影响初始损失和梯度)reg_points已经是归一化过的了
                reg_points_list = []
                scores_list = []
                prop_mask_list = []
                for i in range(len(scores)):
                    # padding queries should not be output
                    reg_points_list.append(reg_points[i])
                    scores_list.append(scores[i])
                    prop_mask = scores.new_ones((len(scores[i]),), dtype=torch.bool)
                    prop_mask[-self.num_queries:] = False
                    prop_mask_list.append(prop_mask)
                pred_dict = {
                    'lines': reg_points_list,
                    'scores': scores_list,
                    'prop_mask': prop_mask_list
                }
                outputs.append(pred_dict)
            if self.streaming_query:
                query_list = []
                ref_pts_list = []
                lines, scores = outputs[-1]['lines'], outputs[-1]['scores']

                for i in range(bs):
                    _lines = lines[i]
                    _queries = inter_states[-1][i]
                    _scores = scores[i]
                    assert len(_lines) == len(_queries)

                    # 拆分为 bevformer 和 diffusion 两部分
                    n_bevformer_queries = self.num_queries  # 假设是原始的查询数量
                    n_diffusion_queries = len(_queries) - n_bevformer_queries

                    # 分别处理 bevformer 部分
                    _bev_queries = _queries[:n_bevformer_queries]
                    _bev_scores = _scores[:n_bevformer_queries]
                    _bev_lines = _lines[:n_bevformer_queries]

                    # 对 bevformer 部分取 topk
                    _bev_scores_max, _ = _bev_scores.max(-1)
                    topk_bev = self.topk_query
                    bev_topk_score, bev_topk_idx = _bev_scores_max.topk(k=topk_bev, dim=-1)

                    # 获取 bevformer 的 topk 结果
                    _bev_queries_topk = _bev_queries[bev_topk_idx]
                    _bev_lines_topk = _bev_lines[bev_topk_idx]

                    # 分别处理 diffusion 部分
                    _diff_queries = _queries[n_bevformer_queries:]
                    _diff_scores = _scores[n_bevformer_queries:]
                    _diff_lines = _lines[n_bevformer_queries:]

                    # 对 diffusion 部分取 topk
                    _diff_scores_max, _ = _diff_scores.max(-1)
                    topk_diff = self.topk_query
                    diff_topk_score, diff_topk_idx = _diff_scores_max.topk(k=topk_diff, dim=-1)

                    # 获取 diffusion 的 topk 结果
                    _diff_queries_topk = _diff_queries[diff_topk_idx]
                    _diff_lines_topk = _diff_lines[diff_topk_idx]

                    # 将两部分结果连接起来
                    _queries_combined = torch.cat([_bev_queries_topk, _diff_queries_topk], dim=0)
                    _lines_combined = torch.cat([_bev_lines_topk, _diff_lines_topk], dim=0)

                    # 将合并结果添加到列表
                    query_list.append(_queries_combined)
                    ref_pts_list.append(_lines_combined.view(-1, self.num_points, 2))

                self.query_memory.update(query_list, img_metas)
                self.reference_points_memory.update(ref_pts_list, img_metas)

            return outputs

    @force_fp32(apply_to=('score_pred', 'lines_pred', 'gt_lines'))
    def _get_target_single(self,
                           score_pred,
                           lines_pred,
                           gt_labels,
                           gt_lines,
                           gt_bboxes_ignore=None):
        """
            Compute regression and classification targets for one image.
            Outputs from a single decoder layer of a single feature level are used.
            Args:
                score_pred (Tensor): Box score logits from a single decoder layer
                    for one image. Shape [num_query, cls_out_channels].
                lines_pred (Tensor):
                    shape [num_query, 2*num_points]
                gt_labels (torch.LongTensor)
                    shape [num_gt, ]
                gt_lines (Tensor):
                    shape [num_gt, 2*num_points].

            Returns:
                tuple[Tensor]: a tuple containing the following for one sample.
                    - labels (LongTensor): Labels of each image.
                        shape [num_query, 1]
                    - label_weights (Tensor]): Label weights of each image.
                        shape [num_query, 1]
                    - lines_target (Tensor): Lines targets of each image.
                        shape [num_query, num_points, 2]
                    - lines_weights (Tensor): Lines weights of each image.
                        shape [num_query, num_points, 2]
                    - pos_inds (Tensor): Sampled positive indices for each image.
                    - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_pred_lines = len(lines_pred)
        # assigner and sampler
        assign_result, gt_permute_idx = self.assigner.assign(preds=dict(lines=lines_pred, scores=score_pred, ),
                                                             gts=dict(lines=gt_lines,
                                                                      labels=gt_labels, ),
                                                             gt_bboxes_ignore=gt_bboxes_ignore)
        sampling_result = self.sampler.sample(
            assign_result, lines_pred, gt_lines)
        num_gt = len(gt_lines)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_gt_inds = sampling_result.pos_assigned_gt_inds

        labels = gt_lines.new_full(
            (num_pred_lines,), self.num_classes, dtype=torch.long)  # (num_q, )
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_lines.new_ones(num_pred_lines)  # (num_q, )

        lines_target = torch.zeros_like(lines_pred)  # (num_q, 2*num_pts)
        lines_weights = torch.zeros_like(lines_pred)  # (num_q, 2*num_pts)

        if num_gt > 0:
            if gt_permute_idx is not None:  # using permute invariant label
                # gt_permute_idx: (num_q, num_gt)
                # pos_inds: which query is positive
                # pos_gt_inds: which gt each pos pred is assigned
                # single_matched_gt_permute_idx: which permute order is matched
                single_matched_gt_permute_idx = gt_permute_idx[
                    pos_inds, pos_gt_inds
                ]
                lines_target[pos_inds] = gt_lines[pos_gt_inds, single_matched_gt_permute_idx].type(
                    lines_target.dtype)  # (num_q, 2*num_pts)
            else:
                lines_target[pos_inds] = sampling_result.pos_gt_bboxes.type(
                    lines_target.dtype)  # (num_q, 2*num_pts)

        lines_weights[pos_inds] = 1.0  # (num_q, 2*num_pts)

        # normalization
        # n = lines_weights.sum(-1, keepdim=True) # (num_q, 1)
        # lines_weights = lines_weights / n.masked_fill(n == 0, 1) # (num_q, 2*num_pts)
        # [0, ..., 0] for neg ind and [1/npts, ..., 1/npts] for pos ind

        return (labels, label_weights, lines_target, lines_weights,
                pos_inds, neg_inds, pos_gt_inds)

    # @force_fp32(apply_to=('preds', 'gts'))
    def get_targets(self, preds, gts, gt_bboxes_ignore_list=None):
        """
            Compute regression and classification targets for a batch image.
            Outputs from a single decoder layer of a single feature level are used.
            Args:
                preds (dict):
                    - lines (Tensor): shape (bs, num_queries, 2*num_points)
                    - scores (Tensor): shape (bs, num_queries, num_class_channels)
                gts (dict):
                    - class_label (list[Tensor]): tensor shape (num_gts, )
                    - lines (list[Tensor]): tensor shape (num_gts, 2*num_points)
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                tuple: a tuple containing the following targets.
                    - labels_list (list[Tensor]): Labels for all images.
                    - label_weights_list (list[Tensor]): Label weights for all \
                        images.
                    - lines_targets_list (list[Tensor]): Lines targets for all \
                        images.
                    - lines_weight_list (list[Tensor]): Lines weights for all \
                        images.
                    - num_total_pos (int): Number of positive samples in all \
                        images.
                    - num_total_neg (int): Number of negative samples in all \
                        images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        # format the inputs
        gt_labels = gts['labels']
        gt_lines = gts['lines']

        lines_pred = preds['lines']

        (labels_list, label_weights_list,
         lines_targets_list, lines_weights_list,
         pos_inds_list, neg_inds_list, pos_gt_inds_list) = multi_apply(
            self._get_target_single, preds['scores'], lines_pred,
            gt_labels, gt_lines, gt_bboxes_ignore=gt_bboxes_ignore_list)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        new_gts = dict(
            labels=labels_list,  # list[Tensor(num_q, )], length=bs
            label_weights=label_weights_list,  # list[Tensor(num_q, )], length=bs, all ones
            lines=lines_targets_list,  # list[Tensor(num_q, 2*num_pts)], length=bs
            lines_weights=lines_weights_list,  # list[Tensor(num_q, 2*num_pts)], length=bs
        )

        return new_gts, num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list

    # @force_fp32(apply_to=('preds', 'gts'))
    def loss_single(self,
                    preds,
                    gts,
                    gt_bboxes_ignore_list=None,
                    reduction='none'):
        """
            Loss function for outputs from a single decoder layer of a single
            feature level.
            Args:
                preds (dict):
                    - lines (Tensor): shape (bs, num_queries, 2*num_points)
                    - scores (Tensor): shape (bs, num_queries, num_class_channels)
                gts (dict):
                    - class_label (list[Tensor]): tensor shape (num_gts, )
                    - lines (list[Tensor]): tensor shape (num_gts, 2*num_points)
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                dict[str, Tensor]: A dictionary of loss components for outputs from
                    a single decoder layer.
        """

        # Get target for each sample
        new_gts, num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list = \
            self.get_targets(preds, gts, gt_bboxes_ignore_list)

        # Batched all data
        # for k, v in new_gts.items():
        #     new_gts[k] = torch.stack(v, dim=0) # tensor (bs, num_q, ...)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight

        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                preds['scores'][0].new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # Classification loss
        # since the inputs needs the second dim is the class dim, we permute the prediction.
        pred_scores = torch.cat(preds['scores'], dim=0)  # (bs*num_q, cls_out_channles)
        cls_scores = pred_scores.reshape(-1, self.cls_out_channels)  # (bs*num_q, cls_out_channels)
        cls_labels = torch.cat(new_gts['labels'], dim=0).reshape(-1)  # (bs*num_q, )
        cls_weights = torch.cat(new_gts['label_weights'], dim=0).reshape(-1)  # (bs*num_q, )

        loss_cls = self.loss_cls(
            cls_scores, cls_labels, cls_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        pred_lines = torch.cat(preds['lines'], dim=0)
        gt_lines = torch.cat(new_gts['lines'], dim=0)
        line_weights = torch.cat(new_gts['lines_weights'], dim=0)

        assert len(pred_lines) == len(gt_lines)
        assert len(gt_lines) == len(line_weights)

        loss_reg = self.loss_reg(
            pred_lines, gt_lines, line_weights, avg_factor=num_total_pos)

        loss_dict = dict(
            cls=loss_cls,
            reg=loss_reg,
        )

        return loss_dict, pos_inds_list, pos_gt_inds_list, new_gts['lines']

    @force_fp32(apply_to=('gt_lines_list', 'preds_dicts'))
    def loss(self,
             gts,
             preds,
             gt_bboxes_ignore=None,
             diffusion_outputs=None,
             reduction='mean'
             ):
        """
            Loss Function.
            Args:
                gts (list[dict]): list length: num_layers
                    dict {
                        'label': list[tensor(num_gts, )], list length: batchsize,
                        'line': list[tensor(num_gts, 2*num_points)], list length: batchsize,
                        ...
                    }
                preds (list[dict]): list length: num_layers
                    dict {
                        'lines': tensor(bs, num_queries, 2*num_points),
                        'scores': tensor(bs, num_queries, class_out_channels),
                    }

                gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                    which can be ignored for each image. Default None.
            Returns:
                dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        if self.use_particle_transformer:
            # 分别计算BEVFormer loss和Diffusion loss
            n_diffusion_queries = diffusion_outputs['n_diffusion_queries']
            n_bevformer_queries = diffusion_outputs['n_bevformer_queries']

            # 准备BEVFormer部分的预测和目标
            bevformer_preds = []
            for pred_layer in preds:
                bevformer_pred_layer = {
                    'lines': [lines_pred[:n_bevformer_queries] for lines_pred in pred_layer['lines']],
                    'scores': [scores_pred[:n_bevformer_queries] for scores_pred in pred_layer['scores']]
                }
                bevformer_preds.append(bevformer_pred_layer)
            # 创建扩散部分的预测和目标
            diffusion_preds = []
            for pred_layer in preds:
                diffusion_pred_layer = {
                    'lines': [lines_pred[n_bevformer_queries:] for lines_pred in pred_layer['lines']],
                    'scores': [scores_pred[n_bevformer_queries:] for scores_pred in pred_layer['scores']]
                }
                diffusion_preds.append(diffusion_pred_layer)

            # 计算BEVFormer损失
            # bevformer_losses, pos_inds_lists, pos_gt_inds_lists, gt_lines_list = multi_apply(
            #     self.loss_single, bevformer_preds, gts, reduction=reduction)  # pos_inds_lists:list6:list4:torch.Size([num_inds]), pos_gt_inds_lists:同上, gt_lines_list:list6:list4:torch.Size([num_query, num_points * 2])
            bevformer_losses, bev_pos_inds_lists, bev_pos_gt_inds_lists, bev_gt_lines_list = multi_apply(
                self.loss_single, bevformer_preds, gts, reduction=reduction)

            diffusion_losses, diff_pos_inds_lists, diff_pos_gt_inds_lists, diff_gt_lines_list = multi_apply(
                self.loss_single, diffusion_preds, gts, reduction=reduction)

            # 获取BEV查询数量 (用于调整diffusion部分索引)
            # num_bevformer_queries = bevformer_preds[0][0].size(1)  # 假设是第一层第一个batch的查询数量
            num_bevformer_queries = n_bevformer_queries
            # 合并结果
            pos_inds_lists = []
            pos_gt_inds_lists = []
            gt_lines_list = []

            for layer_idx in range(len(bev_pos_inds_lists)):
                layer_pos_inds = []
                layer_pos_gt_inds = []
                layer_gt_lines = []

                for batch_idx in range(len(bev_pos_inds_lists[layer_idx])):
                    # 获取当前层当前批次的结果
                    bev_pos_inds = bev_pos_inds_lists[layer_idx][batch_idx]
                    bev_pos_gt_inds = bev_pos_gt_inds_lists[layer_idx][batch_idx]
                    bev_gt_lines = bev_gt_lines_list[layer_idx][batch_idx]

                    diff_pos_inds = diff_pos_inds_lists[layer_idx][batch_idx]
                    diff_pos_gt_inds = diff_pos_gt_inds_lists[layer_idx][batch_idx]
                    diff_gt_lines = diff_gt_lines_list[layer_idx][batch_idx]

                    # 调整diffusion部分索引（加上BEV查询数量）
                    if diff_pos_inds.numel() > 0:
                        adjusted_diff_pos_inds = diff_pos_inds + num_bevformer_queries
                    else:
                        adjusted_diff_pos_inds = diff_pos_inds

                    # 合并两部分结果
                    combined_pos_inds = torch.cat([bev_pos_inds, adjusted_diff_pos_inds], dim=0)
                    combined_pos_gt_inds = torch.cat([bev_pos_gt_inds, diff_pos_gt_inds], dim=0)
                    combined_gt_lines = torch.cat([bev_gt_lines, diff_gt_lines], dim=0)

                    layer_pos_inds.append(combined_pos_inds)
                    layer_pos_gt_inds.append(combined_pos_gt_inds)
                    layer_gt_lines.append(combined_gt_lines)

                pos_inds_lists.append(layer_pos_inds)
                pos_gt_inds_lists.append(layer_pos_gt_inds)
                gt_lines_list.append(layer_gt_lines)

            # 合并BEVFormer loss和Diffusion loss
            losses = []
            for bevformer_loss, diffusion_loss in zip(bevformer_losses, diffusion_losses):
                combined_loss = {}
                bev_loss = {}
                diff_loss = {}
                for k in bevformer_loss:
                    # combined_loss[k] = 0.5 * (bevformer_loss[k] + diffusion_loss[k])
                    bev_loss['bev'+k] = bevformer_loss[k] * 0.5
                    diff_loss['diff'+k] = diffusion_loss[k] * 0.5
                losses.append(bev_loss)
                losses.append(diff_loss)
                # losses.append(combined_loss)

            # # 添加噪声预测损失
            # if diffusion_noise_loss > 0:
            #     losses[-1]['diffusion_noise'] = self.diffusion_loss_weight * diffusion_noise_loss

        else:
            # 原始的损失计算
            losses, pos_inds_lists, pos_gt_inds_lists, gt_lines_list = multi_apply(
                self.loss_single, preds, gts, reduction=reduction)

        # Format the losses
        loss_dict = dict()
        # loss from the last decoder layer
        for k, v in losses[-1].items():
            loss_dict[k] = v

        # Loss from other decoder layers
        num_dec_layer = 0
        for loss in losses[:-1]:
            for k, v in loss.items():
                loss_dict[f'd{num_dec_layer}.{k}'] = v
            num_dec_layer += 1

        return loss_dict, pos_inds_lists, pos_gt_inds_lists, gt_lines_list

    def post_process(self, preds_dict, tokens, thr=0.0):
        lines = preds_dict['lines']  # List[Tensor(num_queries, 2*num_points)]
        bs = len(lines)
        scores = preds_dict['scores']  # (bs, num_queries, 3)
        prop_mask = preds_dict['prop_mask']

        results = []
        for i in range(bs):
            tmp_vectors = lines[i]
            tmp_prop_mask = prop_mask[i]
            num_preds, num_points2 = tmp_vectors.shape
            tmp_vectors = tmp_vectors.view(num_preds, num_points2 // 2, 2)
            # focal loss
            if self.loss_cls.use_sigmoid:
                tmp_scores, tmp_labels = scores[i].max(-1)
                tmp_scores = tmp_scores.sigmoid()
                pos = tmp_scores > thr
            else:
                assert self.num_classes + 1 == self.cls_out_channels
                tmp_scores, tmp_labels = scores[i].max(-1)
                bg_cls = self.cls_out_channels
                pos = tmp_labels != bg_cls

            tmp_vectors = tmp_vectors[pos]
            tmp_scores = tmp_scores[pos]
            tmp_labels = tmp_labels[pos]
            tmp_prop_mask = tmp_prop_mask[pos]

            if len(tmp_scores) == 0:
                single_result = {
                    'vectors': [],
                    'scores': [],
                    'labels': [],
                    'prop_mask': [],
                    'token': tokens[i]
                }
            else:
                single_result = {
                    'vectors': tmp_vectors.detach().cpu().numpy(),
                    'scores': tmp_scores.detach().cpu().numpy(),
                    'labels': tmp_labels.detach().cpu().numpy(),
                    'prop_mask': tmp_prop_mask.detach().cpu().numpy(),
                    'token': tokens[i]
                }
            results.append(single_result)

        return results

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        for k, v in self.__dict__.items():
            if isinstance(v, StreamTensorMemory):
                v.train(*args, **kwargs)

    def eval(self):
        super().eval()
        for k, v in self.__dict__.items():
            if isinstance(v, StreamTensorMemory):
                v.eval()

    def forward(self, *args, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)