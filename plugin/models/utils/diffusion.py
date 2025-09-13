import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import Linear

class DiffusionMapRefiner(nn.Module):
    def __init__(self, embed_dims, num_points, time_steps=1000, beta_schedule='linear'):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_points = num_points
        self.time_steps = time_steps
        
        # 设置beta调度
        if beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, time_steps)
        elif beta_schedule == 'cosine':
            steps = time_steps + 1
            x = torch.linspace(0, time_steps, steps)
            alphas_cumprod = torch.cos(((x / time_steps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        
        # 计算扩散过程需要的常量
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1. / self.alphas))
        self.register_buffer('posterior_variance', self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        
        # 时间嵌入网络
        self.time_embed = nn.Sequential(
            Linear(1, embed_dims),
            nn.SiLU(),
            Linear(embed_dims, embed_dims)
        )
        
        # 去噪网络 - 基于点特征和时间步长的条件
        self.refiner = nn.Sequential(
            Linear(embed_dims + num_points*2 + embed_dims, embed_dims*2),  # 特征+当前点+时间嵌入
            nn.SiLU(),
            Linear(embed_dims*2, embed_dims*2),
            nn.SiLU(),
            Linear(embed_dims*2, num_points*2)
        )
    
    def _extract(self, a, t, x_shape):
        """从预计算的常量中提取适合当前批次的值"""
        batch_size = t.shape[0]
        out = a.gather(-1, t).to(t.device)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(self, x_start, t, noise=None):
        """向x_start添加t时刻的噪声"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """从噪声预测原始信号"""
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        return sqrt_recip_alphas_t * x_t - sqrt_one_minus_alphas_cumprod_t * noise
    
    def _diffusion_training(self, queries, lines, gt_lines):
        """扩散模型训练过程
        
        Args:
            queries: 查询特征 [batch_size, num_queries, embed_dims]
            lines: 预测的线段 [batch_size, num_queries, num_points, 2]
            gt_lines: 真实的线段 [batch_size, num_queries, num_points, 2]
            
        Returns:
            diffusion_loss: 扩散模型的损失
            refined_lines: 经过扩散优化的线段
        """
        batch_size, num_queries = lines.shape[:2]
        device = lines.device
        
        # 展平处理
        lines_flat = lines.reshape(batch_size * num_queries, -1)  # [batch*queries, num_points*2]
        gt_lines_flat = gt_lines.reshape(batch_size * num_queries, -1)  # [batch*queries, num_points*2]
        queries_flat = queries.reshape(batch_size * num_queries, -1)  # [batch*queries, embed_dims]
        
        # 随机采样时间步
        t = torch.randint(0, self.time_steps, (batch_size * num_queries,), device=device).long()
        
        # 添加噪声到真实线段
        noise = torch.randn_like(gt_lines_flat)
        x_noisy = self.q_sample(gt_lines_flat, t, noise=noise)
        
        # 时间嵌入
        t_emb = self.time_embed(t.unsqueeze(-1).float() / self.time_steps)
        
        # 条件输入 (查询特征 + 带噪线段 + 时间嵌入)
        cond_input = torch.cat([queries_flat, x_noisy, t_emb], dim=1)
        
        # 预测噪声
        pred_noise = self.refiner(cond_input)
        
        # 计算简单MSE损失
        diffusion_loss = F.mse_loss(pred_noise, noise)
        
        # 预测优化后的线段 (可选)
        with torch.no_grad():
            pred_lines = self.predict_start_from_noise(x_noisy, t, pred_noise)
            refined_lines = pred_lines.reshape(batch_size, num_queries, self.num_points, 2)
        
        return diffusion_loss, refined_lines
    
    def _diffusion_sampling(self, queries, lines, num_inference_steps=50):
        """扩散模型推理采样过程
        
        Args:
            queries: 查询特征 [batch_size, num_queries, embed_dims]
            lines: 初始预测的线段 [batch_size, num_queries, num_points, 2]
            num_inference_steps: 推理步数
            
        Returns:
            refined_lines: 经过扩散优化的线段
        """
        batch_size, num_queries = lines.shape[:2]
        device = lines.device
        
        # 展平处理
        lines_flat = lines.reshape(batch_size * num_queries, -1)  # [batch*queries, num_points*2]
        queries_flat = queries.reshape(batch_size * num_queries, -1)  # [batch*queries, embed_dims]
        
        # 初始化纯噪声或使用初始预测加噪声
        x = lines_flat + 0.1 * torch.randn_like(lines_flat)  # 从模型预测的线段开始，加少量噪声
        
        # 时间步采样
        timesteps = np.linspace(self.time_steps-1, 0, num_inference_steps).astype(np.int64)
        
        # 逐步去噪
        for i, t in enumerate(timesteps):
            # 创建时间张量
            t_batch = torch.full((batch_size * num_queries,), t, device=device, dtype=torch.long)
            
            # 时间嵌入
            t_emb = self.time_embed(t_batch.unsqueeze(-1).float() / self.time_steps)
            
            # 条件输入
            cond_input = torch.cat([queries_flat, x, t_emb], dim=1)
            
            # 预测噪声
            pred_noise = self.refiner(cond_input)
            
            # 预测x_0
            pred_x_0 = self.predict_start_from_noise(x, t_batch, pred_noise)
            
            # 应用更新步骤
            alpha_t = self._extract(self.alphas, t_batch, x.shape)
            posterior_variance_t = self._extract(self.posterior_variance, t_batch, x.shape)
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (pred_x_0 * torch.sqrt(alpha_t) + torch.sqrt(posterior_variance_t) * noise)
            
        # 重塑为原始形状
        refined_lines = x.reshape(batch_size, num_queries, self.num_points, 2)
        
        return refined_lines
    
    def forward(self, queries, lines, gt_lines=None, training=True, layer_idx=None):
        """添加layer_idx参数允许为不同层设置不同参数"""
        
        # 可以根据layer_idx调整扩散参数
        inference_steps = 50
        if layer_idx is not None:
            # 为前面层设置较少的推理步数以加速训练
            inference_steps = max(10, 50 - 10 * layer_idx)
        
        if training and gt_lines is not None:
            return self._diffusion_training(queries, lines, gt_lines)
        else:
            return self._diffusion_sampling(queries, lines, num_inference_steps=inference_steps)