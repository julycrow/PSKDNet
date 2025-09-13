# !/usr/bin/env python3
"""
真实地图检测模型的扩散噪声影响分析工具

本工具用于分析真实的地图检测模型中扩散噪声对位置依赖性的影响
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2
from typing import Dict, List, Tuple, Optional
import json
import os
import sys

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class RealModelNoiseAnalyzer:
    """真实模型的噪声影响分析器"""

    def __init__(self, model_path: str = None, save_dir: str = "./real_model_analysis"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载预训练模型"""
        try:
            self.model = torch.load(model_path, map_location='cpu')
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")

    def add_noise_to_features(self, features: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
        """向特征图添加噪声"""
        noise = torch.randn_like(features) * noise_level
        return features + noise

    def add_noise_to_queries(self, queries: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
        """向查询向量添加噪声"""
        noise = torch.randn_like(queries) * noise_level
        return queries + noise

    def extract_map_elements_from_prediction(self, prediction: Dict) -> Dict:
        """从模型预测中提取地图元素"""
        map_elements = {
            'lane_lines': [],
            'road_edges': [],
            'crosswalks': []
        }

        if 'lines' in prediction:
            lines = prediction['lines']
            scores = prediction.get('scores', None)

            # 假设lines的形状为 (batch_size, num_queries, num_points*2)
            if isinstance(lines, torch.Tensor):
                lines = lines.detach().cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()

            # 处理每个batch
            for batch_idx in range(len(lines)):
                batch_lines = lines[batch_idx]
                batch_scores = scores[batch_idx] if scores is not None else None

                for query_idx, line_coords in enumerate(batch_lines):
                    # 检查分数阈值
                    if batch_scores is not None:
                        max_score = np.max(batch_scores[query_idx])
                        if max_score < 0.3:  # 分数阈值
                            continue

                    # 重塑为点坐标
                    num_points = len(line_coords) // 2
                    points = line_coords.reshape(num_points, 2)

                    # 根据某种逻辑分类为不同类型的地图元素
                    # 这里简化处理，实际需要根据模型的具体输出格式调整
                    map_elements['lane_lines'].append(points)

        return map_elements

    def analyze_feature_noise_impact(self, sample_data: Dict, noise_levels: List[float] = None):
        """分析特征层面噪声的影响"""
        if noise_levels is None:
            noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

        if self.model is None:
            print("请先加载模型!")
            return

        # 模拟输入数据
        batch_size = 1
        num_cameras = 6
        height, width = 480, 800
        feature_dim = 256

        # 创建模拟的BEV特征
        bev_features = torch.randn(batch_size, feature_dim, 100, 100)

        results = {}
        original_prediction = None

        for noise_level in noise_levels:
            print(f"分析噪声水平: {noise_level}")

            # 添加噪声
            noisy_features = self.add_noise_to_features(bev_features, noise_level)

            # 模拟模型推理（这里需要根据实际模型调整）
            with torch.no_grad():
                try:
                    # 这里需要根据您的具体模型接口调整
                    # prediction = self.model(noisy_features, sample_data['img_metas'])

                    # 模拟预测结果
                    num_queries = 100
                    num_points = 20

                    # 生成模拟的预测结果
                    lines = torch.randn(batch_size, num_queries, num_points * 2)
                    scores = torch.sigmoid(torch.randn(batch_size, num_queries, 3))  # 3类地图元素

                    # 添加噪声影响
                    if noise_level > 0:
                        lines += torch.randn_like(lines) * noise_level * 0.5

                    prediction = {
                        'lines': lines,
                        'scores': scores
                    }

                    if noise_level == 0.0:
                        original_prediction = prediction

                    results[noise_level] = prediction

                except Exception as e:
                    print(f"模型推理失败: {e}")
                    continue

        return results, original_prediction

    def visualize_noise_impact_on_real_predictions(self, results: Dict, original_prediction: Dict):
        """可视化噪声对真实预测的影响"""
        noise_levels = list(results.keys())

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, noise_level in enumerate(noise_levels):
            if idx >= len(axes):
                break

            ax = axes[idx]
            prediction = results[noise_level]

            # 提取地图元素
            map_elements = self.extract_map_elements_from_prediction(prediction)

            # 绘制车道线
            colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
            for i, lane_points in enumerate(map_elements['lane_lines'][:6]):  # 最多显示6条
                if len(lane_points) > 1:
                    ax.plot(lane_points[:, 0], lane_points[:, 1],
                            color=colors[i % len(colors)], alpha=0.7, linewidth=2,
                            label=f'车道{i + 1}' if idx == 0 else "")

            # 如果是原始预测，也绘制参考线
            if noise_level == 0.0:
                for i, lane_points in enumerate(map_elements['lane_lines'][:6]):
                    if len(lane_points) > 1:
                        ax.plot(lane_points[:, 0], lane_points[:, 1],
                                'k--', alpha=0.3, linewidth=1)

            ax.set_title(f'噪声水平: {noise_level:.2f}', fontsize=12)
            ax.set_xlabel('X坐标')
            ax.set_ylabel('Y坐标')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            if idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/real_model_noise_impact.png', dpi=300, bbox_inches='tight')
        plt.show()

    def calculate_geometric_degradation(self, original_prediction: Dict, noisy_predictions: Dict) -> Dict:
        """计算几何结构的退化程度"""
        degradation_metrics = {
            'noise_levels': [],
            'position_drift': [],
            'shape_distortion': [],
            'topology_violations': []
        }

        # 提取原始地图元素
        original_elements = self.extract_map_elements_from_prediction(original_prediction)

        for noise_level, prediction in noisy_predictions.items():
            if noise_level == 0.0:
                continue

            degradation_metrics['noise_levels'].append(noise_level)

            # 提取噪声版本的地图元素
            noisy_elements = self.extract_map_elements_from_prediction(prediction)

            # 计算位置漂移
            position_drifts = []
            for orig_lane, noisy_lane in zip(original_elements['lane_lines'],
                                             noisy_elements['lane_lines']):
                if len(orig_lane) == len(noisy_lane):
                    drift = np.mean(np.sqrt(np.sum((orig_lane - noisy_lane) ** 2, axis=1)))
                    position_drifts.append(drift)

            avg_drift = np.mean(position_drifts) if position_drifts else 0
            degradation_metrics['position_drift'].append(avg_drift)

            # 计算形状扭曲（基于曲率变化）
            shape_distortions = []
            for orig_lane, noisy_lane in zip(original_elements['lane_lines'],
                                             noisy_elements['lane_lines']):
                if len(orig_lane) > 2 and len(noisy_lane) > 2:
                    # 计算曲率
                    orig_curvature = self.calculate_curvature(orig_lane)
                    noisy_curvature = self.calculate_curvature(noisy_lane)

                    if len(orig_curvature) == len(noisy_curvature):
                        distortion = np.mean(np.abs(orig_curvature - noisy_curvature))
                        shape_distortions.append(distortion)

            avg_distortion = np.mean(shape_distortions) if shape_distortions else 0
            degradation_metrics['shape_distortion'].append(avg_distortion)

            # 计算拓扑违规（平行线交叉等）
            violations = self.detect_topology_violations(noisy_elements['lane_lines'])
            degradation_metrics['topology_violations'].append(violations)

        return degradation_metrics

    def calculate_curvature(self, points: np.ndarray) -> np.ndarray:
        """计算路径的曲率"""
        if len(points) < 3:
            return np.array([])

        # 计算一阶和二阶导数
        dx = np.gradient(points[:, 0])
        dy = np.gradient(points[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # 计算曲率
        curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)
        return curvature

    def detect_topology_violations(self, lane_lines: List[np.ndarray]) -> int:
        """检测拓扑违规（如平行线交叉）"""
        violations = 0

        for i in range(len(lane_lines)):
            for j in range(i + 1, len(lane_lines)):
                lane1 = lane_lines[i]
                lane2 = lane_lines[j]

                if len(lane1) > 1 and len(lane2) > 1:
                    # 检查是否有交叉
                    intersections = self.find_line_intersections(lane1, lane2)
                    violations += len(intersections)

        return violations

    def find_line_intersections(self, line1: np.ndarray, line2: np.ndarray) -> List[Tuple]:
        """找到两条线的交点"""
        intersections = []

        for i in range(len(line1) - 1):
            for j in range(len(line2) - 1):
                p1, p2 = line1[i], line1[i + 1]
                p3, p4 = line2[j], line2[j + 1]

                # 计算线段交点
                intersection = self.line_segment_intersection(p1, p2, p3, p4)
                if intersection is not None:
                    intersections.append(intersection)

        return intersections

    def line_segment_intersection(self, p1: np.ndarray, p2: np.ndarray,
                                  p3: np.ndarray, p4: np.ndarray) -> Optional[Tuple]:
        """计算两个线段的交点"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # 平行线

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)

        return None

    def plot_degradation_metrics(self, degradation_metrics: Dict):
        """绘制几何退化指标"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        noise_levels = degradation_metrics['noise_levels']

        # 位置漂移
        axes[0, 0].plot(noise_levels, degradation_metrics['position_drift'],
                        'ro-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('噪声水平')
        axes[0, 0].set_ylabel('平均位置漂移')
        axes[0, 0].set_title('位置依赖性的破坏')
        axes[0, 0].grid(True, alpha=0.3)

        # 形状扭曲
        axes[0, 1].plot(noise_levels, degradation_metrics['shape_distortion'],
                        'go-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('噪声水平')
        axes[0, 1].set_ylabel('平均形状扭曲')
        axes[0, 1].set_title('几何形状的扭曲')
        axes[0, 1].grid(True, alpha=0.3)

        # 拓扑违规
        axes[1, 0].plot(noise_levels, degradation_metrics['topology_violations'],
                        'bo-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('噪声水平')
        axes[1, 0].set_ylabel('拓扑违规数量')
        axes[1, 0].set_title('拓扑结构的破坏')
        axes[1, 0].grid(True, alpha=0.3)

        # 综合退化指标
        # 归一化各指标并计算综合分数
        max_drift = max(degradation_metrics['position_drift']) if degradation_metrics['position_drift'] else 1
        max_distortion = max(degradation_metrics['shape_distortion']) if degradation_metrics['shape_distortion'] else 1
        max_violations = max(degradation_metrics['topology_violations']) if degradation_metrics[
            'topology_violations'] else 1

        if max_drift > 0 and max_distortion > 0:
            normalized_drift = np.array(degradation_metrics['position_drift']) / max_drift
            normalized_distortion = np.array(degradation_metrics['shape_distortion']) / max_distortion
            normalized_violations = np.array(degradation_metrics['topology_violations']) / max(max_violations, 1)

            combined_score = (normalized_drift + normalized_distortion + normalized_violations) / 3

            axes[1, 1].plot(noise_levels, combined_score, 'mo-', linewidth=2, markersize=8)
            axes[1, 1].set_xlabel('噪声水平')
            axes[1, 1].set_ylabel('综合退化分数')
            axes[1, 1].set_title('几何结构综合退化')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/degradation_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()


def create_analysis_script():
    """创建分析脚本"""
    script_content = '''#!/usr/bin/env python3
"""
运行真实模型噪声分析的脚本
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from real_model_noise_analysis import RealModelNoiseAnalyzer

def main():
    # 创建分析器
    analyzer = RealModelNoiseAnalyzer()

    # 模拟样本数据
    sample_data = {
        'img_metas': [{'img_shape': (480, 800), 'pad_shape': (480, 800)}]
    }

    print("开始分析特征噪声对模型预测的影响...")

    # 分析特征噪声影响
    results, original_prediction = analyzer.analyze_feature_noise_impact(sample_data)

    if results and original_prediction:
        # 可视化噪声影响
        print("可视化噪声影响...")
        analyzer.visualize_noise_impact_on_real_predictions(results, original_prediction)

        # 计算几何退化
        print("计算几何退化指标...")
        degradation_metrics = analyzer.calculate_geometric_degradation(
            original_prediction, results
        )

        # 绘制退化指标
        analyzer.plot_degradation_metrics(degradation_metrics)

        print(f"分析完成！结果保存在: {analyzer.save_dir}")
    else:
        print("分析失败，请检查模型加载和数据格式")

if __name__ == "__main__":
    main()
'''

    with open('run_real_model_analysis.py', 'w', encoding='utf-8') as f:
        f.write(script_content)

    print("已创建分析脚本: run_real_model_analysis.py")


if __name__ == "__main__":
    create_analysis_script()

