#!/usr/bin/env python3
"""
Model Comparison Visualizer

This script compares visualization results from two different models.
For each common scene, it creates a comparison image showing:
- Each row: one frame
- Each column: Model1 pred, Model2 pred, GT
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import argparse
from pathlib import Path
from typing import List, Tuple, Dict


class ModelComparisonVisualizer:
    def __init__(self, model1_dir: str, model2_dir: str, output_dir: str = "./comparison_results"):
        self.model1_dir = Path(model1_dir)
        self.model2_dir = Path(model2_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract model names from directory paths
        self.model1_name = self.model1_dir.name
        self.model2_name = self.model2_dir.name
        
        print(f"模型1: {self.model1_name}")
        print(f"模型2: {self.model2_name}")
        print(f"输出目录: {self.output_dir}")
    
    def find_common_scenes(self) -> List[str]:
        """找到两个模型都有的公共场景"""
        model1_scenes = set([d.name for d in self.model1_dir.iterdir() if d.is_dir()])
        model2_scenes = set([d.name for d in self.model2_dir.iterdir() if d.is_dir()])
        
        common_scenes = sorted(list(model1_scenes & model2_scenes))
        print(f"找到 {len(common_scenes)} 个公共场景: {common_scenes}")
        return common_scenes
    
    def find_common_frames(self, scene: str) -> List[str]:
        """找到指定场景中两个模型都有的公共帧"""
        scene1_dir = self.model1_dir / scene
        scene2_dir = self.model2_dir / scene
        
        if not scene1_dir.exists() or not scene2_dir.exists():
            return []
        
        frames1 = set([d.name for d in scene1_dir.iterdir() if d.is_dir()])
        frames2 = set([d.name for d in scene2_dir.iterdir() if d.is_dir()])
        
        common_frames = sorted(list(frames1 & frames2), key=lambda x: int(x))
        return common_frames
    
    def load_image(self, image_path: Path) -> np.ndarray:
        """加载图像并转换为RGB格式"""
        if not image_path.exists():
            # 如果图像不存在，创建一个占位符图像
            placeholder = np.ones((400, 600, 3), dtype=np.uint8) * 128
            cv2.putText(placeholder, "Image Not Found", (150, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return placeholder
        
        image = cv2.imread(str(image_path))
        if image is None:
            # 如果无法读取图像，创建占位符
            placeholder = np.ones((400, 600, 3), dtype=np.uint8) * 128
            cv2.putText(placeholder, "Invalid Image", (150, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return placeholder
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """调整图像大小"""
        return cv2.resize(image, target_size)
    
    def add_border_to_image(self, image: np.ndarray, border_width: int = 2, border_color: tuple = (128, 128, 128)) -> np.ndarray:
        """为图像添加边框"""
        h, w = image.shape[:2]
        
        # 创建带边框的图像
        bordered_image = np.full((h + 2*border_width, w + 2*border_width, 3), border_color, dtype=np.uint8)
        
        # 将原图像放置在中心
        bordered_image[border_width:h+border_width, border_width:w+border_width] = image
        
        return bordered_image
    
    def add_text_label(self, image: np.ndarray, text: str, position: str = "top") -> np.ndarray:
        """在图像上添加文本标签"""
        h, w = image.shape[:2]
        # 创建带标签的新图像
        label_height = 40
        if position == "top":
            new_image = np.ones((h + label_height, w, 3), dtype=np.uint8) * 255
            new_image[label_height:, :] = image
            text_y = 25
        else:  # bottom
            new_image = np.ones((h + label_height, w, 3), dtype=np.uint8) * 255
            new_image[:h, :] = image
            text_y = h + 25
        
        # 使用PIL添加文本（支持更好的字体渲染）
        pil_image = Image.fromarray(new_image)
        draw = ImageDraw.Draw(pil_image)
        
        # 尝试使用更好的字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # 计算文本位置（居中）
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (w - text_width) // 2
        
        draw.text((text_x, text_y - 15), text, fill=(0, 0, 0), font=font)
        
        return np.array(pil_image)
    
    def create_scene_comparison(self, scene: str) -> None:
        """为单个场景创建对比图像"""
        print(f"\n处理场景: {scene}")
        
        common_frames = self.find_common_frames(scene)
        if not common_frames:
            print(f"场景 {scene} 没有公共帧，跳过")
            return
        
        print(f"找到 {len(common_frames)} 个公共帧")
        
        # 准备存储所有行的图像
        comparison_rows = []
        
        # 设置目标图像大小
        target_width, target_height = 600, 400
        
        for frame in common_frames:
            # 构建图像路径
            model1_pred_path = self.model1_dir / scene / frame / "pred" / "map.jpg"
            model2_pred_path = self.model2_dir / scene / frame / "pred" / "map.jpg"
            gt_path = self.model1_dir / scene / frame / "gt" / "map.jpg"  # GT在两个模型中应该相同
            
            # 加载图像
            model1_pred = self.load_image(model1_pred_path)
            model2_pred = self.load_image(model2_pred_path)
            gt = self.load_image(gt_path)
            
            # 调整图像大小
            model1_pred = self.resize_image(model1_pred, (target_width, target_height))
            model2_pred = self.resize_image(model2_pred, (target_width, target_height))
            gt = self.resize_image(gt, (target_width, target_height))
            
            # 为每个图像添加细边框
            model1_pred = self.add_border_to_image(model1_pred, border_width=1, border_color=(100, 100, 100))
            model2_pred = self.add_border_to_image(model2_pred, border_width=1, border_color=(100, 100, 100))
            gt = self.add_border_to_image(gt, border_width=1, border_color=(100, 100, 100))
            
            # 水平拼接三张图像
            row_image = np.hstack([model1_pred, model2_pred, gt])
            comparison_rows.append(row_image)
        
        if not comparison_rows:
            print(f"场景 {scene} 没有有效的图像，跳过")
            return
        
        # 垂直拼接所有行
        scene_comparison = np.vstack(comparison_rows)
        
        # 添加列标题（考虑边框的影响）
        header_height = 50
        border_width = 1
        total_width = (target_width + 2 * border_width) * 3  # 每个图像都有边框
        header = np.ones((header_height, total_width, 3), dtype=np.uint8) * 240
        
        # 使用PIL添加标题
        pil_header = Image.fromarray(header)
        draw = ImageDraw.Draw(pil_header)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # 添加列标题
        titles = [self.model1_name, self.model2_name, "Ground Truth"]
        for i, title in enumerate(titles):
            bbox = draw.textbbox((0, 0), title, font=font)
            text_width = bbox[2] - bbox[0]
            # 考虑边框的影响计算文本位置
            column_width = target_width + 2 * border_width
            text_x = i * column_width + (column_width - text_width) // 2
            draw.text((text_x, 15), title, fill=(0, 0, 0), font=font)
        
        header_array = np.array(pil_header)
        
        # 组合标题和内容
        final_image = np.vstack([header_array, scene_comparison])
        
        # 添加帧编号（在每行左侧）
        final_height, final_width = final_image.shape[:2]
        margin_width = 50
        final_with_margin = np.ones((final_height, final_width + margin_width, 3), dtype=np.uint8) * 255
        final_with_margin[:, margin_width:] = final_image
        
        # 添加帧编号
        pil_final = Image.fromarray(final_with_margin)
        draw = ImageDraw.Draw(pil_final)
        
        try:
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font_small = ImageFont.load_default()
        
        for i, frame in enumerate(common_frames):
            # 考虑边框影响的行高度
            row_height = target_height + 2 * border_width
            y_pos = header_height + i * row_height + row_height // 2
            draw.text((5, y_pos), f"Frame\n{frame}", fill=(0, 0, 0), font=font_small)
        
        # 保存最终图像
        output_path = self.output_dir / f"{scene}_comparison.jpg"
        final_array = np.array(pil_final)
        final_bgr = cv2.cvtColor(final_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), final_bgr)
        
        print(f"✅ 场景 {scene} 对比图已保存: {output_path}")
        print(f"   包含 {len(common_frames)} 帧，图像大小: {final_array.shape}")
    
    def create_all_comparisons(self) -> None:
        """为所有公共场景创建对比图像"""
        common_scenes = self.find_common_scenes()
        
        if not common_scenes:
            print("没有找到公共场景！")
            return
        
        print(f"\n开始处理 {len(common_scenes)} 个场景...")
        
        for scene in common_scenes:
            try:
                self.create_scene_comparison(scene)
            except Exception as e:
                print(f"❌ 处理场景 {scene} 时出错: {e}")
                continue
        
        print(f"\n🎉 完成！所有对比图已保存到: {self.output_dir}")
    
    def create_summary_report(self) -> None:
        """创建总结报告"""
        common_scenes = self.find_common_scenes()
        
        report_path = self.output_dir / "comparison_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("模型对比可视化报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"模型1: {self.model1_name}\n")
            f.write(f"模型2: {self.model2_name}\n\n")
            f.write(f"公共场景数量: {len(common_scenes)}\n")
            f.write(f"公共场景列表: {', '.join(common_scenes)}\n\n")
            
            for scene in common_scenes:
                frames = self.find_common_frames(scene)
                f.write(f"场景 {scene}: {len(frames)} 帧\n")
            
            f.write(f"\n输出文件保存在: {self.output_dir}\n")
        
        print(f"📋 报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='创建两个模型的可视化结果对比')
    parser.add_argument('--model1-dir', 
                        default='/root/disk4/jmk3/Project/PSKDNet/visualize_result/nusc_baseline_480_60x30_30e_gpu8_0.6335',
                        help='模型1的可视化结果目录')
    parser.add_argument('--model2-dir',
                        default='/root/disk4/jmk3/Project/PSKDNet/visualize_result/nusc_baseline_480_60x30_30e_add20_0.6799',
                        help='模型2的可视化结果目录')
    parser.add_argument('--output-dir',
                        default='./model_comparison_results',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.model1_dir):
        print(f"❌ 模型1目录不存在: {args.model1_dir}")
        return
    
    if not os.path.exists(args.model2_dir):
        print(f"❌ 模型2目录不存在: {args.model2_dir}")
        return
    
    print("🚀 开始模型对比可视化...")
    print("=" * 50)
    
    # 创建对比器
    visualizer = ModelComparisonVisualizer(
        model1_dir=args.model1_dir,
        model2_dir=args.model2_dir,
        output_dir=args.output_dir
    )
    
    # 生成所有对比图
    visualizer.create_all_comparisons()
    
    # 生成总结报告
    visualizer.create_summary_report()


if __name__ == "__main__":
    main()
