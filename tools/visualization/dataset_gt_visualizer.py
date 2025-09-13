#!/usr/bin/env python3
"""
Dataset GT Visualizer

This tool extracts and visualizes ground truth data from datasets without grids, axes, or titles.
Extracted from relation_improvement_analysis.py and focused on clean GT visualization.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import warnings
import argparse
import mmcv
from mmcv import Config
import os
import sys
import random

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Add project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import original analysis tools
from diffusion_noise_analysis import DiffusionNoiseAnalyzer, import_plugin

# Set font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Configure matplotlib for vectorized PDF output
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts
plt.rcParams['ps.fonttype'] = 42   # TrueType fonts


class DatasetGTVisualizer:
    """Dataset Ground Truth Visualizer"""
    
    def __init__(self, config_path: str, save_dir: str = "./dataset_gt_visualizations"):
        """
        Initialize the visualizer
        
        Args:
            config_path: Path to the dataset configuration file
            save_dir: Directory to save visualization results
        """
        self.config_path = config_path
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize noise analyzer for dataset loading
        self.noise_analyzer = DiffusionNoiseAnalyzer(save_dir=save_dir, config_path=config_path)
        
        # Load dataset
        self.load_dataset()
        
        print(f"🎯 Dataset GT Visualizer initialized")
        print(f"📁 Save directory: {save_dir}")
        print(f"⚙️ Config path: {config_path}")
    
    def load_dataset(self):
        """Load dataset"""
        self.noise_analyzer.load_dataset()
        self.dataset = self.noise_analyzer.dataset
        
        if self.dataset is not None:
            print(f"✅ Dataset loaded successfully with {len(self.dataset)} samples")
        else:
            print("❌ Failed to load dataset")
    
    def extract_gt_map_data(self, sample_idx: int = 0) -> Dict:
        """Extract GT map data from dataset"""
        return self.noise_analyzer.extract_gt_map_data(sample_idx)
    
    def visualize_gt_clean(self, map_data: Dict, sample_idx: int = 0, 
                          image_format: str = 'pdf', figsize: Tuple[int, int] = (10, 8)):
        """
        Visualize GT data with clean appearance (no grids, axes, titles)
        
        Args:
            map_data: Dictionary containing GT map data
            sample_idx: Sample index for filename
            image_format: Output format ('pdf', 'png', 'svg')
            figsize: Figure size as (width, height)
        """
        # Create figure with clean appearance
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Calculate coordinate ranges for consistent scaling
        all_points = []
        for lane_points in map_data['lane_lines']:
            all_points.extend(lane_points)
        for edge_points in map_data['road_edges']:
            all_points.extend(edge_points)
        for crosswalk_points in map_data['crosswalks']:
            all_points.extend(crosswalk_points)
        
        if all_points:
            all_points = np.array(all_points)
            x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
            y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
            x_margin = (x_max - x_min) * 0.05
            y_margin = (y_max - y_min) * 0.05
            x_lim = [x_min - x_margin, x_max + x_margin]
            y_lim = [y_min - y_margin, y_max + y_margin]
        else:
            x_lim = [0, 80]
            y_lim = [0, 70]
        
        # Draw map elements with clean styling
        self._draw_map_elements_clean(ax, map_data)
        
        # Set limits and clean appearance
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        
        # Remove all axes, grids, and labels for clean output
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Set white background
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Save with clean filename
        filename = f'{self.save_dir}/gt_sample_{sample_idx:04d}.{image_format}'
        
        # Save with appropriate parameters for vectorized output
        if image_format == 'pdf':
            plt.savefig(filename, format=image_format, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', pad_inches=0.1)
        elif image_format == 'svg':
            plt.savefig(filename, format=image_format, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', pad_inches=0.1)
        else:
            plt.savefig(filename, format=image_format, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', pad_inches=0.1)
        
        plt.close()  # Close to save memory
        
        print(f"✅ GT visualization saved: {filename}")
        return filename
    
    def _draw_map_elements_clean(self, ax, map_data: Dict, overlay_elements: List[np.ndarray] = None):
        """Draw map elements with clean styling"""
        
        # Define colors and styles for GT elements (lighter colors)
        lane_color, lane_alpha, lane_style = '#FFAAAA', 0.7, '-'  # Light red
        edge_color, edge_alpha, edge_style = '#AAFFAA', 0.7, '-'  # Light green
        cross_color, cross_alpha, cross_style = '#AAAAFF', 0.7, '-'  # Light blue
        
        # Define overlay color (unified for all overlay elements, normal depth)
        overlay_color = '#666666'  # Dark gray for overlay elements
        
        # Draw overlay elements first (so they appear behind GT)
        if overlay_elements:
            for overlay_points in overlay_elements:
                ax.plot(overlay_points[:, 0], overlay_points[:, 1], 
                       color=overlay_color, linewidth=2.5, linestyle='-',
                       marker='o', markersize=4, markerfacecolor=overlay_color,
                       markeredgecolor=overlay_color, markeredgewidth=0.5, alpha=0.9)
        
        # Draw lane lines (divider) - lighter color
        for lane_points in map_data['lane_lines']:
            ax.plot(lane_points[:, 0], lane_points[:, 1], 
                   color=lane_color, alpha=lane_alpha, linewidth=2.5, linestyle=lane_style,
                   marker='o', markersize=4, markerfacecolor=lane_color,
                   markeredgecolor=lane_color, markeredgewidth=0.5)
        
        # Draw road edges (boundary) - lighter color
        for edge_points in map_data['road_edges']:
            ax.plot(edge_points[:, 0], edge_points[:, 1], 
                   color=edge_color, alpha=edge_alpha, linewidth=2.5, linestyle=edge_style,
                   marker='s', markersize=4, markerfacecolor=edge_color,
                   markeredgecolor=edge_color, markeredgewidth=0.5)
        
        # Draw crosswalks (ped_crossing) - lighter color
        for cross_points in map_data['crosswalks']:
            ax.plot(cross_points[:, 0], cross_points[:, 1], 
                   color=cross_color, alpha=cross_alpha, linewidth=2.5, linestyle=cross_style,
                   marker='^', markersize=4, markerfacecolor=cross_color,
                   markeredgecolor=cross_color, markeredgewidth=0.5)
    
    def _draw_map_elements_with_overlay(self, ax, map_data: Dict, overlay_elements: List[np.ndarray]):
        """Draw map elements with overlay elements using lighter colors for GT, darker for overlay"""
        
        # Define colors and styles for GT elements (lighter colors)
        lane_color, lane_alpha, lane_style = '#FFAAAA', 0.7, '-'  # Light red
        edge_color, edge_alpha, edge_style = '#AAFFAA', 0.7, '-'  # Light green
        cross_color, cross_alpha, cross_style = '#AAAAFF', 0.7, '-'  # Light blue
        
        # Define overlay color (unified for all overlay elements, normal depth)
        overlay_color = '#666666'  # Dark gray for overlay elements
        
        # First, draw overlay elements with darker color (so they appear behind GT)
        for overlay_points in overlay_elements:
            ax.plot(overlay_points[:, 0], overlay_points[:, 1], 
                   color=overlay_color, linewidth=1.5, linestyle='-',
                   marker='o', markersize=3, markerfacecolor=overlay_color,
                   markeredgecolor=overlay_color, markeredgewidth=0.3, alpha=0.9)
        
        # Then, draw original GT elements with lighter colors (on top)
        for lane_points in map_data['lane_lines']:
            ax.plot(lane_points[:, 0], lane_points[:, 1], 
                   color=lane_color, alpha=lane_alpha, linewidth=2.5, linestyle=lane_style,
                   marker='o', markersize=4, markerfacecolor=lane_color,
                   markeredgecolor=lane_color, markeredgewidth=0.5)
        
        for edge_points in map_data['road_edges']:
            ax.plot(edge_points[:, 0], edge_points[:, 1], 
                   color=edge_color, alpha=edge_alpha, linewidth=2.5, linestyle=edge_style,
                   marker='s', markersize=4, markerfacecolor=edge_color,
                   markeredgecolor=edge_color, markeredgewidth=0.5)
        
        for cross_points in map_data['crosswalks']:
            ax.plot(cross_points[:, 0], cross_points[:, 1], 
                   color=cross_color, alpha=cross_alpha, linewidth=2.5, linestyle=cross_style,
                   marker='^', markersize=4, markerfacecolor=cross_color,
                   markeredgecolor=cross_color, markeredgewidth=0.5)
    
    def _generate_overlay_elements(self, map_data: Dict, x_lim: List[float], y_lim: List[float], 
                                 num_elements: int = 20) -> List[np.ndarray]:
        """
        Generate overlay elements with completely random points
        Each element has 20 completely random points within the coordinate bounds
        
        Args:
            map_data: Original GT map data (used for reference only)
            x_lim: X coordinate limits [min, max]
            y_lim: Y coordinate limits [min, max]
            num_elements: Number of overlay elements to generate (default: 20)
        
        Returns:
            List of numpy arrays, each containing 20 random points
        """
        overlay_elements = []
        
        for _ in range(num_elements):
            # Generate 20 completely random points
            random_points = np.zeros((20, 2))
            
            # Each point is completely random within the bounds
            for i in range(20):
                random_points[i, 0] = np.random.uniform(x_lim[0], x_lim[1])  # Random x
                random_points[i, 1] = np.random.uniform(y_lim[0], y_lim[1])  # Random y
            
            overlay_elements.append(random_points)
        
        return overlay_elements
    
    def _generate_random_line(self, x_lim: List[float], y_lim: List[float]) -> np.ndarray:
        """
        Generate a random line with exactly 20 points
        
        Args:
            x_lim: X coordinate limits [min, max]
            y_lim: Y coordinate limits [min, max]
        
        Returns:
            Array of 20 points representing a line
        """
        # Generate random start and end points
        start_x = np.random.uniform(x_lim[0], x_lim[1])
        start_y = np.random.uniform(y_lim[0], y_lim[1])
        end_x = np.random.uniform(x_lim[0], x_lim[1])
        end_y = np.random.uniform(y_lim[0], y_lim[1])
        
        # Create 20 points along the line
        t = np.linspace(0, 1, 20)
        x_points = start_x + t * (end_x - start_x)
        y_points = start_y + t * (end_y - start_y)
        
        # Add some random variation to make it more realistic
        noise_scale = min(abs(end_x - start_x), abs(end_y - start_y)) * 0.1
        if noise_scale > 0:
            x_points += np.random.normal(0, noise_scale * 0.5, 20)
            y_points += np.random.normal(0, noise_scale * 0.5, 20)
        
        return np.column_stack([x_points, y_points])
    
    def visualize_gt_with_overlay(self, map_data: Dict, sample_idx: int = 0, 
                                image_format: str = 'pdf', figsize: Tuple[int, int] = (10, 8),
                                num_overlay_elements: int = 20):
        """
        Visualize GT data with overlay elements (clean appearance, no grids, axes, titles)
        
        Args:
            map_data: Dictionary containing GT map data
            sample_idx: Sample index for filename
            image_format: Output format ('pdf', 'png', 'svg')
            figsize: Figure size as (width, height)
            num_overlay_elements: Number of overlay elements to generate
        """
        # Create figure with clean appearance
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Calculate coordinate ranges for consistent scaling
        all_points = []
        for lane_points in map_data['lane_lines']:
            all_points.extend(lane_points)
        for edge_points in map_data['road_edges']:
            all_points.extend(edge_points)
        for crosswalk_points in map_data['crosswalks']:
            all_points.extend(crosswalk_points)
        
        if all_points:
            all_points = np.array(all_points)
            x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
            y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            x_lim = [x_min - x_margin, x_max + x_margin]
            y_lim = [y_min - y_margin, y_max + y_margin]
        else:
            x_lim = [0, 80]
            y_lim = [0, 70]
        
        # Generate overlay elements
        overlay_elements = self._generate_overlay_elements(map_data, x_lim, y_lim, num_overlay_elements)
        
        # Draw map elements with overlay
        self._draw_map_elements_with_overlay(ax, map_data, overlay_elements)
        
        # Set limits and clean appearance
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        
        # Remove all axes, grids, and labels for clean output
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Set white background
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Save with clean filename indicating overlay
        filename = f'{self.save_dir}/gt_sample_{sample_idx:04d}_overlay_{num_overlay_elements}.{image_format}'
        
        # Save with appropriate parameters for vectorized output
        if image_format == 'pdf':
            plt.savefig(filename, format=image_format, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', pad_inches=0.1)
        elif image_format == 'svg':
            plt.savefig(filename, format=image_format, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', pad_inches=0.1)
        else:
            plt.savefig(filename, format=image_format, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', pad_inches=0.1)
        
        plt.close()  # Close to save memory
        
        print(f"✅ GT visualization with overlay saved: {filename}")
        print(f"   Original elements: {len(map_data['lane_lines'])} lanes, {len(map_data['road_edges'])} edges, {len(map_data['crosswalks'])} crosswalks")
        print(f"   Overlay elements: {len(overlay_elements)} random elements")
        return filename

    def _add_noise_to_elements(self, map_data: Dict, noise_level: float = 0.1) -> Dict:
        """
        Add noise to GT elements
        
        Args:
            map_data: Original GT map data
            noise_level: Noise level (0.0 = no noise, 1.0 = high noise)
        
        Returns:
            Dictionary with noisy elements
        """
        noisy_data = {
            'lane_lines': [],
            'road_edges': [],
            'crosswalks': []
        }
        
        # Add noise to lane lines
        for lane_points in map_data['lane_lines']:
            noisy_points = lane_points.copy()
            if noise_level > 0:
                noise_x = np.random.normal(0, noise_level * 5, len(noisy_points))  # 5 meters std
                noise_y = np.random.normal(0, noise_level * 5, len(noisy_points))
                noisy_points[:, 0] += noise_x
                noisy_points[:, 1] += noise_y
            noisy_data['lane_lines'].append(noisy_points)
        
        # Add noise to road edges
        for edge_points in map_data['road_edges']:
            noisy_points = edge_points.copy()
            if noise_level > 0:
                noise_x = np.random.normal(0, noise_level * 5, len(noisy_points))
                noise_y = np.random.normal(0, noise_level * 5, len(noisy_points))
                noisy_points[:, 0] += noise_x
                noisy_points[:, 1] += noise_y
            noisy_data['road_edges'].append(noisy_points)
        
        # Add noise to crosswalks
        for cross_points in map_data['crosswalks']:
            noisy_points = cross_points.copy()
            if noise_level > 0:
                noise_x = np.random.normal(0, noise_level * 5, len(noisy_points))
                noise_y = np.random.normal(0, noise_level * 5, len(noisy_points))
                noisy_points[:, 0] += noise_x
                noisy_points[:, 1] += noise_y
            noisy_data['crosswalks'].append(noisy_points)
        
        return noisy_data
    
    def _expand_elements_to_20(self, map_data: Dict, x_lim: List[float], y_lim: List[float]) -> Dict:
        """
        Expand elements to exactly 20 elements by duplicating and adding variations
        
        Args:
            map_data: Original GT map data
            x_lim: X coordinate limits
            y_lim: Y coordinate limits
        
        Returns:
            Dictionary with expanded elements (20 total)
        """
        expanded_data = {
            'lane_lines': [],
            'road_edges': [],
            'crosswalks': []
        }
        
        # Get all original elements
        all_elements = []
        element_types = []
        
        for lane_points in map_data['lane_lines']:
            all_elements.append(lane_points)
            element_types.append('lane_lines')
        
        for edge_points in map_data['road_edges']:
            all_elements.append(edge_points)
            element_types.append('road_edges')
        
        for cross_points in map_data['crosswalks']:
            all_elements.append(cross_points)
            element_types.append('crosswalks')
        
        # If we have fewer than 20 elements, duplicate and add variations
        target_count = 20
        current_count = len(all_elements)
        
        if current_count == 0:
            # Generate random elements if no original elements
            for i in range(target_count):
                random_element = self._generate_random_element_with_structure(x_lim, y_lim)
                expanded_data['lane_lines'].append(random_element)
        else:
            # Duplicate existing elements with variations
            for i in range(target_count):
                # Select an element to duplicate (cycle through originals)
                source_idx = i % current_count
                source_element = all_elements[source_idx]
                source_type = element_types[source_idx]
                
                # Add variation to the duplicated element
                varied_element = source_element.copy()
                variation_scale = min(abs(np.max(source_element[:, 0]) - np.min(source_element[:, 0])), 
                                    abs(np.max(source_element[:, 1]) - np.min(source_element[:, 1]))) * 0.1
                
                if variation_scale > 0:
                    noise_x = np.random.normal(0, variation_scale, len(varied_element))
                    noise_y = np.random.normal(0, variation_scale, len(varied_element))
                    varied_element[:, 0] += noise_x
                    varied_element[:, 1] += noise_y
                
                # Add to appropriate category
                expanded_data[source_type].append(varied_element)
        
        return expanded_data
    
    def _generate_random_element_with_structure(self, x_lim: List[float], y_lim: List[float]) -> np.ndarray:
        """Generate a random element with 20 points that has some structure"""
        # Generate a random path with some structure
        start_x = np.random.uniform(x_lim[0], x_lim[1])
        start_y = np.random.uniform(y_lim[0], y_lim[1])
        
        # Create a path with some direction
        direction = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(10, 30)
        
        t = np.linspace(0, 1, 20)
        end_x = start_x + length * np.cos(direction)
        end_y = start_y + length * np.sin(direction)
        
        # Ensure end point is within bounds
        end_x = np.clip(end_x, x_lim[0], x_lim[1])
        end_y = np.clip(end_y, y_lim[0], y_lim[1])
        
        # Create base line
        x_points = start_x + t * (end_x - start_x)
        y_points = start_y + t * (end_y - start_y)
        
        # Add some curvature
        curvature = np.random.uniform(-5, 5)
        curve_noise = curvature * np.sin(t * np.pi)
        
        # Apply curvature perpendicular to the line direction
        perp_direction = direction + np.pi/2
        x_points += curve_noise * np.cos(perp_direction)
        y_points += curve_noise * np.sin(perp_direction)
        
        return np.column_stack([x_points, y_points])
    
    def _generate_completely_random_elements(self, x_lim: List[float], y_lim: List[float], 
                                           num_elements: int = 20) -> List[np.ndarray]:
        """Generate completely random elements (final stage)"""
        random_elements = []
        
        for _ in range(num_elements):
            # Generate 20 completely random points
            random_points = np.zeros((20, 2))
            
            for i in range(20):
                random_points[i, 0] = np.random.uniform(x_lim[0], x_lim[1])
                random_points[i, 1] = np.random.uniform(y_lim[0], y_lim[1])
            
            random_elements.append(random_points)
        
        return random_elements
    
    def _draw_training_process_elements(self, ax, elements, title: str, stage: str):
        """Draw elements for training process visualization"""
        
        # Define colors based on stage
        if stage == 'gt':
            # GT: Original colors with full opacity
            lane_color, lane_alpha = 'red', 0.9
            edge_color, edge_alpha = 'green', 0.9
            cross_color, cross_alpha = 'blue', 0.9
            linewidth, markersize = 2.5, 4
        elif stage == 'early':
            # Early training: Slightly lighter
            lane_color, lane_alpha = 'red', 0.7
            edge_color, edge_alpha = 'green', 0.7
            cross_color, cross_alpha = 'blue', 0.7
            linewidth, markersize = 2.5, 4
        elif stage == 'mid':
            # Mid training: Keep original colors same as early training
            lane_color, lane_alpha = 'red', 0.7
            edge_color, edge_alpha = 'green', 0.7
            cross_color, cross_alpha = 'blue', 0.7
            linewidth, markersize = 2.0, 3
        elif stage == 'final':
            # Final: Unified color, for random elements
            lane_color = edge_color = cross_color = '#333333'
            lane_alpha = edge_alpha = cross_alpha = 0.8
            linewidth, markersize = 1.5, 3
        
        # Draw elements based on type
        if isinstance(elements, dict):
            # Dictionary format (GT and expanded)
            for lane_points in elements['lane_lines']:
                ax.plot(lane_points[:, 0], lane_points[:, 1], 
                       color=lane_color, alpha=lane_alpha, linewidth=linewidth, linestyle='-',
                       marker='o', markersize=markersize, markerfacecolor=lane_color,
                       markeredgecolor=lane_color, markeredgewidth=0.5)
            
            for edge_points in elements['road_edges']:
                ax.plot(edge_points[:, 0], edge_points[:, 1], 
                       color=edge_color, alpha=edge_alpha, linewidth=linewidth, linestyle='-',
                       marker='s', markersize=markersize, markerfacecolor=edge_color,
                       markeredgecolor=edge_color, markeredgewidth=0.5)
            
            for cross_points in elements['crosswalks']:
                ax.plot(cross_points[:, 0], cross_points[:, 1], 
                       color=cross_color, alpha=cross_alpha, linewidth=linewidth, linestyle='-',
                       marker='^', markersize=markersize, markerfacecolor=cross_color,
                       markeredgecolor=cross_color, markeredgewidth=0.5)
        else:
            # List format (random elements)
            for element_points in elements:
                ax.plot(element_points[:, 0], element_points[:, 1], 
                       color=lane_color, alpha=lane_alpha, linewidth=linewidth, linestyle='-',
                       marker='o', markersize=markersize, markerfacecolor=lane_color,
                       markeredgecolor=lane_color, markeredgewidth=0.3)
        
        # ax.set_title(title, fontsize=14, fontweight='bold', pad=20)  # Remove title
    
    def visualize_training_process(self, map_data: Dict, sample_idx: int = 0, 
                                 image_format: str = 'pdf', figsize: Tuple[int, int] = (15, 5)):
        """
        Visualize the training process from GT to random elements
        
        Args:
            map_data: Dictionary containing GT map data
            sample_idx: Sample index for filename
            image_format: Output format ('pdf', 'png', 'svg')
            figsize: Figure size as (width, height)
        """
        # Create figure with 4 subplots
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # Calculate coordinate ranges
        all_points = []
        for lane_points in map_data['lane_lines']:
            all_points.extend(lane_points)
        for edge_points in map_data['road_edges']:
            all_points.extend(edge_points)
        for crosswalk_points in map_data['crosswalks']:
            all_points.extend(crosswalk_points)
        
        if all_points:
            all_points = np.array(all_points)
            x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
            y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            x_lim = [x_min - x_margin, x_max + x_margin]
            y_lim = [y_min - y_margin, y_max + y_margin]
        else:
            x_lim = [0, 80]
            y_lim = [0, 70]
        
        # Generate data in order but display in reverse order
        # Stage 1: Original GT with slight noise
        stage1_data = self._add_noise_to_elements(map_data, noise_level=0.05)
        
        # Stage 2: Expanded to 20 elements based on stage1 results
        stage2_data = self._expand_elements_to_20(stage1_data, x_lim, y_lim)
        stage2_data = self._add_noise_to_elements(stage2_data, noise_level=0.15)
        
        # Stage 3: Mid training with more noise based on stage2
        stage3_data = self._add_noise_to_elements(stage2_data, noise_level=0.8)
        
        # Stage 4: Final - completely random elements
        stage4_data = self._generate_completely_random_elements(x_lim, y_lim, 20)
        
        # Display in reverse order (stage4 -> stage3 -> stage2 -> stage1)
        self._draw_training_process_elements(axes[0], stage4_data, 
                                           'Stage 4: Final Prediction', 'final')
        
        self._draw_training_process_elements(axes[1], stage3_data, 
                                           'Stage 3: Mid Training', 'mid')
        
        self._draw_training_process_elements(axes[2], stage2_data, 
                                           'Stage 2: Early Training', 'early')
        
        self._draw_training_process_elements(axes[3], stage1_data, 
                                           'Stage 1: Random Init', 'gt')
        
        # Set limits and clean appearance for all subplots
        for ax in axes:
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_aspect('equal')
            
            # Remove axes, grids, and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Set white background
            ax.set_facecolor('white')
        
        # Set white background for the entire figure
        fig.patch.set_facecolor('white')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the training process visualization
        filename = f'{self.save_dir}/training_sample_{sample_idx:04d}_training_process.{image_format}'
        
        if image_format == 'pdf':
            plt.savefig(filename, format=image_format, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', pad_inches=0.1)
        elif image_format == 'svg':
            plt.savefig(filename, format=image_format, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', pad_inches=0.1)
        else:
            plt.savefig(filename, format=image_format, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', pad_inches=0.1)
        
        # Save individual stages as separate files
        individual_files = []
        stage_data = [
            (stage4_data, 'stage4_final_prediction', 'final'),
            (stage3_data, 'stage3_mid_training', 'mid'),
            (stage2_data, 'stage2_early_training', 'early'),
            (stage1_data, 'stage1_random_init', 'gt')
        ]
        
        for i, (data, stage_name, stage_type) in enumerate(stage_data):
            # Create individual figure
            fig_single, ax_single = plt.subplots(1, 1, figsize=(figsize[0]/4, figsize[1]))
            
            # Draw the stage
            self._draw_training_process_elements(ax_single, data, '', stage_type)
            
            # Set limits and clean appearance
            ax_single.set_xlim(x_lim)
            ax_single.set_ylim(y_lim)
            ax_single.set_aspect('equal')
            
            # Remove axes, grids, and labels
            ax_single.set_xticks([])
            ax_single.set_yticks([])
            ax_single.spines['top'].set_visible(False)
            ax_single.spines['right'].set_visible(False)
            ax_single.spines['bottom'].set_visible(False)
            ax_single.spines['left'].set_visible(False)
            
            # Set white background
            ax_single.set_facecolor('white')
            fig_single.patch.set_facecolor('white')
            
            # Save individual stage
            stage_filename = f'{self.save_dir}/training_sample_{sample_idx:04d}_{stage_name}.{image_format}'
            
            if image_format == 'pdf':
                plt.savefig(stage_filename, format=image_format, bbox_inches='tight', 
                           facecolor='white', edgecolor='none', pad_inches=0.1)
            elif image_format == 'svg':
                plt.savefig(stage_filename, format=image_format, bbox_inches='tight', 
                           facecolor='white', edgecolor='none', pad_inches=0.1)
            else:
                plt.savefig(stage_filename, format=image_format, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none', pad_inches=0.1)
            
            plt.close(fig_single)
            individual_files.append(stage_filename)
        
        plt.close()
        
        print(f"✅ Training process visualization saved: {filename}")
        print(f"   Shows progression from GT to random elements across 4 stages")
        print(f"✅ Individual stage files saved:")
        for stage_file in individual_files:
            print(f"   - {os.path.basename(stage_file)}")
        
        return [filename] + individual_files
    
    def visualize_multiple_samples(self, sample_indices: List[int] = None, 
                                  image_format: str = 'pdf', figsize: Tuple[int, int] = (10, 8)):
        """
        Visualize multiple samples
        
        Args:
            sample_indices: List of sample indices to visualize. If None, visualize first 10 samples
            image_format: Output format ('pdf', 'png', 'svg')
            figsize: Figure size as (width, height)
        """
        if self.dataset is None:
            print("❌ No dataset loaded")
            return
        
        if sample_indices is None:
            sample_indices = list(range(min(10, len(self.dataset))))
        
        print(f"🎯 Visualizing {len(sample_indices)} samples...")
        
        generated_files = []
        for i, sample_idx in enumerate(sample_indices):
            try:
                # Extract GT data
                map_data = self.extract_gt_map_data(sample_idx)
                
                # Visualize GT data
                filename = self.visualize_gt_clean(map_data, sample_idx, image_format, figsize)
                generated_files.append(filename)
                
                print(f"📊 Progress: {i+1}/{len(sample_indices)} samples processed")
                
            except Exception as e:
                print(f"❌ Error processing sample {sample_idx}: {e}")
                continue
        
        print(f"\n🎉 Completed! Generated {len(generated_files)} visualizations")
        print(f"📁 Files saved in: {self.save_dir}")
        
        return generated_files
    
    def visualize_multiple_samples_with_overlay(self, sample_indices: List[int] = None, 
                                               image_format: str = 'pdf', figsize: Tuple[int, int] = (10, 8),
                                               num_overlay_elements: int = 20):
        """
        Visualize multiple samples with overlay elements
        
        Args:
            sample_indices: List of sample indices to visualize. If None, visualize first 10 samples
            image_format: Output format ('pdf', 'png', 'svg')
            figsize: Figure size as (width, height)
            num_overlay_elements: Number of overlay elements to generate for each sample
        """
        if self.dataset is None:
            print("❌ No dataset loaded")
            return
        
        if sample_indices is None:
            sample_indices = list(range(min(10, len(self.dataset))))
        
        print(f"🎯 Visualizing {len(sample_indices)} samples with overlay elements...")
        
        generated_files = []
        for i, sample_idx in enumerate(sample_indices):
            try:
                # Extract GT data
                map_data = self.extract_gt_map_data(sample_idx)
                
                # Visualize GT data with overlay
                filename = self.visualize_gt_with_overlay(map_data, sample_idx, image_format, figsize, num_overlay_elements)
                generated_files.append(filename)
                
                print(f"📊 Progress: {i+1}/{len(sample_indices)} samples processed")
                
            except Exception as e:
                print(f"❌ Error processing sample {sample_idx}: {e}")
                continue
        
        print(f"\n🎉 Completed! Generated {len(generated_files)} visualizations with overlay")
        print(f"📁 Files saved in: {self.save_dir}")
        
        return generated_files

    def visualize_multiple_training_processes(self, sample_indices: List[int] = None, 
                                            image_format: str = 'pdf', figsize: Tuple[int, int] = (15, 5)):
        """
        Visualize multiple training processes
        
        Args:
            sample_indices: List of sample indices to visualize. If None, visualize first 10 samples
            image_format: Output format ('pdf', 'png', 'svg')
            figsize: Figure size as (width, height)
        """
        if self.dataset is None:
            print("❌ No dataset loaded")
            return
        
        if sample_indices is None:
            sample_indices = list(range(min(10, len(self.dataset))))
        
        print(f"🎯 Visualizing training processes for {len(sample_indices)} samples...")
        
        generated_files = []
        for i, sample_idx in enumerate(sample_indices):
            try:
                # Extract GT data
                map_data = self.extract_gt_map_data(sample_idx)
                
                # Visualize training process
                filenames = self.visualize_training_process(map_data, sample_idx, image_format, figsize)
                generated_files.extend(filenames)  # extend instead of append since we now return multiple files
                
                print(f"📊 Progress: {i+1}/{len(sample_indices)} samples processed")
                
            except Exception as e:
                print(f"❌ Error processing sample {sample_idx}: {e}")
                continue
        
        print(f"\n🎉 Completed! Generated {len(generated_files)} training process visualizations")
        print(f"📁 Files saved in: {self.save_dir}")
        
        return generated_files

    def create_summary_report(self, generated_files: List[str]):
        """Create a summary report of the visualization process"""
        
        summary_content = f"""# Dataset GT Visualization Summary

## 📊 Visualization Results

- **Total samples processed**: {len(generated_files)}
- **Output directory**: {self.save_dir}
- **Configuration**: {self.config_path}

## 📁 Generated Files

"""
        
        for i, filename in enumerate(generated_files, 1):
            summary_content += f"{i}. `{os.path.basename(filename)}`\n"
        
        summary_content += f"""
## 🎯 Visualization Features

- ✅ Clean appearance (no grids, axes, or titles)
- ✅ Vectorized output for high-quality printing
- ✅ Consistent color coding:
  - 🔴 Red: Lane lines (divider) - linewidth=2.5, marker='o', markersize=4, alpha=0.9
  - 🟢 Green: Road edges (boundary) - linewidth=2.5, marker='s', markersize=4, alpha=0.9
  - 🔵 Blue: Crosswalks (ped_crossing) - linewidth=2.5, marker='^', markersize=4, alpha=0.9
- ✅ Equal aspect ratio for accurate geometric representation
- ✅ Automatic coordinate scaling

## 🔧 Technical Details

- **Font type**: TrueType (Type 42) for vector compatibility
- **Background**: White
- **Padding**: Minimal (0.1 inches)
- **DPI**: 300 (for raster formats)

---
*Generated by Dataset GT Visualizer*
*Date: {import_plugin.__doc__ if hasattr(import_plugin, '__doc__') else 'N/A'}*
"""
        
        # Save summary
        summary_filename = os.path.join(self.save_dir, 'visualization_summary.md')
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"📋 Summary report saved: {summary_filename}")
        return summary_filename


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Dataset GT Visualizer - Clean GT visualization from datasets')
    parser.add_argument('--config', 
                        default='./plugin/configs/nusc_newsplit_480_60x30.py',
                        help='Path to dataset configuration file')
    parser.add_argument('--save-dir', 
                        default='./dataset_gt_visualizations',
                        help='Directory to save visualization results')
    parser.add_argument('--samples', 
                        default='0',
                        help='Comma-separated list of sample indices to visualize (e.g., "0,1,2,3,4")')
    parser.add_argument('--format', 
                        default='pdf',
                        choices=['pdf', 'png', 'svg'],
                        help='Output image format')
    parser.add_argument('--figsize', 
                        default='10,8',
                        help='Figure size as width,height (e.g., "10,8")')
    parser.add_argument('--overlay', 
                        action='store_true',
                        help='Generate overlay visualization with additional elements')
    parser.add_argument('--overlay-elements', 
                        type=int, default=20,
                        help='Number of overlay elements to generate (default: 20)')
    parser.add_argument('--training-process', 
                        action='store_true',
                        help='Generate training process visualization showing progression from GT to random elements')
    
    args = parser.parse_args()
    
    # Parse sample indices
    try:
        sample_indices = [int(x.strip()) for x in args.samples.split(',')]
    except ValueError:
        print("❌ Invalid sample indices format. Use comma-separated integers.")
        return
    
    # Parse figure size
    try:
        figsize = tuple(map(int, args.figsize.split(',')))
    except ValueError:
        print("❌ Invalid figure size format. Use width,height (e.g., '10,8').")
        return
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return
    
    print("🚀 Starting Dataset GT Visualization...")
    print("=" * 50)
    
    # Create visualizer
    visualizer = DatasetGTVisualizer(
        config_path=args.config,
        save_dir=args.save_dir
    )
    
    # Generate visualizations
    if args.training_process:
        print("🎨 Generating training process visualizations...")
        generated_files = visualizer.visualize_multiple_training_processes(
            sample_indices=sample_indices,
            image_format=args.format,
            figsize=figsize
        )
    elif args.overlay:
        print(f"🎨 Generating overlay visualizations with {args.overlay_elements} additional elements...")
        generated_files = visualizer.visualize_multiple_samples_with_overlay(
            sample_indices=sample_indices,
            image_format=args.format,
            figsize=figsize,
            num_overlay_elements=args.overlay_elements
        )
    else:
        print("🎨 Generating clean GT visualizations...")
        generated_files = visualizer.visualize_multiple_samples(
            sample_indices=sample_indices,
            image_format=args.format,
            figsize=figsize
        )
    
    # Create summary report
    if generated_files:
        visualizer.create_summary_report(generated_files)
    
    print("\n🎉 Dataset GT visualization completed!")


if __name__ == "__main__":
    main()
