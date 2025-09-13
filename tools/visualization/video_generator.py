#!/usr/bin/env python3
"""
Visualization Video Generator

This script processes the output from visualize.py and creates:
1. Combines 6 surround-view images into a 2x3 grid
2. Adds GT bird's-eye view on the right
3. Adds Pred bird's-eye view on the far right
4. Generates a video from all frames with 1 second intervals
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import glob
from typing import List, Tuple


class VisualizationVideoGenerator:
    def __init__(self, input_dir: str, output_dir: str = "./video_output", scene_name: str = None):
        """
        Initialize the video generator
        
        Args:
            input_dir: Directory containing visualization results from visualize.py
            output_dir: Directory to save combined images and video
            scene_name: Specific scene name to process (e.g., 'scene-0331'). If None, uses the first scene found.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find scene directory
        self.scene_dirs = [d for d in self.input_dir.iterdir() if d.is_dir() and d.name.startswith('scene-')]
        if not self.scene_dirs:
            raise ValueError(f"No scene directories found in {input_dir}")
        
        # Select specific scene or use first one
        if scene_name:
            scene_path = self.input_dir / scene_name
            if not scene_path.exists() or not scene_path.is_dir():
                available_scenes = [d.name for d in self.scene_dirs]
                raise ValueError(f"Scene '{scene_name}' not found. Available scenes: {available_scenes}")
            self.scene_dir = scene_path
        else:
            self.scene_dir = self.scene_dirs[0]  # Use the first scene found
        
        print(f"🎬 Processing scene: {self.scene_dir.name}")
        print(f"📁 Output directory: {output_dir}")
    
    def get_camera_layout(self) -> List[str]:
        """
        Define the camera layout for 2x3 grid
        NuScenes camera names: CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT, 
                              CAM_BACK, CAM_BACK_LEFT, CAM_FRONT_LEFT
        """
        # Standard layout: Front cameras on top, back cameras on bottom
        camera_layout = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',      # Top row
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'           # Bottom row
        ]
        return camera_layout
    
    def load_and_resize_image_keep_ratio(self, image_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Load and resize an image while keeping aspect ratio
        
        Args:
            image_path: Path to the image
            target_size: Target size as (width, height)
        
        Returns:
            Resized image as numpy array with padding to match target size
        """
        if not image_path.exists():
            # Create placeholder image if file doesn't exist
            placeholder = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 128
            cv2.putText(placeholder, f"Missing: {image_path.name}", 
                       (10, target_size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            return placeholder
        
        image = cv2.imread(str(image_path))
        if image is None:
            # Create placeholder if image can't be loaded
            placeholder = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 128
            cv2.putText(placeholder, f"Error: {image_path.name}", 
                       (10, target_size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 0, 0), 1)
            return placeholder
        
        # Get original dimensions
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor to fit within target size while keeping aspect ratio
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized_image = cv2.resize(image, (new_w, new_h))
        
        # Create canvas with target size and white background
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        
        # Calculate position to center the image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Place resized image on canvas
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
        
        return canvas
    
    def create_combined_frame(self, frame_dir: Path, target_cam_size: Tuple[int, int] = (400, 300),
                            target_bev_size: Tuple[int, int] = (600, 600)) -> np.ndarray:
        """
        Create a combined frame with 6 camera views and 2 bird's-eye views
        
        Args:
            frame_dir: Directory containing gt and pred folders
            target_cam_size: Target size for each camera view (width, height)
            target_bev_size: Target size for bird's-eye views (width, height)
        
        Returns:
            Combined image as numpy array
        """
        camera_layout = self.get_camera_layout()
        
        # Load camera images
        camera_images = []
        gt_dir = frame_dir / 'gt'
        
        for camera_name in camera_layout:
            # Look for camera images in gt directory
            camera_files = list(gt_dir.glob(f"*{camera_name}*.jpg")) + list(gt_dir.glob(f"*{camera_name}*.png"))
            if camera_files:
                camera_path = camera_files[0]
            else:
                # Create placeholder if camera image not found
                print(f"⚠️ Camera image not found for {camera_name} in {gt_dir}")
                camera_path = gt_dir / f"{camera_name}_placeholder.jpg"
            
            camera_img = self.load_and_resize_image_keep_ratio(camera_path, target_cam_size)
            camera_images.append(camera_img)
        
        # Create 2x3 grid of camera images
        top_row = np.hstack(camera_images[:3])    # First 3 cameras
        bottom_row = np.hstack(camera_images[3:]) # Last 3 cameras
        camera_grid = np.vstack([top_row, bottom_row])
        
        # Load GT bird's-eye view
        gt_bev_files = list(gt_dir.glob("*map*.jpg")) + list(gt_dir.glob("*map*.png"))
        if gt_bev_files:
            gt_bev_path = gt_bev_files[0]
        else:
            print(f"⚠️ GT bird's-eye view not found in {gt_dir}")
            gt_bev_path = gt_dir / "map_placeholder.jpg"
        
        gt_bev = self.load_and_resize_image_keep_ratio(gt_bev_path, target_bev_size)
        
        # Load Pred bird's-eye view
        pred_dir = frame_dir / 'pred'
        pred_bev_files = list(pred_dir.glob("*map*.jpg")) + list(pred_dir.glob("*map*.png"))
        if pred_bev_files:
            pred_bev_path = pred_bev_files[0]
        else:
            print(f"⚠️ Pred bird's-eye view not found in {pred_dir}")
            pred_bev_path = pred_dir / "map_placeholder.jpg"
        
        pred_bev = self.load_and_resize_image_keep_ratio(pred_bev_path, target_bev_size)
        
        # Calculate heights and pad if necessary
        cam_height = camera_grid.shape[0]
        bev_height = target_bev_size[1]
        
        # Make all components the same height
        target_height = max(cam_height, bev_height)
        
        # Pad camera grid if needed
        if cam_height < target_height:
            padding = target_height - cam_height
            camera_grid = np.pad(camera_grid, ((0, padding), (0, 0), (0, 0)), 
                               mode='constant', constant_values=255)
        
        # Pad BEV images if needed
        if bev_height < target_height:
            padding = target_height - bev_height
            gt_bev = np.pad(gt_bev, ((0, padding), (0, 0), (0, 0)), 
                          mode='constant', constant_values=255)
            pred_bev = np.pad(pred_bev, ((0, padding), (0, 0), (0, 0)), 
                            mode='constant', constant_values=255)
        
        # Add text labels
        gt_bev_labeled = gt_bev.copy()
        pred_bev_labeled = pred_bev.copy()
        
        cv2.putText(gt_bev_labeled, "Ground Truth", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(pred_bev_labeled, "Prediction", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Combine all components horizontally
        combined_frame = np.hstack([camera_grid, gt_bev_labeled, pred_bev_labeled])
        
        return combined_frame
    
    def process_all_frames(self, cam_size: Tuple[int, int] = (400, 300),
                          bev_size: Tuple[int, int] = (600, 600)) -> List[str]:
        """
        Process all frames and create combined images
        
        Args:
            cam_size: Size for camera views (width, height)
            bev_size: Size for bird's-eye views (width, height)
        
        Returns:
            List of paths to generated combined images
        """
        # Get all frame directories (numbered 1 to N)
        frame_dirs = []
        for item in self.scene_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                frame_dirs.append((int(item.name), item))
        
        # Sort by frame number
        frame_dirs.sort(key=lambda x: x[0])
        
        if not frame_dirs:
            print("❌ No numbered frame directories found")
            return []
        
        print(f"🎬 Found {len(frame_dirs)} frames to process")
        
        generated_files = []
        
        for frame_num, frame_dir in frame_dirs:
            try:
                print(f"📷 Processing frame {frame_num}/{len(frame_dirs)}")
                
                # Create combined frame
                combined_img = self.create_combined_frame(frame_dir, cam_size, bev_size)
                
                # Add frame number overlay
                cv2.putText(combined_img, f"Frame {frame_num}", 
                           (20, combined_img.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Save combined frame
                output_path = self.output_dir / f"frame_{frame_num:04d}.jpg"
                cv2.imwrite(str(output_path), combined_img)
                generated_files.append(str(output_path))
                
            except Exception as e:
                print(f"❌ Error processing frame {frame_num}: {e}")
                continue
        
        print(f"✅ Generated {len(generated_files)} combined frames")
        return generated_files
    
    def create_video(self, image_files: List[str], fps: float = 1.0, 
                    output_name: str = "visualization_video.mp4") -> str:
        """
        Create video from image files
        
        Args:
            image_files: List of image file paths
            fps: Frames per second (1.0 for 1 second intervals)
            output_name: Output video filename
        
        Returns:
            Path to generated video
        """
        if not image_files:
            print("❌ No image files to create video")
            return ""
        
        video_path = self.output_dir / output_name
        
        try:
            # Read first image to get dimensions
            first_img = cv2.imread(image_files[0])
            height, width = first_img.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            for img_path in image_files:
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize if necessary
                    if img.shape[:2] != (height, width):
                        img = cv2.resize(img, (width, height))
                    video_writer.write(img)
                else:
                    print(f"⚠️ Could not read image: {img_path}")
            
            video_writer.release()
            print(f"🎥 Video created: {video_path}")
            print(f"   Duration: {len(image_files)} seconds")
            print(f"   FPS: {fps}")
            
            return str(video_path)
            
        except Exception as e:
            print(f"❌ Error creating video: {e}")
            return ""
    
    def generate_visualization_video(self, cam_size: Tuple[int, int] = (400, 300),
                                   bev_size: Tuple[int, int] = (600, 600),
                                   fps: float = 1.0) -> str:
        """
        Complete pipeline: process frames and create video
        
        Args:
            cam_size: Size for camera views (width, height)
            bev_size: Size for bird's-eye views (width, height)
            fps: Video frame rate (1.0 for 1 second intervals)
        
        Returns:
            Path to generated video
        """
        print("🚀 Starting visualization video generation...")
        print("=" * 60)
        
        # Process all frames
        image_files = self.process_all_frames(cam_size, bev_size)
        
        if not image_files:
            print("❌ No frames processed, cannot create video")
            return ""
        
        # Create video
        video_path = self.create_video(image_files, fps, 
                                     f"{self.scene_dir.name}_visualization.mp4")
        
        print("=" * 60)
        print("🎉 Visualization video generation completed!")
        
        return video_path


def main():
    parser = argparse.ArgumentParser(description='Generate visualization video from visualize.py output')
    parser.add_argument('input_dir', 
                        help='Directory containing visualization results from visualize.py')
    parser.add_argument('--output-dir', 
                        default='./video_output',
                        help='Directory to save combined images and video')
    parser.add_argument('--scene', 
                        default=None,
                        help='Specific scene to process (e.g., "scene-0331"). If not specified, uses the first scene found.')
    parser.add_argument('--cam-size', 
                        default='400,300',
                        help='Camera view size as width,height (e.g., "400,300")')
    parser.add_argument('--bev-size', 
                        default='600,600',
                        help='Bird\'s-eye view size as width,height (e.g., "600,600")')
    parser.add_argument('--fps', 
                        type=float, default=1.0,
                        help='Video frame rate (1.0 for 1 second intervals)')
    
    args = parser.parse_args()
    
    # Parse sizes
    try:
        cam_width, cam_height = map(int, args.cam_size.split(','))
        cam_size = (cam_width, cam_height)
    except ValueError:
        print("❌ Invalid camera size format. Use width,height (e.g., '400,300')")
        return
    
    try:
        bev_width, bev_height = map(int, args.bev_size.split(','))
        bev_size = (bev_width, bev_height)
    except ValueError:
        print("❌ Invalid BEV size format. Use width,height (e.g., '600,600')")
        return
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    try:
        # Create generator
        generator = VisualizationVideoGenerator(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            scene_name=args.scene
        )
        
        # Generate video
        video_path = generator.generate_visualization_video(
            cam_size=cam_size,
            bev_size=bev_size,
            fps=args.fps
        )
        
        if video_path:
            print(f"\n🎥 Video successfully created: {video_path}")
        else:
            print("\n❌ Failed to create video")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
