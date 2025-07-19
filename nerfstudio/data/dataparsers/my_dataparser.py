# Copyright 2024 the authors of NeuRAD and contributors.
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for Street Gaussians / Waymo-style data format."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Type, Union, Any

import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json


@dataclass
class StreetGaussianDataParserConfig(DataParserConfig):
    """Configuration for Street Gaussians data parser (Waymo-style format)."""
    
    _target: Type = field(default_factory=lambda: StreetGaussianDataParser)
    """Target class to instantiate."""
    
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    
    orientation_method: str = "none"
    """The method to use for orientation. Options: pca, up, vertical, none."""
    
    center_method: str = "none"
    """The method to use to center the poses. Options: poses, focus, none."""
    
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    
    depth_unit_scale_factor: float = 1.0
    """Scales the depth values to meters."""
    
    start_time: float = 0.0
    """Start time for temporal sampling."""
    
    end_time: float = 1.0
    """End time for temporal sampling."""
    
    cameras_to_use: List[str] = field(default_factory=lambda: ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"])
    """List of camera names to use."""


@dataclass  
class StreetGaussianDataParser(DataParser):
    """Street Gaussians Data Parser for Waymo-style data format.
    
    This parser is designed to load Street Gaussians / Waymo Open Dataset format.
    Expected data structure (similar to Street Gaussians preprocessing):
    
    data/
    ├── transforms.json          # Contains camera poses, intrinsics, and metadata
    ├── images/                  # Directory containing camera images
    │   ├── 000/                 # Frame number
    │   │   ├── CAM_FRONT.jpg
    │   │   ├── CAM_FRONT_LEFT.jpg
    │   │   ├── CAM_FRONT_RIGHT.jpg
    │   │   ├── CAM_BACK_LEFT.jpg
    │   │   └── CAM_BACK_RIGHT.jpg
    │   ├── 001/
    │   └── ...
    ├── masks/ (optional)        # Directory containing sky masks
    │   ├── 000/
    │   │   ├── CAM_FRONT.png
    │   │   └── ...
    │   └── ...
    └── depths/ (optional)       # Directory containing depth maps
        ├── 000/
        │   ├── CAM_FRONT.npz
        │   └── ...
        └── ...
    """

    config: StreetGaussianDataParserConfig
    includes_time: bool = True  # Street Gaussians includes temporal information

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        """Generate dataparser outputs for the given split.
        
        Args:
            split: Which dataset split to generate (train/test).
            
        Returns:
            DataparserOutputs containing all the data for the specified split.
        """
        # Load transforms.json file containing camera poses and intrinsics
        transforms_file = self.config.data / "transforms.json"
        
        if not transforms_file.exists():
            raise FileNotFoundError(f"transforms.json not found at {transforms_file}")
            
        with open(transforms_file, "r") as f:
            meta = json.load(f)
        
        # Get camera information and frame data
        cameras_info = meta.get("cameras", {})
        frames = meta.get("frames", [])
        
        if not frames:
            raise ValueError("No frames found in transforms.json")
        
        # Temporal filtering based on start_time and end_time
        total_frames = len(frames)
        start_idx = int(self.config.start_time * total_frames)
        end_idx = int(self.config.end_time * total_frames)
        frames = frames[start_idx:end_idx]
        
        # Collect all camera data
        all_image_filenames = []
        all_poses = []
        all_times = []
        all_camera_types = []
        all_fx = []
        all_fy = []
        all_cx = []
        all_cy = []
        all_heights = []
        all_widths = []
        all_distortion_params = []
        
        for frame_idx, frame in enumerate(frames):
            frame_time = frame.get("time", frame_idx / len(frames))
            
            # Process each camera in the frame
            for camera_name in self.config.cameras_to_use:
                if camera_name not in frame.get("cameras", {}):
                    continue
                    
                camera_data = frame["cameras"][camera_name]
                
                # Get image path
                image_path = camera_data.get("image_path", f"images/{frame_idx:03d}/{camera_name}.jpg")
                if not image_path.startswith("/"):
                    image_path = self.config.data / image_path
                else:
                    image_path = Path(image_path)
                
                # Check if image exists
                if not image_path.exists():
                    # Try with different extensions
                    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                        img_with_ext = image_path.with_suffix(ext)
                        if img_with_ext.exists():
                            image_path = img_with_ext
                            break
                    else:
                        print(f"Warning: Image {image_path} not found, skipping...")
                        continue
                
                all_image_filenames.append(image_path)
                
                # Get camera pose (transform matrix)
                pose = np.array(camera_data["transform_matrix"], dtype=np.float32)
                all_poses.append(pose)
                all_times.append(frame_time)
                
                # Get camera intrinsics - check camera-specific first, then global
                camera_intrinsics = camera_data.get("intrinsics", cameras_info.get(camera_name, {}))
                
                # Extract intrinsic parameters
                if "fx" in camera_intrinsics and "fy" in camera_intrinsics:
                    fx = camera_intrinsics["fx"]
                    fy = camera_intrinsics["fy"]
                elif "camera_angle_x" in camera_intrinsics:
                    # Load image to get dimensions for FOV calculation
                    sample_image = Image.open(image_path)
                    width, height = sample_image.size
                    camera_angle_x = camera_intrinsics["camera_angle_x"]
                    fx = fy = 0.5 * width / np.tan(0.5 * camera_angle_x)
                else:
                    # Default values if not specified
                    sample_image = Image.open(image_path)
                    width, height = sample_image.size
                    fx = fy = width * 0.7  # Reasonable default
                
                cx = camera_intrinsics.get("cx", sample_image.size[0] / 2.0)
                cy = camera_intrinsics.get("cy", sample_image.size[1] / 2.0)
                
                all_fx.append(fx)
                all_fy.append(fy)
                all_cx.append(cx)
                all_cy.append(cy)
                
                # Get image dimensions
                if 'sample_image' not in locals():
                    sample_image = Image.open(image_path)
                width, height = sample_image.size
                all_widths.append(width)
                all_heights.append(height)
                
                # Handle distortion parameters
                distortion = camera_intrinsics.get("distortion", {})
                if distortion:
                    k1 = distortion.get("k1", 0.0)
                    k2 = distortion.get("k2", 0.0)
                    k3 = distortion.get("k3", 0.0)
                    k4 = distortion.get("k4", 0.0)
                    p1 = distortion.get("p1", 0.0)
                    p2 = distortion.get("p2", 0.0)
                    all_distortion_params.append([k1, k2, k3, k4, p1, p2])
                    all_camera_types.append(CameraType.OPENCV)
                else:
                    all_distortion_params.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    all_camera_types.append(CameraType.PERSPECTIVE)
        
        if not all_image_filenames:
            raise ValueError("No valid images found")
        
        # Convert to tensors
        num_images = len(all_image_filenames)
        fx_tensor = torch.tensor(all_fx, dtype=torch.float32)
        fy_tensor = torch.tensor(all_fy, dtype=torch.float32)
        cx_tensor = torch.tensor(all_cx, dtype=torch.float32)
        cy_tensor = torch.tensor(all_cy, dtype=torch.float32)
        height_tensor = torch.tensor(all_heights, dtype=torch.int32)
        width_tensor = torch.tensor(all_widths, dtype=torch.int32)
        
        # Convert poses to torch tensors (assume 4x4, take 3x4)
        poses_array = np.array(all_poses)
        if poses_array.shape[-1] == 4 and poses_array.shape[-2] == 4:
            poses_tensor = torch.from_numpy(poses_array[:, :3, :4]).float()
        else:
            poses_tensor = torch.from_numpy(poses_array).float()
        
        # Handle distortion parameters
        distortion_tensor = None
        if any(any(dist) for dist in all_distortion_params):
            distortion_tensor = torch.tensor(all_distortion_params, dtype=torch.float32)
        
        # Assume all cameras have same type for simplicity
        camera_type = all_camera_types[0] if all_camera_types else CameraType.PERSPECTIVE
        
        # Create times tensor for temporal data
        times_tensor = torch.tensor(all_times, dtype=torch.float32)
        
        # Create cameras object
        cameras = Cameras(
            fx=fx_tensor,
            fy=fy_tensor,
            cx=cx_tensor,
            cy=cy_tensor,
            height=height_tensor,
            width=width_tensor,
            camera_to_worlds=poses_tensor,
            camera_type=camera_type,
            distortion_params=distortion_tensor,
            times=times_tensor,
        )
        
        # Apply auto-scaling if enabled
        if self.config.auto_scale_poses:
            cameras = self._auto_scale_poses(cameras)
        
        # Split data into train/eval
        indices = np.arange(num_images)
        if split == "train":
            num_train = int(num_images * self.config.train_split_fraction)
            indices = indices[:num_train]
        else:  # eval/test
            num_train = int(num_images * self.config.train_split_fraction)
            indices = indices[num_train:]
        
        # Filter data for the split
        image_filenames = [all_image_filenames[i] for i in indices]
        cameras = cameras[indices]
        
        # Check for mask files (sky masks)
        mask_filenames = None
        mask_dir = self.config.data / "masks"
        if mask_dir.exists():
            mask_filenames = []
            for img_path in image_filenames:
                # Extract frame number and camera name from path
                # Expected: images/000/CAM_FRONT.jpg -> masks/000/CAM_FRONT.png
                relative_path = img_path.relative_to(self.config.data / "images")
                mask_path = mask_dir / relative_path.with_suffix(".png")
                
                if mask_path.exists():
                    mask_filenames.append(mask_path)
                else:
                    mask_filenames.append(None)
        
        # Check for depth files
        depth_filenames = None
        depth_dir = self.config.data / "depths"
        if depth_dir.exists():
            depth_filenames = []
            for img_path in image_filenames:
                # Extract frame number and camera name from path
                # Expected: images/000/CAM_FRONT.jpg -> depths/000/CAM_FRONT.npz
                relative_path = img_path.relative_to(self.config.data / "images")
                depth_path = depth_dir / relative_path.with_suffix(".npz")
                
                if depth_path.exists():
                    depth_filenames.append(depth_path)
                else:
                    depth_filenames.append(None)
        
        # Create scene box - for Street Gaussians, use larger scale for outdoor scenes
        scene_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-scene_scale, -scene_scale, -scene_scale/2],  # Smaller vertical range
                 [scene_scale, scene_scale, scene_scale]],
                dtype=torch.float32,
            )
        )
        
        # Prepare metadata
        metadata = {
            "depth_filenames": depth_filenames,
            "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            "camera_names": self.config.cameras_to_use,
        }
        
        # Add any custom metadata from your data format
        if "metadata" in meta:
            metadata.update(meta["metadata"])
        
        # Create and return DataparserOutputs
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=meta.get("alpha_color"),  # Background color if available
            scene_box=scene_box,
            mask_filenames=mask_filenames,
            metadata=metadata,
            dataparser_scale=self.config.scale_factor,
        )
        
        return dataparser_outputs
    
    def _auto_scale_poses(self, cameras: Cameras) -> Cameras:
        """Automatically scale camera poses to fit in a unit bounding box.
        
        Args:
            cameras: Input cameras object.
            
        Returns:
            Scaled cameras object.
        """
        poses = cameras.camera_to_worlds
        positions = poses[:, :3, 3]
        
        # Calculate bounding box of camera positions
        min_pos = positions.min(dim=0)[0]
        max_pos = positions.max(dim=0)[0]
        
        # Calculate scale to fit in unit box
        scale = 1.0 / (max_pos - min_pos).max()
        
        # Center the poses
        center = (min_pos + max_pos) / 2
        
        # Apply scaling and centering
        scaled_poses = poses.clone()
        scaled_poses[:, :3, 3] = (positions - center) * scale
        
        # Update cameras
        cameras.camera_to_worlds = scaled_poses
        
        return cameras
    
    def _apply_coordinate_transform(self, poses: torch.Tensor) -> torch.Tensor:
        """Apply coordinate system transformations if needed.
        
        This method can be used to convert between different coordinate systems
        (e.g., OpenCV to OpenGL, or custom coordinate systems).
        
        Args:
            poses: Input camera poses [N, 3, 4].
            
        Returns:
            Transformed poses [N, 3, 4].
        """
        # Example: Convert from OpenCV to OpenGL coordinate system
        # transform = torch.tensor([
        #     [1, 0, 0, 0],
        #     [0, -1, 0, 0], 
        #     [0, 0, -1, 0],
        #     [0, 0, 0, 1]
        # ], dtype=poses.dtype, device=poses.device)
        # 
        # # Apply transform
        # poses_4x4 = torch.cat([poses, torch.zeros_like(poses[:, :1])], dim=1)
        # poses_4x4[:, 3, 3] = 1
        # transformed = transform @ poses_4x4
        # return transformed[:, :3]
        
        return poses