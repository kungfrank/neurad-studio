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

"""Custom data parser for your own data format."""

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
class MyDataParserConfig(DataParserConfig):
    """Configuration for custom data parser."""
    
    _target: Type = field(default_factory=lambda: MyDataParser)
    """Target class to instantiate."""
    
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    
    orientation_method: str = "pca"
    """The method to use for orientation. Options: pca, up, vertical, none."""
    
    center_method: str = "poses"
    """The method to use to center the poses. Options: poses, focus, none."""
    
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""


@dataclass  
class MyDataParser(DataParser):
    """Custom Data Parser.
    
    This parser is designed to load your custom data format. You should modify the 
    _generate_dataparser_outputs method to match your specific data structure.
    
    Expected data structure:
    data/
    ├── transforms.json  # Contains camera poses and intrinsics
    ├── images/          # Directory containing images
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ...
    └── masks/ (optional) # Directory containing mask images
        ├── mask_001.png
        ├── mask_002.png
        └── ...
    """

    config: MyDataParserConfig
    includes_time: bool = False  # Set to True if your data includes time information

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
        
        # Extract camera intrinsics
        if "camera_angle_x" in meta:
            # If field of view is provided
            camera_angle_x = meta["camera_angle_x"]
        elif "fl_x" in meta:
            # If focal length is provided directly
            fl_x = meta["fl_x"]
            fl_y = meta.get("fl_y", fl_x)
        else:
            raise ValueError("Camera intrinsics not found in transforms.json")
        
        # Extract image information
        frames = meta["frames"]
        image_filenames = []
        poses = []
        
        # Process each frame
        for frame in frames:
            # Get image filename
            fname = frame["file_path"]
            if not fname.startswith("/"):
                fname = self.config.data / fname
            else:
                fname = Path(fname)
            
            # Check if image exists
            if not fname.exists():
                # Try with different extensions
                for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                    fname_with_ext = fname.with_suffix(ext)
                    if fname_with_ext.exists():
                        fname = fname_with_ext
                        break
                else:
                    print(f"Warning: Image {fname} not found, skipping...")
                    continue
            
            image_filenames.append(fname)
            
            # Get camera pose (transform matrix)
            pose = np.array(frame["transform_matrix"])
            poses.append(pose)
        
        poses = np.array(poses).astype(np.float32)
        image_filenames = [Path(f) for f in image_filenames]
        
        # Load a sample image to get dimensions
        if image_filenames:
            sample_image = Image.open(image_filenames[0])
            image_width, image_height = sample_image.size
        else:
            raise ValueError("No valid images found")
        
        # Create camera intrinsics
        if "camera_angle_x" in meta:
            # Calculate focal length from field of view
            focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)
            fx = fy = focal_length
        else:
            fx = fl_x
            fy = fl_y
            
        cx = meta.get("cx", image_width / 2.0)
        cy = meta.get("cy", image_height / 2.0)
        
        # Handle distortion parameters (if available)
        distortion_params = None
        camera_type = CameraType.PERSPECTIVE
        
        if "k1" in meta or "k2" in meta:
            # Radial distortion parameters
            k1 = meta.get("k1", 0.0)
            k2 = meta.get("k2", 0.0)
            k3 = meta.get("k3", 0.0)
            k4 = meta.get("k4", 0.0)
            p1 = meta.get("p1", 0.0)
            p2 = meta.get("p2", 0.0)
            distortion_params = torch.tensor([k1, k2, k3, k4, p1, p2], dtype=torch.float32)
            camera_type = CameraType.OPENCV
        
        # Create camera intrinsics tensor
        num_images = len(image_filenames)
        fx_tensor = torch.full((num_images,), fx, dtype=torch.float32)
        fy_tensor = torch.full((num_images,), fy, dtype=torch.float32)
        cx_tensor = torch.full((num_images,), cx, dtype=torch.float32)
        cy_tensor = torch.full((num_images,), cy, dtype=torch.float32)
        
        # Create height and width tensors
        height_tensor = torch.full((num_images,), image_height, dtype=torch.int32)
        width_tensor = torch.full((num_images,), image_width, dtype=torch.int32)
        
        # Handle distortion
        if distortion_params is not None:
            distortion_params = distortion_params.repeat(num_images, 1)
        
        # Convert poses to torch tensors
        poses_tensor = torch.from_numpy(poses[:, :3, :4]).float()
        
        # Apply transforms if needed (e.g., coordinate system conversion)
        # poses_tensor = self._apply_coordinate_transform(poses_tensor)
        
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
            distortion_params=distortion_params,
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
        image_filenames = [image_filenames[i] for i in indices]
        cameras = cameras[indices]
        
        # Check for mask files
        mask_filenames = None
        mask_dir = self.config.data / "masks"
        if mask_dir.exists():
            mask_filenames = []
            for fname in image_filenames:
                # Create corresponding mask filename
                mask_fname = mask_dir / fname.name.replace(fname.suffix, ".png")
                if mask_fname.exists():
                    mask_filenames.append(mask_fname)
                else:
                    mask_filenames.append(None)  # No mask for this image
        
        # Create scene box
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-self.config.scene_scale, -self.config.scene_scale, -self.config.scene_scale],
                 [self.config.scene_scale, self.config.scene_scale, self.config.scene_scale]],
                dtype=torch.float32,
            )
        )
        
        # Prepare metadata
        metadata = {
            "depth_filenames": None,  # Add depth filenames if available
            "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
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