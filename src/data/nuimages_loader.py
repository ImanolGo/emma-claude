from pathlib import Path
from typing import Dict, List, Optional, Iterator
import numpy as np
from nuimages import NuImages
import logging
from PIL import Image
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SampleData:
    token: str
    images: Dict[str, np.ndarray]
    ego_history: np.ndarray
    command: str
    ground_truth: Optional[Dict] = None
    scene_token: Optional[str] = None

class NuImagesLoader:
    """Handles loading and preprocessing of nuImages data"""
    
    def __init__(self, dataroot: str, version: str = 'v1.0-mini'):
        self.dataroot = Path(dataroot)
        if not self.dataroot.exists():
            raise ValueError(f"NuImages dataroot not found: {dataroot}")
            
        self.nuim = NuImages(dataroot=str(self.dataroot), version=version, verbose=True, lazy=True)
        self.camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
        
        logger.info(f"Initialized NuImages loader with {len(self.nuim.sample)} samples")

    def get_samples(self, num_samples: Optional[int] = None) -> Iterator[SampleData]:
        """Get iterator over samples"""
        samples = self.nuim.sample[:num_samples] if num_samples else self.nuim.sample
        for sample in samples:
            try:
                yield self.get_sample_data(sample['token'])
            except Exception as e:
                logger.warning(f"Error loading sample {sample['token']}: {e}")
                continue

    def get_sample_data(self, sample_token: str) -> SampleData:
        """Get all relevant data for a sample"""
        sample = self.nuim.get('sample', sample_token)
        
        # Get all sample data tokens
        sd_tokens = self.nuim.get_sample_content(sample_token)
        if sd_tokens is None:
            raise ValueError(f"No sample content found for token {sample_token}")
        
        # Load images
        images = {}
        key_camera_token = sample['key_camera_token']
        cam_data = self.nuim.get('sample_data', key_camera_token)
        sensor = self.nuim.shortcut('sample_data', 'sensor', key_camera_token)
        
        if sensor['channel'] == 'CAM_FRONT':
            img_path = self.dataroot / cam_data['filename']
            if img_path.exists():
                images['CAM_FRONT'] = np.array(Image.open(img_path))
        
        # Get ego pose history
        ego_history = self._get_ego_history(sample, sd_tokens)
        
        # Get routing command
        command = self._get_routing_command(sample)
        
        # Get ground truth future trajectory
        ground_truth = self._get_ground_truth(sample, sd_tokens)

        return SampleData(
            token=sample_token,
            images=images,
            ego_history=ego_history,
            command=command,
            ground_truth=ground_truth,
            scene_token=sample['log_token']
        )

    def _get_ego_history(self, sample: Dict, sd_tokens: Dict) -> np.ndarray:
        """Get ego vehicle pose history"""
        history = []
        key_camera_token = sample['key_camera_token']
        current_sd = self.nuim.get('sample_data', key_camera_token)
        
        # Get past 6 frames (3 seconds at 2Hz)
        for _ in range(6):
            if not current_sd['prev']:
                break
                
            current_sd = self.nuim.get('sample_data', current_sd['prev'])
            ego_pose = self.nuim.get('ego_pose', current_sd['ego_pose_token'])
            
            history.append([
                ego_pose['translation'][0],
                ego_pose['translation'][1]
            ])
            
        return np.array(history[::-1])  # Reverse to get chronological order

    def _get_routing_command(self, sample: Dict) -> str:
        """Extract routing command from log data"""
        log = self.nuim.get('log', sample['log_token'])
        
        # This is a simplified version - in practice you'd need to analyze
        # the ego pose data to determine the appropriate command
        return "go_straight"  # Default command

    def _get_ground_truth(self, sample: Dict, sd_tokens: Dict) -> Dict:
        """Get ground truth future trajectory"""
        future_positions = []
        key_camera_token = sample['key_camera_token']
        current_sd = self.nuim.get('sample_data', key_camera_token)
        
        # Get next 6 frames (3 seconds at 2Hz)
        for _ in range(6):
            if not current_sd['next']:
                break
                
            current_sd = self.nuim.get('sample_data', current_sd['next'])
            ego_pose = self.nuim.get('ego_pose', current_sd['ego_pose_token'])
            
            future_positions.append([
                ego_pose['translation'][0],
                ego_pose['translation'][1]
            ])
        
        return {
            'trajectory': np.array(future_positions),
            'timestamps': np.arange(len(future_positions)) * 0.5  # 2Hz sampling
        }

    def get_annotations(self, sample_token: str) -> Dict:
        """Get object and surface annotations for a sample"""
        object_tokens, surface_tokens = self.nuim.list_anns(sample_token)
        return {
            'object_tokens': object_tokens,
            'surface_tokens': surface_tokens
        }