import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import json
import anthropic
from tqdm import tqdm

from data.nuscenes_loader import NuScenesLoader
from utils.geometry import calculate_metrics

logger = logging.getLogger(__name__)

class ClaudeEMMA:
    def __init__(self, api_key: str, model_version: str = "claude-3-opus-20240229"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_version = model_version
        
        # Load prompts
        self.system_prompt = self._load_prompt("system")
        self.analysis_prompt = self._load_prompt("analysis")
    
    def _load_prompt(self, prompt_type: str) -> str:
        """Load prompt template from prompts directory"""
        prompt_path = Path(__file__).parent / "prompts" / f"{prompt_type}.txt"
        return prompt_path.read_text()
    
    def predict_trajectory(
        self,
        camera_images: Dict[str, np.ndarray],
        ego_history: np.ndarray,
        command: str,
        **kwargs
    ) -> Dict:
        """
        Generate predictions using Claude
        """
        # Prepare image for Claude
        image_data = self._prepare_image(camera_images["CAM_FRONT"])
        
        # Prepare prompt
        prompt = self._format_prompt(
            ego_history=ego_history,
            command=command,
            **kwargs
        )
        
        # Call Claude API
        try:
            message = self.client.messages.create(
                model=self.model_version,
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            # Parse response
            response = json.loads(message.content[0].text)
            return response
            
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            raise
    
    def evaluate_samples(
        self,
        nusc_loader: NuScenesLoader,
        num_samples: int,
        output_dir: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Run evaluation on multiple samples
        """
        metrics_list = []
        
        for sample in tqdm(nusc_loader.get_samples(num_samples)):
            # Get prediction
            prediction = self.predict_trajectory(
                camera_images=sample['images'],
                ego_history=sample['ego_history'],
                command=sample['command']
            )
            
            # Calculate metrics
            metrics = calculate_metrics(
                prediction=prediction,
                ground_truth=sample['ground_truth']
            )
            metrics_list.append(metrics)
            
            # Save results if output_dir provided
            if output_dir:
                self._save_results(
                    sample_token=sample['token'],
                    prediction=prediction,
                    metrics=metrics,
                    output_dir=output_dir
                )
        
        # Aggregate metrics
        return self._aggregate_metrics(metrics_list)