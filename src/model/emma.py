import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import json
from anthropic import Anthropic
from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image

from data.nuimages_loader import NuImagesLoader
from utils.geometry import calculate_metrics

logger = logging.getLogger(__name__)

class ClaudeEMMA:
    def __init__(self, api_key: str, model_version: str = "claude-3-5-sonnet-20241022"):
        self.client = Anthropic(api_key=api_key)
        self.model_version = model_version
        
        # Load prompts
        self.system_prompt = self._load_prompt("system")
        self.analysis_prompt = self._load_prompt("analysis")
    
    def _load_prompt(self, prompt_type: str) -> str:
        """Load prompt template from prompts directory"""
        prompt_path = Path(__file__).parent / "prompts" / f"{prompt_type}.txt"
        return prompt_path.read_text()
    
    def _prepare_image(self, image: np.ndarray) -> str:
        """
        Convert numpy image array to base64 string for Claude API
        
        Args:
            image (np.ndarray): Input image as numpy array from PIL.Image
            
        Returns:
            str: Base64 encoded PNG image data
        """
        
        # Convert numpy array to PIL Image
        if image.dtype != np.uint8:
            # Normalize float arrays to 0-255 range
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        
        # Convert to PNG format in memory
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        
        # Get base64 encoded string
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_str

    def _format_prompt(self,
                  ego_history: np.ndarray,
                  command: str,
                  **kwargs) -> str:
        """
        Format the analysis prompt with specific scene data
        
        Args:
            ego_history: Array of shape (T, 2) with past (x,y) positions
            command: High-level routing command (e.g., "go straight")
            **kwargs: Additional scene-specific information
            
        Returns:
            str: Formatted prompt for Claude
        """
        # Format ego history as string
        history_str = "Vehicle history (most recent first):\n"
        for i, (x, y) in enumerate(ego_history):
            time = (len(ego_history) - i - 1) * 0.5  # 2Hz sampling
            history_str += f"t-{time:.1f}s: ({x:.2f}m, {y:.2f}m)\n"
        
        # Replace placeholders in analysis prompt template
        prompt = self.analysis_prompt.format(
            command=command,
            history=history_str
        )
        
        # Add system prompt as context
        full_prompt = f"{self.system_prompt}\n\n{prompt}"
        
        return full_prompt
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate metrics across multiple samples
        
        Args:
            metrics_list: List of dictionaries containing metrics for each sample
                Each dictionary contains metrics like 'ade', 'fde', 'smoothness'
        
        Returns:
            Dict containing aggregated metrics:
                - mean and std for each metric
                - success rate (percentage of valid predictions)
        """
        if not metrics_list:
            return {}
        
        # Initialize aggregated metrics
        aggregated = {}
        
        # Get all metric keys from first sample
        metric_keys = metrics_list[0].keys()
        
        # Calculate statistics for each metric
        for key in metric_keys:
            # Get all values for this metric, filtering out None and invalid values
            values = [m[key] for m in metrics_list if key in m and m[key] is not None]
            
            if values:
                values_array = np.array(values)
                aggregated.update({
                    f"{key}_mean": float(np.mean(values_array)),
                    f"{key}_std": float(np.std(values_array)),
                    f"{key}_min": float(np.min(values_array)),
                    f"{key}_max": float(np.max(values_array)),
                    f"{key}_valid_samples": len(values)
                })
        
        # Calculate overall success rate
        total_samples = len(metrics_list)
        successful_samples = sum(1 for m in metrics_list if all(m.get(key) is not None for key in metric_keys))
        aggregated['success_rate'] = successful_samples / total_samples if total_samples > 0 else 0.0
        
        return aggregated
    
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
        nuim_loader: NuImagesLoader,
        num_samples: int,
        output_dir: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Run evaluation on multiple samples
        """
        metrics_list = []
        
        for sample in tqdm(nuim_loader.get_samples(num_samples)):
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