import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Arrow, Circle, Polygon
from typing import Dict, Optional, List, Tuple
import cv2
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EMMAVisualizer:
    """Visualization tools for EMMA predictions"""
    
    def __init__(self):
        self.colors = {
            'trajectory': '#2ecc71',    # Green for predicted trajectory
            'gt_trajectory': '#e74c3c',  # Red for ground truth
            'vehicle': '#3498db',       # Blue for vehicles
            'pedestrian': '#f1c40f',    # Yellow for pedestrians
            'cyclist': '#9b59b6',       # Purple for cyclists
            'ego': '#2c3e50'           # Dark blue for ego vehicle
        }
        
        # Default visualization settings
        self.max_range = 50  # meters
        self.fig_size = (20, 10)
        
    def visualize_prediction(self,
                           front_image: np.ndarray,
                           prediction: Dict,
                           ground_truth: Optional[Dict] = None,
                           save_path: Optional[Path] = None) -> None:
        """Create comprehensive visualization of EMMA's prediction"""
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=self.fig_size)
            gs = plt.GridSpec(2, 2, figure=fig)
            
            # Camera view with detections (top left)
            ax_camera = fig.add_subplot(gs[0, 0])
            self._plot_camera_view(ax_camera, front_image, prediction)
            
            # Bird's eye view (top right)
            ax_bev = fig.add_subplot(gs[0, 1])
            self._plot_birds_eye_view(ax_bev, prediction, ground_truth)
            
            # Text description (bottom)
            ax_text = fig.add_subplot(gs[1, :])
            self._plot_text_description(ax_text, prediction)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
                logger.info(f"Visualization saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            raise
            
    def _plot_camera_view(self, 
                         ax: plt.Axes, 
                         image: np.ndarray, 
                         prediction: Dict) -> None:
        """Plot camera view with detected objects"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        ax.imshow(image)
        ax.set_title("Front Camera View with Detections")
        
        h, w = image.shape[:2]
        
        # Draw bounding boxes for critical objects
        for obj in prediction['critical_objects']:
            position = obj['position']
            
            # Project 3D position to 2D image space (simplified)
            img_x, img_y = self._project_to_image(position, (w, h))
            
            if img_x is not None and img_y is not None:
                # Draw object marker
                circle = Circle((img_x, img_y), radius=10, 
                              color=self.colors.get(obj['type'], 'white'),
                              alpha=0.6)
                ax.add_patch(circle)
                
                # Add label
                ax.text(img_x + 15, img_y, 
                       f"{obj['type']}: {position[0]:.1f}m",
                       color='white', 
                       backgroundcolor='black')
        
        ax.axis('off')
    
    def _plot_birds_eye_view(self,
                            ax: plt.Axes,
                            prediction: Dict,
                            ground_truth: Optional[Dict] = None) -> None:
        """Plot bird's eye view with trajectories and objects"""
        ax.set_title("Bird's Eye View")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        
        # Set plotting boundaries
        ax.set_xlim([-10, self.max_range])
        ax.set_ylim([-self.max_range/2, self.max_range/2])
        
        # Draw ego vehicle
        ego_rect = Rectangle((-2, -1), 4, 2, 
                           color=self.colors['ego'], 
                           alpha=0.7)
        ax.add_patch(ego_rect)
        
        # Plot predicted trajectory
        trajectory = np.array(prediction['trajectory'])
        ax.plot(trajectory[:, 0], trajectory[:, 1],
                color=self.colors['trajectory'],
                linewidth=2, marker='o',
                label='Predicted Trajectory')
        
        # Plot ground truth if available
        if ground_truth and 'trajectory' in ground_truth:
            gt_trajectory = np.array(ground_truth['trajectory'])
            ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1],
                   color=self.colors['gt_trajectory'],
                   linewidth=2, marker='o',
                   label='Ground Truth')
        
        # Plot critical objects
        for obj in prediction['critical_objects']:
            x, y = obj['position']
            if abs(x) > self.max_range or abs(y) > self.max_range:
                continue
            
            circle = Circle((x, y), radius=1,
                          color=self.colors.get(obj['type'], 'gray'),
                          alpha=0.7)
            ax.add_patch(circle)
            
            # Add velocity arrows if available
            if 'velocity' in obj:
                vx, vy = obj['velocity']
                arrow = Arrow(x, y, vx, vy,
                            width=0.5,
                            color=self.colors.get(obj['type'], 'gray'))
                ax.add_patch(arrow)
        
        ax.grid(True)
        ax.legend()
        ax.set_aspect('equal')
    
    def _plot_text_description(self,
                             ax: plt.Axes,
                             prediction: Dict) -> None:
        """Plot text description and analysis"""
        ax.axis('off')
        
        text = f"""Scene Description:
{prediction['scene_description']}

Critical Objects:
"""
        for obj in prediction['critical_objects']:
            pos = obj['position']
            text += f"- {obj['type']} at position ({pos[0]:.1f}, {pos[1]:.1f})m\n"
        
        if 'reasoning' in prediction:
            text += f"\nReasoning:\n{prediction['reasoning']}"
        
        ax.text(0.05, 0.95, text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=10,
                family='monospace',
                wrap=True)
    
    @staticmethod
    def _project_to_image(position_3d: List[float], 
                         image_size: Tuple[int, int]) -> Tuple[Optional[float], Optional[float]]:
        """
        Simple projection from 3D coordinates to image space
        In practice, you would use proper camera calibration and projection matrices
        """
        x, y = position_3d[:2]
        w, h = image_size
        
        # Simple perspective projection (this is a very basic approximation)
        if x <= 0:  # Behind the camera
            return None, None
            
        # Scale and project
        scale = 50 / (x + 1e-6)  # Avoid division by zero
        img_x = w/2 + y * scale * w/2
        img_y = h/2 - scale * h/4
        
        # Check if point is within image bounds
        if 0 <= img_x < w and 0 <= img_y < h:
            return img_x, img_y
        return None, None