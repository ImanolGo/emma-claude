import numpy as np
from typing import Dict, List, Tuple
import numpy.typing as npt

def calculate_metrics(prediction: Dict, 
                     ground_truth: Dict) -> Dict[str, float]:
    """
    Calculate evaluation metrics for predictions
    
    Args:
        prediction: Dictionary containing predicted trajectory
        ground_truth: Dictionary containing ground truth trajectory
    
    Returns:
        Dictionary of metrics
    """
    pred_traj = np.array(prediction['trajectory'])
    gt_traj = np.array(ground_truth['trajectory'])
    
    metrics = {
        'ade': average_displacement_error(pred_traj, gt_traj),
        'fde': final_displacement_error(pred_traj, gt_traj),
        'smoothness': trajectory_smoothness(pred_traj)
    }
    
    return metrics

def average_displacement_error(prediction: npt.NDArray, 
                             ground_truth: npt.NDArray) -> float:
    """
    Calculate Average Displacement Error (ADE)
    
    Args:
        prediction: Array of shape (T, 2) containing predicted positions
        ground_truth: Array of shape (T, 2) containing ground truth positions
    
    Returns:
        Average displacement error in meters
    """
    if len(prediction) != len(ground_truth):
        raise ValueError("Prediction and ground truth must have same length")
        
    return float(np.mean(np.sqrt(np.sum((prediction - ground_truth) ** 2, axis=1))))

def final_displacement_error(prediction: npt.NDArray, 
                           ground_truth: npt.NDArray) -> float:
    """
    Calculate Final Displacement Error (FDE)
    
    Args:
        prediction: Array of shape (T, 2) containing predicted positions
        ground_truth: Array of shape (T, 2) containing ground truth positions
    
    Returns:
        Final displacement error in meters
    """
    if len(prediction) != len(ground_truth):
        raise ValueError("Prediction and ground truth must have same length")
        
    return float(np.sqrt(np.sum((prediction[-1] - ground_truth[-1]) ** 2)))

def trajectory_smoothness(trajectory: npt.NDArray) -> float:
    """
    Calculate trajectory smoothness using second derivatives
    
    Args:
        trajectory: Array of shape (T, 2) containing positions
    
    Returns:
        Smoothness metric (lower is smoother)
    """
    if len(trajectory) < 3:
        return 0.0
        
    # Calculate second derivatives
    second_derivatives = np.diff(trajectory, n=2, axis=0)
    
    # Calculate smoothness as mean squared acceleration
    return float(np.mean(np.sum(second_derivatives ** 2, axis=1)))

def transform_to_ego_frame(points: npt.NDArray,
                          ego_position: npt.NDArray,
                          ego_heading: float) -> npt.NDArray:
    """
    Transform points from global to ego-centric coordinate frame
    
    Args:
        points: Array of shape (N, 2) containing points in global frame
        ego_position: Array of shape (2,) containing ego vehicle position
        ego_heading: Heading angle in radians
    
    Returns:
        Array of points in ego-centric frame
    """
    # Translate to ego position
    points_translated = points - ego_position
    
    # Rotate to ego heading
    c, s = np.cos(-ego_heading), np.sin(-ego_heading)
    rotation_matrix = np.array([[c, -s], [s, c]])
    
    return points_translated @ rotation_matrix.T

def get_trajectory_curvature(trajectory: npt.NDArray) -> npt.NDArray:
    """
    Calculate the curvature at each point of the trajectory
    
    Args:
        trajectory: Array of shape (T, 2) containing positions
    
    Returns:
        Array of curvature values
    """
    # Calculate first derivatives
    dx_dt = np.gradient(trajectory[:, 0])
    dy_dt = np.gradient(trajectory[:, 1])
    
    # Calculate second derivatives
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    
    # Calculate curvature
    curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / \
                (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
                
    return curvature