import typer
import os
from pathlib import Path
from typing import Optional
import logging
from rich.console import Console
from rich.logging import RichHandler
import hydra
from omegaconf import DictConfig, OmegaConf
import json

from data.nuscenes_loader import NuScenesLoader
from model.emma import ClaudeEMMA
from visualization.visualizer import EMMAVisualizer
from utils.logger import setup_logger

app = typer.Typer(help="EMMA-Claude: Autonomous Driving with Claude API")
console = Console()

def load_config() -> DictConfig:
    """Load configuration from yaml file"""
    with hydra.initialize(version_base=None, config_path="../../config"):
        cfg = hydra.compose(config_name="config")
    return cfg

@app.command()
def predict(
    sample_token: str = typer.Argument(..., help="NuScenes sample token to process"),
    output_dir: Optional[Path] = typer.Option("outputs/debug", help="Directory to save outputs"),
    save_visualization: bool = typer.Option(True, help="Save visualization of predictions"),
):
    """Process a single samplve and generate predictions"""
    # Load config and setup logging    
    
    cfg = load_config()
    logger = logging.getLogger(__name__)

    console.print(f"Processing sample: {sample_token}")
    console.print(f"Output directory: {output_dir}")
    console.print(f"isualizationo: {save_visualization}")
    console.print(f"Loaded configuration: {OmegaConf.to_yaml(cfg)}")

    try:
        # Initialize components
        nusc_loader = NuScenesLoader(
            dataroot=cfg.data.nuscenes_root,
            version=cfg.data.version
        )
        
        emma = ClaudeEMMA(
            api_key=cfg.model.api_key,
            model_version=cfg.model.version
        )
        
        visualizer = EMMAVisualizer(cfg.visualization)
        
        # Process sample
        with console.status("Processing sample..."):
            data = nusc_loader.get_sample_data(sample_token)
            prediction = emma.predict_trajectory(
                camera_images=data.images,
                ego_history=data.ego_history,
                command=data.command
            )
        
        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Save prediction
            prediction_path = output_dir / f"{sample_token}_prediction.json"
            with open(prediction_path, 'w') as f:
                json.dump(prediction, f, indent=2)
            
            # Save visualization
            if save_visualization:
                vis_path = output_dir / f"{sample_token}_visualization.png"
                visualizer.visualize_prediction(
                    front_image=data.images['CAM_FRONT'],
                    prediction=prediction,
                    ground_truth=data.ground_truth,
                    save_path=vis_path
                )
            
            logger.info(f"Results saved to: {output_dir}")
        
        # Display results summary
        logger.info("\n[bold]Prediction Summary:[/bold]")
        logger.info(f"Scene Description: {prediction['scene_description']}")
        logger.info(f"Number of critical objects: {len(prediction['critical_objects'])}")
        
    except Exception as e:
        logger.error(f"Error processing sample: {e}", exc_info=True)
        raise typer.Exit(code=1)

@app.command()
def evaluate(
    num_samples: int = typer.Option(10, help="Number of samples to evaluate"),
    output_dir: Optional[Path] = typer.Option("outputs/evaluation", help="Directory to save results"),
):
    """Run evaluation on multiple samples"""
    # Load config and setup logging
    cfg = load_config()
    setup_logger(cfg.logging)
    logger = logging.getLogger(__name__)

    logger.info(f"Evaluating {num_samples} samples")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Loaded configuration: {OmegaConf.to_yaml(cfg)}")
    
    try:
        # Initialize components
        nusc_loader = NuScenesLoader(
            dataroot=cfg.data.nuscenes_root,
            version=cfg.data.version
        )
        
        emma = ClaudeEMMA(
            api_key=cfg.model.api_key,
            model_version=cfg.model.version
        )
        
        # Run evaluation
        with console.status(f"Evaluating {num_samples} samples...") as status:
            metrics = emma.evaluate_samples(
                nusc_loader=nusc_loader,
                num_samples=num_samples,
                output_dir=output_dir
            )
            
            status.update("Saving results...")
            
            if output_dir:
                metrics_path = Path(output_dir) / "evaluation_metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
        
        # Display results
        logger.info("\n[bold]Evaluation Results:[/bold]")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise typer.Exit(code=1)

@app.command()
def visualize(
    sample_token: str = typer.Argument(..., help="NuScenes sample token to visualize"),
    prediction_path: Path = typer.Argument(..., help="Path to prediction JSON file"),
    output_path: Optional[Path] = typer.Option(
        "outputs/visualization/vis.png",
        help="Path to save visualization"
    ),
):
    """Visualize predictions for a sample"""
    # Load config and setup logging
    cfg = load_config()
    setup_logger(cfg.logging)
    logger = logging.getLogger(__name__)

    logger.info(f"Visualizing sample: {sample_token}")
    logger.info(f"Prediction path: {prediction_path}")
    logger.info(f"Output path: {output_path}")
    
    try:
        # Load prediction
        with open(prediction_path) as f:
            prediction = json.load(f)
        
        # Initialize components
        nusc_loader = NuScenesLoader(
            dataroot=cfg.data.nuscenes_root,
            version=cfg.data.version
        )
        
        visualizer = EMMAVisualizer(cfg.visualization)
        
        # Get sample data
        data = nusc_loader.get_sample_data(sample_token)
        
        # Create visualization
        visualizer.visualize_prediction(
            front_image=data.images['CAM_FRONT'],
            prediction=prediction,
            ground_truth=data.ground_truth,
            save_path=output_path
        )
        
        if output_path:
            console.print(f"Visualization saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}", exc_info=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()