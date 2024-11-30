import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
import typer

from data.nuscenes_loader import NuScenesLoader
from model.emma import ClaudeEMMA
from visualization.visualizer import EMMAVisualizer
from utils.logger import setup_logger

app = typer.Typer()
logger = logging.getLogger(__name__)

@app.command()
def predict(
    sample_token: str = typer.Argument(..., help="NuScenes sample token to process"),
    config_path: str = typer.Option("config/predict.yaml", help="Path to config file"),
    output_dir: Path = typer.Option("outputs", help="Directory to save outputs"),
):
    """
    Process a single sample and generate predictions
    """
    setup_logger()
    logger.info(f"Processing sample: {sample_token}")
    
    # Load config
    with hydra.initialize(config_path="."):
        cfg = hydra.compose(config_name="predict")
    
    # Initialize components
    nusc_loader = NuScenesLoader(
        dataroot=cfg.data.nuscenes_root,
        version=cfg.data.version
    )
    
    emma = ClaudeEMMA(
        api_key=cfg.model.api_key,
        model_version=cfg.model.version
    )
    
    visualizer = EMMAVisualizer()
    
    # Process sample
    try:
        data = nusc_loader.get_sample_data(sample_token)
        prediction = emma.predict_trajectory(
            camera_images=data['images'],
            ego_history=data['ego_history'],
            command=data['command']
        )
        
        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save visualization
        vis_path = output_dir / f"{sample_token}_prediction.png"
        visualizer.visualize_prediction(
            front_image=data['images']['CAM_FRONT'],
            prediction=prediction,
            save_path=vis_path
        )
        
        logger.info(f"Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error processing sample: {e}")
        raise

@app.command()
def evaluate(
    config_path: str = typer.Option("config/evaluate.yaml", help="Path to config file"),
    output_dir: Path = typer.Option("outputs", help="Directory to save outputs"),
    num_samples: int = typer.Option(100, help="Number of samples to evaluate"),
):
    """
    Run evaluation on multiple samples
    """
    setup_logger()
    logger.info(f"Starting evaluation on {num_samples} samples")
    
    # Load config
    with hydra.initialize(config_path="."):
        cfg = hydra.compose(config_name="evaluate")
    
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
    metrics = emma.evaluate_samples(
        nusc_loader=nusc_loader,
        num_samples=num_samples,
        output_dir=output_dir
    )
    
    logger.info(f"Evaluation results: {metrics}")

if __name__ == "__main__":
    app()