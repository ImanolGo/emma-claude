# EMMA-Claude

An implementation of EMMA (End-to-End Multimodal Model for Autonomous Driving) using the Claude API, based on the [EMMA paper](https://arxiv.org/abs/2410.23262). This implementation uses Claude for trajectory prediction and scene understanding instead of the original Gemini model.

## Features

- End-to-end autonomous driving trajectory prediction
- Integration with nuScenes dataset
- Real-time visualization tools for predictions
- Scene understanding and critical object detection
- Comprehensive evaluation metrics
- Command-line interface for different operations


## Installation

This project uses `uv` for dependency management. First, install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the development environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/emma-claude.git
cd emma-claude

# Create and activate virtual environment using uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required environment variables:
- `ANTHROPIC_API_KEY`: Your Claude API key
- `NUSCENES_DATAROOT`: Path to your nuScenes dataset

## Usage

### Command Line Interface

1. Process a single sample:
```bash
python -m src.scripts.cli predict sample_token_123 --output-dir outputs
```

2. Run evaluation:
```bash
python -m src.scripts.cli evaluate --num-samples 100 --output-dir eval_results
```

3. Visualize predictions:
```bash
python -m src.scripts.cli visualize sample_token_123 predictions/sample_123.json
```

### Python API

```python
from src.model.emma import ClaudeEMMA
from src.data.nuscenes_loader import NuScenesLoader
from src.visualization.visualizer import EMMAVisualizer

# Initialize components
emma = ClaudeEMMA(api_key="your-api-key")
nusc_loader = NuScenesLoader(dataroot="path/to/nuscenes")
visualizer = EMMAVisualizer()

# Process a sample
sample_data = nusc_loader.get_sample_data(sample_token)
prediction = emma.predict_trajectory(
    camera_images=sample_data.images,
    ego_history=sample_data.ego_history,
    command=sample_data.command
)

# Visualize results
visualizer.visualize_prediction(
    front_image=sample_data.images['CAM_FRONT'],
    prediction=prediction,
    ground_truth=sample_data.ground_truth,
    save_path="prediction.png"
)
```

## Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black src/ tests/
ruff check src/ tests/
mypy src/
```

## Evaluation Metrics

The implementation includes several metrics for evaluating prediction quality:

- Average Displacement Error (ADE)
- Final Displacement Error (FDE)
- Trajectory Smoothness
- Scene Understanding Accuracy

## Visualization

The visualization module provides:

1. Camera View:
   - Object detections with distance annotations
   - Critical object highlighting

2. Bird's Eye View:
   - Predicted trajectory
   - Ground truth trajectory (when available)
   - Critical objects with velocity vectors
   - Ego vehicle position and orientation

3. Text Description:
   - Scene analysis
   - Critical object list
   - Reasoning explanation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this implementation in your research, please cite both the original EMMA paper and this implementation:

```bibtex
@article{hwang2024emma,
  title={EMMA: End-to-End Multimodal Model for Autonomous Driving},
  author={Hwang, Jyh-Jing and others},
  journal={arXiv preprint arXiv:2410.23262},
  year={2024}
}
```