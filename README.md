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

1. First, install system dependencies and set up the Python environment:

```bash
# Make the setup script executable
chmod +x scripts/setup.sh

# Run the setup script
./scripts/setup.sh
```

2. Copy the environment file and configure your credentials:
```bash
cp .env.example .env
```

3. Allow direnv:
```bash
direnv allow
```

If you prefer to install dependencies manually:

1. Install system dependencies (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    python3-matplotlib
```

2. Create and activate virtual environment:
```bash
uv venv --python python3.10
source .venv/bin/activate
```

3. Install Python dependencies:
```bash
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

## Development Environment

This project uses `direnv` to automatically manage environment variables and virtual environments. 

### Prerequisites

1. Install direnv:
```bash
# On macOS
brew install direnv

# On Ubuntu/Debian
sudo apt-get install direnv

# On Fedora
sudo dnf install direnv
```

2. Add direnv hook to your shell configuration:

For bash (`~/.bashrc`):
```bash
eval "$(direnv hook bash)"
```

For zsh (`~/.zshrc`):
```zsh
eval "$(direnv hook zsh)"
```

For fish (`~/.config/fish/config.fish`):
```fish
direnv hook fish | source
```

3. Allow direnv in the project directory:
```bash
direnv allow
```

The included `.envrc` will automatically:
- Create and activate a Python virtual environment using `uv`
- Set up the PYTHONPATH
- Load environment variables from `.env`
- Configure development paths

Note: Make sure to copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

## Dataset Setup

This project uses the nuScenes dataset. There are different versions available:

## Dataset Setup

This project uses the nuScenes image data for autonomous driving predictions. For development and testing, we recommend using the mini dataset (~4GB).

### Required Data Structure

After downloading, your data directory should look like this:
```
/data/sets/nuimages/
    samples/     - Sensor data for keyframes (annotated images)
    sweeps/      - Sensor data for intermediate frames (unannotated images)
    v1.0-mini/   - JSON tables with metadata and annotations
```

### Setup Instructions

1. Create an account at [nuScenes website](https://www.nuscenes.org/nuscenes#download) and accept the Terms of Use.

2. Download the following files for the mini set:
   - `v1.0-mini` (metadata and annotations)
   - `samples` (keyframe images)
   - `sweeps` (intermediate frame images)

3. Extract the archives to your data directory without overwriting folders that occur in multiple archives.

4. Update your `.env` file with the dataset path:
```bash
NUSCENES_DATAROOT={WORSPACE_DIR}/data/sets/nuimages
```

### Verifying the Installation

Install and test the nuscenes-devkit:
```bash
# Install devkit
uv pip install nuscenes-devkit

# Verify setup (in python)
from nuscenes import NuImages
nusc = NuImages(version='v1.0-mini', dataroot='{WORSPACE_DIR}/data/sets/nuimages', verbose=True, lazy=True)
```

Note: While the full nuScenes dataset includes lidar, radar, and map data, this project focuses only on the image data for Claude-based predictions.

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