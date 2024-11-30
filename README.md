# EMMA-Claude

An implementation of EMMA (End-to-End Multimodal Model for Autonomous Driving) using the Claude API, based on the [EMMA paper](https://arxiv.org/abs/2410.23262).

## Features

- End-to-end autonomous driving trajectory prediction
- Integration with nuScenes dataset
- Visualization tools for predictions and analysis
- Scene understanding and critical object detection

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

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies using uv
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required environment variables:
- `ANTHROPIC_API_KEY`: Your Claude API key
- `NUSCENES_DATAROOT`: Path to your nuScenes dataset

## Usage

Basic usage example:

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
    camera_images=sample_data['images'],
    ego_history=sample_data['ego_history'],
    command="go straight"
)

# Visualize results
visualizer.visualize_prediction(
    front_image=sample_data['images']['CAM_FRONT'],
    prediction=prediction,
    save_path='prediction.png'
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
```

## License

MIT License - see LICENSE file for details.