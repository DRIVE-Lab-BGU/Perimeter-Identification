# Traffic Congestion Perimeter Identification

This thesis introduces a novel approach for identifying and defining the perimeter of traffic congestion, based on an integrated analysis of heat maps and graph representations of the transportation network. To achieve this, the study applies Deep Reinforcement Learning (DRL) algorithms capable of decoding complex spatial and topological information to efficiently delineate congested areas.

## What the project does

* loads a SUMO network (`osm.net.xml`) and vehicle trajectories (`fcd.csv`)
* builds density heatmaps for selected timesteps
* represents junctions as graph nodes with spatial and density features
* trains a PPO agent with a GCN backbone to select perimeter-defining junctions
* generates:

  * convex-hull visualizations
  * final post-processed perimeter images
  * statistics and metrics

## Expected project structure

The Python scripts assume the repository is organized like this:

```text
project\_root/
├── config.yaml
├── single\_run\_pipeline.sh
├── data/
│   └── <City>/
│       ├── osm.net.xml
│       ├── fcd.csv
│       └── city\_config.yaml         # optional city-specific timestep override
├── outputs/
├── scripts/
│   ├── train\_single\_run.py
│   ├── generate\_convexhull.py
│   ├── generate\_final\_images.py
│   └── one\_shot.py
└── src/
    ├── NetworkHeatmap.py
    ├── PI\_env.py
    ├── data\_utils.py
    ├── gcn\_agent.py
    ├── gcn\_env.py
    ├── gcn\_model.py
    └── preprocessing.py
```

## Data requirements

For each city, create a folder under `data/`:

```text
data/<City>/
├── osm.net.xml
├── fcd.csv
└── city\_config.yaml   # optional
```

### Required files

* `osm.net.xml`: SUMO road network file
* `fcd.csv`: vehicle trajectory / floating-car-data file

### Required `fcd.csv` columns

The code uses these columns:

* `timestep\_time`
* `vehicle\_x`
* `vehicle\_y`
* `vehicle\_lane`

### Optional city-specific config

If `data/<City>/city\_config.yaml` exists, it overrides the timestep lists from the main `config.yaml`.

## Prerequisites

- Python 3.10+ recommended
- pip install -r requirements.txt

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install torch numpy pandas scipy matplotlib seaborn pyyaml opencv-python gymnasium networkx sumolib
```

The project imports `sumolib` directly. A PyPI package named `sumolib` is available, and the official SUMO documentation also notes that `sumolib` can come from the SUMO tools directory under `SUMO\_HOME/tools`.

If your environment provides `sumolib` through a local SUMO installation instead of PyPI, set `SUMO\_HOME` and add its `tools` directory to your Python path as described in the SUMO docs.
## Configuration

Main configuration is stored in `config.yaml`.

It controls:

* model architecture
* PPO hyperparameters
* training length
* train/eval timesteps
* density / congestion threshold
* data and output paths

The default output model name is:

```text
outputs/<City>/final\_model.pt
```

## How to run the project

### 1\. Train a model

Run from the repository root:

```bash
python scripts/train\_single\_run.py --city Toronto --config config.yaml
```

This will:

* preload heatmaps for the selected train/eval timesteps
* train the GCN+PPO agent
* save the trained model
* save evaluation CSV files

### 2\. Generate convex-hull images

```bash
python scripts/generate\_convexhull.py --city Toronto --config config.yaml --set eval
```

Options for `--set`:

* `eval` - evaluation timesteps only
* `train` - training timesteps only
* `all` - both training and evaluation timesteps

### 3\. Generate final post-processing images

```bash
python scripts/generate\_final\_images.py --city Toronto --config config.yaml --set eval
```

### 4\. Compute final metrics

```bash
python scripts/one\_shot.py --city Toronto --config config.yaml
```

This exports the final comparison table and per-timestep edge details.

## Run the full pipeline

You can run the whole workflow with:

```bash
bash single\_run\_pipeline.sh --city Toronto --config config.yaml
```

Pipeline steps:

1. training
2. convex-hull image generation
3. final post-processing image generation
4. metric calculation

## Useful command variations

### Evaluate a city using a model trained on another city

Generate images with a model from a different city:

```bash
python scripts/generate\_convexhull.py \\
  --city Washington \\
  --model-city Toronto \\
  --config config.yaml \\
  --set eval
```

```bash
python scripts/generate\_final\_images.py \\
  --city Washington \\
  --model-city Toronto \\
  --config config.yaml \\
  --set eval
```

```bash
python scripts/one\_shot.py \\
  --city Washington \\
  --model-city Toronto \\
  --config config.yaml
```

### Skip training and only run post-training stages

```bash
bash single\_run\_pipeline.sh --city Toronto --config config.yaml --skip-train
```

## Outputs

Typical outputs are written under:

```text
outputs/<City>/
```

Examples:

```text
outputs/<City>/
├── final\_model.pt
├── train\_evaluation\_data.csv
├── test\_evaluation\_data.csv
├── train\_set/
│   ├── convexhull/
│   └── convexhull\_binary/
├── evaluation\_set/
│   ├── convexhull/
│   ├── convexhull\_binary/
│   └── post\_processing/
└── metrics/
    ├── metrics\_final\_comparison.csv
    └── edge\_details\_timestep\_<timestep>.csv
```

The preprocessing cache is stored inside each city's data folder:

```text
data/<City>/cache/
```

## Notes and caveats

* Run commands from the **repository root**.
* In practice, you should always pass `--city`, because several scripts use `args.city` directly instead of consistently falling back to `default\_city`.
* Cached timestep files are reused automatically on later runs.

