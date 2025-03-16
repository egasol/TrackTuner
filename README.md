# TrackTuner

**TrackTuner** is a tool for optimizing the parameters of a multi-object tracker. It provides a framework for synthetic data generation, tracking, performance evaluation, parameter optimization, and visualization—ideal for developers and researchers working on advanced tracking algorithms.

---

## Table of Contents

- [Features](#features)
  - [1. Synthetic Reference and Detection Generator](#1-synthetic-reference-and-detection-generator)
  - [2. Tracker Runner](#2-tracker-runner)
  - [3. Performance Evaluator](#3-performance-evaluator)
  - [4. Parameter Optimizer](#4-parameter-optimizer)
  - [5. Visualizer](#5-visualizer)
- [Example Usage Summary](#example-usage-summary)
- [License](#license)
- [Contact](#contact)

---

## Features

### 1. Synthetic Reference and Detection Generator

The **Synthetic Reference and Detection Generator** simulates realistic object tracking data while introducing imperfections (e.g., noise, missing detections, false positives) similar to those observed in real-world detectors. This module consists of two key sub-components:

#### Reference Generator
- **Purpose**: Generate synthetic object tracks that mimic realistic movement across frames.
- **Customizable Parameters**:
  - `--num-frames`: Total number of frames to simulate.
  - `--num-tracks`: Number of unique tracks.
- **Behavior**:
  - Each track covers a random range of frames (with a minimum length constraint).
  - Positions (`x`, `y`, `z`) are initialized within pre-defined bounds (e.g., -10 to 10).

#### Detection Generator
- **Purpose**: Impose noise and imperfections to simulate a real detector.
- **Key Features**:
  - **ID Removal**: Eliminates reference track IDs to mimic the output from traditional object detector.
  - **Random Position Noise**: Perturbs detection positions (`--position-randomize`).
  - **Missing Detections**: Randomly omits detections (`--delete-probability`).
  - **False Positives**: Introduces extra random detections (`--add-probability`).
- **Additional Note**: Global coordinate bounds are computed to ensure that false positives appear within realistic spatial limits.

#### Example Command
```
python annotator.py \
  --output-references references.json \
  --output-detections detections.json \
  --num-frames 100 \
  --num-tracks 10 \
  --position-randomize 0.1 \
  --delete-probability 0.15 \
  --add-probability 0.5
```

---

### 2. Tracker Runner

The **Tracker Runner** processes detections using a multi-object tracker that relies on a Kalman Filter for state estimation and an efficient association algorithm for matching detections to existing tracks. The component manages track creation, updating, and deletion to maintain robust tracking over time.

#### Key Features
- **Kalman Filter for State Estimation**:
  - Maintains a 9-dimensional state vector: `x`, `y`, `z`, `vx`, `vy`, `vz`, `ax`, `ay`, `az`.
  - Predicts and updates object states based on detections.
- **Detection-to-Track Association**:
  - Uses the Hungarian Algorithm (via scipy.optimize.linear_sum_assignment) to minimize the Euclidean distance between predicted tracks and detections.
- **Robust Track Management**:
  - **Track Staging**: Differentiates *Initialized* tracks (new) from *Confirmed* tracks (reliable).
  - **Adaptive Handling**:
    - Creates new tracks for unmatched detections.
    - Removes stale tracks exceeding maximum age or consecutive misses.
  - Maintains a history of positions for smoothing.
- **Customizable Settings**:
  - Configurable parameters include measurement noise, process noise, covariance, distance thresholds, maximum age, minimum hits, and maximum consecutive misses.

#### Example Command
```
python tracker.py \
  --input-detections detections.json \
  --input-parameters parameters.json \
  --output output_tracks.json
```

---

### 3. Performance Evaluator

The **Performance Evaluator** compares the tracker’s output against reference annotations by computing metrics that gauge tracking quality.

#### Key Features
- **Annotation Analysis**:
  - Evaluates each object's lifespan and records the number of frames during which it was successfully tracked.
- **Track Analysis**:
  - Reviews track lifespan, reliability, and associations with reference objects.
- **False Positive Detection**:
  - Identifies tracks that do not correspond to any annotated objects.
- **Metric Computation**:
  - **Tracked Percentage**: Proportion of each object’s lifespan during which it is correctly tracked.
  - **ID Switches**: Number of times a reference object is assigned a different track ID.
  - **Composite Performance Metric**: Custom score combining tracked percentage, ID switches, and false positives.
- **Multi-Metric Evaluation**:
  - Provides average values for tracked percentages, ID switches, and total false positives across objects.

---

### 4. Parameter Optimizer

The **Parameter Optimizer** leverages the Optuna framework to automatically fine-tune the tracker parameters. Through multiple optimization trials, it identifies the best combination of settings that deliver optimal tracking performance.

#### Key Features
- **Automated Hyperparameter Search**:
  - Explores a range of values for measurement noise, process noise, covariance, distance threshold, maximum age, minimum hits, and consecutive misses.
- **Performance-Based Objective**:
  - Minimizes a custom performance metric that reflects tracking accuracy, ID switches, and false positives.
- **Flexible and Scalable**:
  - Processes multiple file pairs of references and detections.
  - Allows users to specify the number of optimization trials.
- **Visualization Insights**:
  - Generates visualizations such as Optimization History and Parameter Importances, helping you interpret the optimization process.

---

### 5. Visualizer

The **Visualizer** produces clear, interactive 3D visualizations of reference tracks, detections, and tracked outputs. These visualizations aid in analyzing tracking performance and debugging the tracking algorithm.

#### Key Features
- **3D Plotting**:
  - Uses Matplotlib's Axes3D for plotting trajectories in 3D space.
  - Supports visualization of:
    - Reference tracks.
    - Detections (without unique IDs).
    - Confirmed tracked outputs (with unique IDs).
- **Color-Coded Tracks**:
  - Automatically assigns distinct colors to different tracks for better differentiation.
- **Temporal Alpha Variation**:
  - Alters the transparency of points based on the frame index to effectively convey temporal progression.
- **Customizable Output**:
  - Configurable input files, plot titles, and DPI (resolution) settings.

#### Example Command
```
python visualizer.py \
  --input-references references.json \
  --input-detections detections.json \
  --input-tracked tracked.json \
  --output visualization.png \
  --dpi 50
```

#### Sample Visualization Output:

![Visualizer Example](media/comparison.png)

---