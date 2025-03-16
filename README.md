# TrackTuner

**TrackTuner** is a tool designed to optimize parameters for a multi-object tracker. It offers a comprehensive framework with multiple components for synthetic data generation, tracking, evaluation, optimization, and visualization. 

---

## Features

### 1. Synthetic Reference and Detection Generator

The **Synthetic Reference and Detection Generator** simulates realistic object tracking data and generates detections that mimic imperfections seen in real-world object detectors. It consists of two primary functionalities:

#### Reference Generator
- **Purpose**: Creates synthetic tracks simulating object motion across frames.
- **Key Features**:
  - Allows customization of:
    - Number of frames (`--num-frames`).
    - Number of unique tracks (`--num-tracks`).
  - Each track spans a randomly chosen range of frames (with a minimum length constraint) and includes positions (`x`, `y`, `z`) initialized within defined bounds (e.g., -10 to 10).

#### Detection Generator
- **Purpose**: Introduces imperfections into the synthetic tracks for a realistic simulation.
- **Key Features**:
  - **Removes reference track IDs**: To mimic an actual object detector without tracking.
  - **Random Position Noise**: Perturbs detection positions to reflect detector inaccuracies (`--position-randomize`).
  - **Missing Detections**: Randomly removes detections to simulate missed objects (`--delete-probability`).
  - **False Positives**: Generates additional random detections as noise (`--add-probability`).

The generator also calculates global bounds for `x`, `y`, and `z` coordinates to ensure that false positives are added within appropriate ranges.

---

#### Example Usage

To generate synthetic references and detections:
```bash
python track_generator.py \
  --output-references references.json \
  --output-detections detections.json \
  --num-frames 100 \
  --num-tracks 10 \
  --position-randomize 0.1 \
  --delete-probability 0.15 \
  --add-probability 0.5
```


### 2. Tracker Runner

The **Tracker Runner** component processes detections and applies a multi-object tracker to generate track data. By utilizing a Kalman Filter for state estimation and an association algorithm for matching detections to tracks, the tracker ensures robust and accurate tracking of objects across frames.

---

#### Key Features
1. **Kalman Filter for State Estimation**:
   - Tracks the position, velocity, and acceleration of each object.
   - Maintains a 9-dimensional state vector (`x`, `y`, `z`, `vx`, `vy`, `vz`, `ax`, `ay`, `az`).
   - Updates object states based on detections or predictions.

2. **Detection-to-Track Association**:
   - Uses the Hungarian Algorithm (via `scipy.optimize.linear_sum_assignment`) to match detections to existing tracks.
   - Association is based on minimizing the Euclidean distance between predicted track positions and detection points.

3. **Robust Track Management**:
   - Tracks are categorized into stages:
     - **Initialized**: New tracks created from unassociated detections.
     - **Confirmed**: Tracks with sufficient detection history to ensure reliability.
   - Handles cases such as:
     - Removing tracks with excessive consecutive misses.
     - Adding new tracks for unmatched detections.
   - Ensures track longevity by maintaining a history of positions.

4. **Customizable Tracker Settings**:
   - Supports the following configurable parameters:
     - Measurement noise, process noise, and initial covariance values.
     - Distance threshold for associating detections with tracks.
     - Maximum age for tracks without updates.
     - Minimum hits to confirm a track.
     - Maximum consecutive misses before a track is discarded.

---

#### Example Usage

To run the tracker on a set of detections:
```bash
python tracker_runner.py \
  --input-detections detections.json \
  --input-parameters parameters.json \
  --output output_tracks.json
```

### 3. Performance Evaluator

The **Performance Evaluator** assesses the quality of the tracking algorithm by comparing its output tracks with the reference annotations. It gathers and computes various metrics to evaluate tracking performance, such as tracking accuracy, false positives, and ID switches.

---

## Key Features

1. **Annotation Analysis**:
   - Evaluates each object in the reference annotations.
   - Tracks its lifespan and records how many frames it was successfully tracked.

2. **Track Analysis**:
   - Examines the lifespan, reliability, and association of each track to the corresponding reference object.

3. **False Positive Detection**:
   - Counts tracks that do not correspond to any objects in the annotations.

4. **Metric Computation**:
   - Calculates statistics, including:
     - **Tracked Percentage**: Measures how well objects were tracked across their lifespan.
     - **ID Switches**: Tracks the number of times a reference object switched to a different track.
     - **Performance Metric**: Combines metrics into a single score using a customizable formula.

5. **Multi-Metric Evaluation**:
   - Provides average tracked percentage, average ID switches, and total false positives as separate metrics for detailed insights.

---

#### Example Usage

To evaluate the performance of the tracker:
```bash
python evaluator.py \
  --annotations annotations.json \
  --tracks tracked.json
```

### 4. Parameter Optimizer

The **Parameter Optimizer** component utilizes the Optuna framework to fine-tune the parameters of the tracker. By running multiple optimization trials and evaluating the resulting tracking performance, it determines the combination of parameters that yields the best results for the given dataset.

---

#### Key Features

1. **Automated Parameter Optimization**:
   - Leverages **Optuna** for efficient hyperparameter optimization.
   - Explores a range of parameter values for:
     - Measurement noise, process noise, and covariance.
     - Distance threshold for detection-to-track association.
     - Maximum age, minimum hits, and consecutive misses for track management.

2. **Performance-Based Objective**:
   - The optimizer evaluates tracker performance based on metrics provided by the **Performance Evaluator**.
   - Minimizes a custom performance metric that accounts for tracking accuracy, ID switches, and false positives.

3. **Customizable and Scalable**:
   - Accepts multiple reference and detection files for robust evaluation.
   - Users can define the number of optimization trials to control the depth of parameter tuning.

4. **Visualizations**:
   - Generates visual insights into the optimization process:
     - **Optimization History**: Tracks the improvement of the objective value across trials.
     - **Parameter Importances**: Highlights the most impactful parameters on tracking performance.

---

### 5. Visualizer

The **Visualizer** component generates intuitive and detailed 3D visualizations of reference tracks, detections, and tracked outputs. By plotting the movement of objects over time, it aids in the analysis and debugging of tracking algorithms.

---

#### Key Features

1. **3D Plotting of Tracks**:
   - Visualizes tracks in 3D space using Matplotlib's `Axes3D`.
   - Supports plotting of:
     - Reference tracks.
     - Detections (without unique IDs).
     - Tracked outputs (with unique IDs).

2. **Color-Coded Tracks**:
   - Assigns a unique color to each track ID for easy differentiation.

3. **Temporal Alpha Variation**:
   - Adjusts transparency (alpha) of points based on the frame index to convey temporal progression.

4. **Customizable Outputs**:
   - Allows configuration of input data files, plot titles, and visualization resolution (DPI).

---

#### Example Usage

To generate a visualization comparing references, detections, and tracked outputs:
```bash
python visualizer.py \
  --input-references references.json \
  --input-detections detections.json \
  --input-tracked tracked.json \
  --output visualization.png \
  --dpi 50
```

---

#### Example output

![Visualizer Example](media/comparison.png)

---
