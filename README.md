# gTec EEG Pipeline

Real-time EEG acquisition, visualization, recording, filtering, and motor-imagery classification pipeline for g.tec-based BCI experiments.

The project is designed around modular Python nodes that communicate through local TCP/UDP sockets. It can be used with g.tec Unicorn/g.Nautilus devices, previously recorded `.mat` files, or the built-in simulated `test` acquisition mode.

## Main Features

- Real-time EEG acquisition from g.tec Unicorn and g.Nautilus devices.
- Simulated acquisition mode for testing the pipeline without hardware.
- Real-time filtering and visualization with PyQtGraph.
- Recording of EEG data, timestamps, and events into MATLAB-compatible `.mat` files.
- FgMDM/Riemannian motor-imagery classifier training and online inference.
- Optional multi-PC setup for coupled BCI experiments.
- Optional fusion/weighting utilities for combining classifier outputs.

## Repository Structure

```text
.
|-- startPipeline_main.py        # Main launcher for acquisition, recording, filtering, classification, and output mapping
|-- startPipeline_second.py      # Secondary launcher for two-user/two-PC experiments
|-- startVisualizer.py           # Acquisition + filter + real-time visualizer launcher
|-- create_classifier.py         # Train a subject-specific FgMDM classifier from recorded data
|-- compute_CoupleWeights.py     # Compute fusion weights for coupled classifiers
|-- classNodes/                  # Node implementations
|-- classLaunchers/              # Process launchers for each node
|-- utils/                       # Socket, buffer, and helper utilities
|-- py_utils/                    # EEG/data/signal-processing utilities
|-- riemann_utils/               # Riemannian covariance utilities
`-- fgmdm_riemann/               # Local FgMDM package
```

## Requirements

This repository has been tested with:

- Python 3.11.9
- Windows
- g.tec drivers/libraries for hardware acquisition

For real g.tec acquisition, install the official g.tec packages and activate the required license according to the g.tec documentation. The relevant Python modules are imported as `pygds` for g.Nautilus and `UnicornPy` for Unicorn.

The pipeline can still be tested without g.tec hardware by using `device = 'test'`.

## Installation

Clone the repository with its submodules:

```bash
git clone --recurse-submodules <repository-url>
cd gTec_EEGpipeline
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the local classifier package and Python dependencies:

```bash
python -m pip install -e fgmdm_riemann
python -m pip install -r requirements.txt
```

## Data Layout

The scripts expect a `data/` directory with this general structure:

```text
data/
|-- recordings/
|   `-- <subject_code>/
|       `-- <YYYYMMDD>/
|           `-- <subject>.<date>.<time>.<run_type>.<task>.mat
|-- models/
|   `-- <subject_code>/
|       `-- <model_name>.joblib
|   `-- test/
|       `-- modelTest.joblib
|-- weights/
|   `-- <weights_name>.mat
|-- lapMask8Unicorn.mat
`-- lapMask16Nautilus.mat
```

Recording files are expected to contain:

- `s`: EEG signal array shaped as samples by channels.
- `h`: header dictionary/struct containing at least `SampleRate`, `channels`, `dataChunkSize`, and `EVENT`.
- `h.EVENT`: event information containing `TYP`, `POS`, and `DUR`.

Common event codes used by the scripts include:

- `769`, `770`: left-hand/right-hand motor imagery for `mi_lhrh`.
- `771`, `773`: task-specific classes for `mi_bfbh`.
- `781`: feedback period.
- `event + 0x8000`: event-closing marker used during recording.

The repository includes the default Laplacian mask files used by the active scripts:

- `data/lapMask8Unicorn.mat`
- `data/lapMask16Nautilus.mat`

These files are not needed for raw acquisition, recording, or basic visualization. They are needed when Laplacian spatial filtering is enabled during classifier training or online classification. Other `.mat` files, such as recordings, models exported as MATLAB files, and result files, remain ignored by default.

The repository also includes `data/models/test/modelTest.joblib`, a small test model used by the default `test` configuration. It is provided so the full launcher can be started without private participant models.

## Quick Start Without Hardware

Use the simulated acquisition mode to check that the socket pipeline and visualizer are working:

1. Open `startVisualizer.py`.
2. Set:

```python
device = 'test'
```

3. Run:

```bash
python startVisualizer.py
```

This starts acquisition, filtering, and the real-time visualizer. Press `F12` to stop the launcher processes.

## Running the Full Pipeline

Most experiment settings are configured near the top of `startPipeline_main.py`:

```python
useMultiplePc = False
runType = "test"          # "calibration", "evaluation", or "test"
task = "mi_lhrh_TEST"
subjectCode = "test"
device = "test"           # "test", "un", "na", a device id, or a .mat file path
model = "modelTest.joblib"
alpha = 0.99
weights = "same"
```

Then run:

```bash
python startPipeline_main.py
```

Depending on `runType`, the launcher starts:

- `PortManager`: shares port information between nodes.
- `Acquisition`: reads EEG from hardware, a `.mat` file, or simulated data.
- `Recorder`: saves streamed EEG and event information.
- `Filter`: applies requested online filters.
- `Classifier`: computes online class probabilities.
- `OutputMapper`: integrates one or more classifier outputs into a continuous control value.

Use `runType = "calibration"` when collecting training data. Use `runType = "evaluation"` or `"test"` when running with an existing classifier.

The default `subjectCode = "test"` and `model = "modelTest.joblib"` configuration uses the included demo model at `data/models/test/modelTest.joblib`.

Press `F12` to stop the launcher processes. Some individual nodes also define node-specific function-key shortcuts in `classLaunchers/`.

## Pipeline Timing Telemetry

The pipeline includes lightweight timing telemetry. It does not change EEG payloads,
model inputs, filters, recording output, or the socket protocol.

By default it is quiet during a healthy run: it prints a compact summary when a node
stops and prints a warning if a stage falls behind. The measurements separate:

- `Acquisition`: source cadence. A warning here points to the headset, driver, or
  acquisition loop rather than the classifier.
- `Filter` and `Classifier`: cadence, local processing time, and age of the packet
  received from the preceding node.
- `OutputMapper`: completed probability-merge rate and incoming probability age.

The configuration is near the top of each launcher. In `startPipeline_main.py`,
for example:

```python
telemetryEnabled = True
telemetryReportSeconds = 5
telemetryVerbose = False
```

Set `telemetryVerbose = True` for one compact summary per interval. Set
`telemetryEnabled = False` to disable it. The expected cadence is calculated from
the live acquisition settings (`dataChunkSize / SampleRate`), so it automatically
adapts when either setting changes. Packet-age measurements across two PCs are only
an absolute latency measurement when both computers' clocks are synchronized;
cadence and local processing-time measurements remain reliable on each machine.

## Device Options

The `device` variable controls the acquisition source:

```python
device = "test"                 # Simulated signal, useful for debugging
device = "un"                   # Auto-connect to an available Unicorn device
device = "UN-..."               # Connect to a specific Unicorn device id
device = "na"                   # Auto-connect to a g.Nautilus device
device = "NA-..."               # Connect to a specific g.Nautilus device id
device = r"path\to\file.mat"    # Replay a recorded MATLAB file
```

If `device = None`, the acquisition node first tries g.Nautilus and then Unicorn.

## Training a Classifier

To train an FgMDM classifier from recorded calibration data:

1. Put calibration recordings under `data/recordings/<subject_code>/<YYYYMMDD>/`.
2. Check the configuration values in `create_classifier.py`, especially:

```python
bandPass = [[6, 24]]
stopBand = [[14, 18]]
wantedChannels = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
windowsLength = 1
normalizationMethod = 'lwf'
```

3. Run:

```bash
python create_classifier.py
```

The script loads selected recordings, preprocesses the signal, computes normalized covariance matrices, trains the FgMDM classifier, and builds a model dictionary compatible with the online `Classifier` node.

Note: in the current script, the final `save(filename, model)` line is commented out. Enable it if you want the trained model to be written to `data/models/<subject_code>/`.

## Coupled-BCI Weights

For two-user or coupled-classifier experiments, use:

```bash
python compute_CoupleWeights.py
```

This calls the utilities in `utils/extract_coupleWeights.py`, asks for the relevant recordings and models, estimates model scores, and can save normalized fusion weights under `data/weights/`.

In `startPipeline_main.py`, weights can be set as:

```python
weights = "same"                 # Equal weighting
weights = "file_name.mat"        # Load weights from data/weights/
weights = [1]                    # Single classifier
```

## Multi-PC Use

For multi-PC experiments, set:

```python
useMultiplePc = True
```

Run `startPipeline_main.py` on the main machine and `startPipeline_second.py` on the secondary machine. The secondary launcher will ask for the IP address of the main machine when needed.

Make sure both machines are on the same network and that the selected TCP/UDP ports are not blocked by the firewall.


## Citation

If you use this repository in academic work, please cite the associated publication once it becomes available.

A paper describing this pipeline and its use in coupled BCI experiments is expected for the Graz BCI Conference, but it has not been published yet. Until the final citation is available, please cite this repository and mention the forthcoming Graz conference paper.

Suggested temporary citation:

```bibtex
@inproceedings{palatella_mindrun_2026,
  author = {Palatella, Alessio and Tortora, Stefano and Menegatti, Emanuele and Tonin, Luca and Alimardani, Maryam},
  title = {MindRun: an Accessible Multi-Player BCI Game for Collaborative Motor Imagery Training},
  booktitle = {Proceedings of the Graz BCI Conference},
  year = {2026},
  note = {Forthcoming}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
