# ELEC872 — Image Properties and Face Recognition

This project investigates whether face embeddings encode image processing cues,
such as JPEG compression, sharpening, and interpolation, and how those cues may
affect recognition and fairness. Phase 1 prepares the data, models, and features
for the proposal's later probing and mitigation experiments.

## Models

| Model | Role | Phase 1 Training | Embedding |
|---|---|---|---|
| ResNet50 | CNN baseline | Full backbone and a 480-class identity head; ImageNetV2 initialization | 2,048-dimensional avgpool output |
| FaRL | General face representation baseline | Official ep16 checkpoint; frozen for inference | 512-dimensional visual projection |
| LVFace-T | Face recognition baseline | Official `vit_t` checkpoint; frozen for inference | 512-dimensional final embedding |

## Dataset and Processing

The Kaggle `hearfool/vggface2` upload provides the source images. Excluding 14
images flagged during review leaves **140,925 training images and 35,459 internal
validation images from 480 identities**. Photograph groups remain disjoint between
training and validation. The original validation split of 60 identities is held
out for later embedding comparisons on unseen identities.

Each source produces a reference condition and six JPEG conditions: quality
75, 85, or 95, with 4:2:0 or 4:4:4 chroma subsampling. Sharpening, resizing,
interpolation, and masking are additional conditions. Labels identify controlled
processing settings, rather than physical cameras.

## CUDA Environment

Training and feature extraction require an NVIDIA GPU and CUDA-enabled PyTorch.
There is no CPU training option or automatic CPU fallback. The pinned environment
uses Python 3.11, PyTorch 2.10.0, torchvision 0.25.0, and CUDA 12.6 wheels; an appropriate NVIDIA
driver is required. No system driver or global package is installed by this project.

On Windows, create a separate environment:

```powershell
.\scripts\Setup-Environment.ps1
```

On Linux, create and activate a Python virtual environment, then install
`requirements-cuda.txt`. Obtain the dataset and model weights with
`scripts/acquire.py` and create the source inventory with `python -m phase1.data inventory`.
The historical `requirements.lock.txt` records the earlier CPU experiments;
use `requirements-cuda.txt` for the current implementation.

## Unified ResNet50 Training

`phase1/train.py` is the single trainer for small experiments and full training.
The former `main_train.py` has been merged into it. Dataset size is selected by
the manifest, rather than by a separate training implementation.

Defaults: **Adam**, learning rate 0.001, weight decay 0.0001, batch size 2,
2 epochs, CUDA GPU 0, and 2 host preprocessing threads. Images use EXIF orientation
correction, shortest-side resize to 232, center crop to 224 × 224, and ImageNet
normalization. The baseline uses original crops without additional augmentation.

Start or resume a new Adam run from the project directory:

```powershell
.\scripts\Start-MainTraining.ps1 -Run main_resnet_adam -Gpu 0
```

Or use the same trainer directly in the CUDA environment:

```powershell
python -m phase1.train --run main_resnet_adam --gpu 0 --epochs 2
# Resume the same experiment:
python -m phase1.train --run main_resnet_adam --gpu 0 --epochs 2 --resume checkpoints/main_resnet_adam/last.pt
```

To pause after saving the current batch:

```powershell
.\scripts\Stop-MainTraining.ps1 -Run main_resnet_adam
```

Direct CLI users should remove the run's `STOP` file before resuming; the Windows
launcher handles this automatically. It also skips runs whose requested epochs
have already completed with validation.

Checkpoints save model weights, optimizer state, CPU and CUDA RNG state, and the
mid-epoch sample position. They are written atomically every 300 steps, at a
requested pause, and before/after validation. Validation restarts from its beginning
if interrupted. Progress includes the CUDA device, GPU name, and GPU memory usage.

## Existing Checkpoints and Experiment Results

New runs default to `main_resnet_adam`, preserving the earlier SGD run in
`checkpoints/main_resnet`. `--resume` requires matching optimizer, learning rate,
identity mapping, manifest, and batch size. Legacy SGD checkpoints require
`--optimizer sgd`; checkpoints from the old basic trainer resume at an epoch boundary.
CPU-to-CUDA migration preserves saved training state but is not a promise of
bit-identical results across devices.

To initialize a **new** Adam experiment from an existing face checkpoint, use
`--initialize-from checkpoints/main_resnet/last.pt` with a new run name. This loads
matching identity weights and resets the optimizer, epoch, and step counters.
Its source checkpoint hash is recorded. Adam never loads SGD optimizer state.

The following are **historical SGD results**, recorded before the CUDA-only
implementation; they are not Adam/CUDA training results:

| Completed Epoch | Training Accuracy | Internal Validation Accuracy | Internal Validation Loss |
|---|---|---|---|
| 1 | 44.81% | 72.23% | 1.1834 |
| 2 | 79.93% | 81.12% | 0.8389 |

These results classify the same 480 identities using different photographs.
The earlier schedule continued into epoch 3; its retained checkpoint includes
31,948 additional training images. An exact epoch 2 checkpoint was not retained,
so 81.12% must not be attributed to the retained weights without re-evaluation.
See [the historical training record](reports/main_resnet_stop_at_epoch2.json).

## Features and Verification

`data.py` prepares image manifests and processing conditions. `extract.py` runs
frozen models on CUDA and stores raw/L2-normalized embeddings with source,
identity, processing condition, split, and checkpoint information. ONNX CUDA
execution requires the GPU runtime included in `requirements-cuda.txt`.

The earlier verified smoke and pilot runs contain 25 sources / 400 variants
and 99 sources / 693 variants. New CUDA experiments use separate run names:

```powershell
python scripts/execute_verified.py --run smoke_adam --gpu 0
python scripts/execute_pilot.py --run pilot_adam --gpu 0
python -m pytest -q tests
```

Training integration tests require an actual CUDA GPU and are skipped when one
is unavailable. Host-only tests check input validation and checkpoint compatibility;
they do not stand in for GPU training verification. The development laptop has no
NVIDIA GPU, so the new CUDA training path has not been executed on that machine.

ResNet50 features from different checkpoints must not be pooled. Later probes
require a new, consistent feature extraction if the training checkpoint changes.
The next proposal stages cover property probes, recognition sensitivity,
adversarial experiments, and gradient-reversal mitigation; these remain future work.
Third-party sources and licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
