# ELEC872 — Image Properties and Face Recognition

This project investigates whether face embeddings encode image processing cues,
such as JPEG compression, sharpening, and interpolation, and how those cues may
affect recognition and fairness. Phase 1 prepares the data, models, and feature
pipeline needed for the proposal's later probing and mitigation experiments.

## Models

| Model | Role | Training in Phase 1 | Embedding |
|---|---|---|---|
| ResNet50 | CNN baseline | ImageNetV2 initialization; full backbone and a 480-class identity head trained on faces | 2,048-dimensional avgpool output |
| FaRL | General face representation baseline | Official ep16 checkpoint; frozen for inference | 512-dimensional visual projection |
| LVFace-T | Face recognition baseline | Official `vit_t` checkpoint; frozen for inference | 512-dimensional final embedding |

## Dataset and Processing

The implementation uses the Kaggle `hearfool/vggface2` upload. After excluding
14 images flagged during review, the main training manifest contains
**140,925 training images and 35,459 internal validation images from 480 identities**.
Photograph groups remain separate between training and internal validation.
The original validation split of 60 identities is held out for later embedding
comparisons on unseen identities.

Each source image produces an independent reference condition and six JPEG
conditions: quality 75, 85, or 95, with 4:2:0 or 4:4:4 chroma subsampling.
Sharpening, resizing, interpolation, and masking are available as additional
processing conditions. These are controlled processing labels for the
proposal's analysis, rather than labels identifying physical cameras.

## ResNet50 Training

`phase1.main_train` trains the full ResNet50 backbone and identity head using:

- Cross-entropy identity classification loss.
- SGD with learning rate 0.001, momentum 0.9, and weight decay 0.0001.
- Batch size 2, 2 CPU threads, and a target of 2 epochs.
- EXIF orientation correction, resize to a shortest side of 232 pixels,
  center crop to 224 × 224, and ImageNet normalization.
- Original face crops, with no additional augmentation during baseline training.

From the project directory, use the prepared Python environment, dataset,
and model weights to start or resume training:

```powershell
.\scripts\Start-MainTraining.ps1
```

To save after the current batch and pause:

```powershell
.\scripts\Stop-MainTraining.ps1
```

The checkpoint at `checkpoints/main_resnet/last.pt` preserves model weights,
optimizer momentum, RNG state, and the mid-epoch sample position. It is saved
every 300 steps and on a requested pause. Validation runs after each epoch.
The launcher starts no additional training once the 2-epoch target, including
validation, has been completed.

## Training Results

| Completed Epoch | Training Accuracy | Internal Validation Accuracy | Internal Validation Loss |
|---|---|---|---|
| 1 | 44.81% | 72.23% | 1.1834 |
| 2 | 79.93% | 81.12% | 0.8389 |

Epoch 2 training and validation are complete. These validation results measure
classification of the same 480 identities using different photographs; they do
not measure recognition of unseen identities.

The earlier 10-epoch configuration continued into epoch 3 before training
stopped. The retained `last.pt` includes 31,948 training images from epoch 3;
a separate checkpoint from the exact end of epoch 2 was not retained. The
81.12% result above belongs to the epoch 2 evaluation and must not be attributed
to the retained checkpoint without evaluating it separately. The recorded
training results and checkpoint status are in
[reports/main_resnet_stop_at_epoch2.json](reports/main_resnet_stop_at_epoch2.json).

## Features and Next Experiments

The verified smoke and pilot runs extracted features from all three models
for 25 sources / 400 variants and 99 sources / 693 variants, respectively.
Features are stored as raw and L2-normalized arrays, with source, identity,
processing condition, split, and checkpoint information attached to each row.

The smoke and pilot ResNet50 features use different checkpoints. Later probes
must use features extracted with one consistent checkpoint, including a new
extraction if the main training checkpoint is selected.

The next proposal stages cover image-property probes, recognition sensitivity,
adversarial experiments, and mitigation using an auxiliary classifier with
gradient reversal. These experiments have not yet been implemented in Phase 1.
Further implementation details are in [docs/decisions.md](docs/decisions.md),
and the proposal requirements are mapped in
[docs/requirements_matrix.md](docs/requirements_matrix.md).
