# Phase 1 Execution Report

**Phase 1 is not yet complete.** See the ResNet50 training snapshot below. Connecting the team remote repository and validating the baseline remain outstanding.

Report date: 2026-10-22 (America/New_York). The Phase 1 deadline is **Oct 26**, as stated in the proposal.

## Requirements Status

| ID | Requirement | Status | Scope Actually Executed | Evidence | Remaining Work |
|---|---|---|---|---|---|
| P1/U3/U5 | Acquire and inspect the Kaggle dataset | Complete | Inventory of 197,693 images; separate decoding audit of 1,697 images | [reports/data_inventory.json](../reports/data_inventory.json); [reports/acquisition.json](../reports/acquisition.json) | — |
| P2/U2/U11 | Obtain official FaRL and LVFace weights and run inference | Complete | FaRL ep16 / LVFace-T; frozen weights and evaluation mode; verified output dimensions; LVFace ONNX comparison | [reports/model_verification.json](../reports/model_verification.json); [features/smoke](../features/smoke) | — |
| P3/U12 | Train ResNet50 on face identities | In progress | Main training on 480 identities, with 140,925 training images and 35,459 internal validation images; target of 10 epochs | [reports/main_resnet_progress.json](../reports/main_resnet_progress.json); [reports/main_manifest.json](../reports/main_manifest.json); [reports/main_training_verification.json](../reports/main_training_verification.json) | Complete training and validate a usable face recognition baseline |
| P4 | Prepare the repository | Awaiting external input | Local Git setup complete; data and weights excluded | [.git](../.git); [.gitignore](../.gitignore) | Existing team remote URL required; creating a remote, publishing, or pushing requires separate instructions |
| P5/U9/U10 | Integrate the supplied processing code | Complete | 25 source images × 16 conditions; core IDENTITY+JPEG6 conditions plus 9 additional modules | [phase1/upstream_processors.py](../phase1/upstream_processors.py); [reports/smoke_processing.json](../reports/smoke_processing.json) | — |
| U6/U7 | Verify source links, labels, and split leakage | Complete | Full source inventory and photo-group/identity split checks for the experimental samples | [data/manifests](../data/manifests); [reports/smoke_validation.json](../reports/smoke_validation.json) | Near-duplicate checks across the full dataset and overlap with pretraining data remain unverified |
| U8 | Prepare reference faces and perform visual review | Complete | Existing crops retained; EXIF orientation applied 1 time; PNG references; sample review coverage documented separately | [reports/reference_review.json](../reports/reference_review.json); [reports/pilot_reference_review.json](../reports/pilot_reference_review.json); [reports/smoke_sources.png](../reports/smoke_sources.png) | Faces and identities have not been manually reviewed across the entire upload |
| U13/U15 | Verify model features, row alignment, and basic similarities | Complete | Separate raw and L2-normalized features; source/person/processing/model/hash/layer mappings; 3 similarity groups | [features](../features); reports/*_validation.json | — |
| U14 | Run a 25-image smoke test and scale up using measured results | Complete | 25-image smoke run; 99-image pilot run | [reports/smoke_validation.json](../reports/smoke_validation.json); [reports/pilot_validation.json](../reports/pilot_validation.json) | — |
| U4/U16/U17 | Record the environment, execution, and resume checks | Complete | Project virtual environment; actual commands and exit codes; checkpoint and feature shard resumption | [reports/environment.json](../reports/environment.json); [reports/command_runs.jsonl](../reports/command_runs.jsonl); [reports/resume_verification.json](../reports/resume_verification.json) | — |
| U1/U18 | Prepare documentation, figures, and the Phase 2 handoff | Complete | Requirement sources distinguished from implementation choices; workflow; comparisons of actual samples; status and limitations | [README.md](../README.md); [docs](../docs); [reports](../reports); [data/manifests](../data/manifests); [features](../features) | — |

## Main ResNet50 Training

Snapshot timestamp (UTC): 2026-10-22T23:56:47.844882+00:00. Process confirmed running: True. Status: running.
Target: 10 epochs. Current epoch: 1. Total optimizer steps: 240.
The latest progress is recorded in `reports/main_resnet_progress.json`, and log locations are listed in `logs/main_resnet_launch.json`. This paragraph is a snapshot taken when the report was generated.
Use `scripts/Stop-MainTraining.ps1` and `scripts/Start-MainTraining.ps1` to pause and resume training. Intermediate checkpoints are saved every 300 steps.

## Data and Model Validation

- Inventory: 480 training identities / 176,398 images, and 60 validation identities / 21,295 images. Overlapping person IDs between these splits: 0.
- Decoding has not been checked across the full dataset. Refer to the separate audit of 1,697 images and the `source_decode_coverage` recorded for each run. The data comes from the Kaggle upload, rather than the complete original VGGFace2 dataset.
- Separate audit: 1,697 images / 463 identities, with 0 decoding failures (`reports/decode_audit.json`). These sample checks do not establish validity across all images.
- All 6 processing module tests passed. Resumed and continuous ResNet50 training produced identical values for all 320 model tensors and the optimizer state. The maximum absolute difference between LVFace PyTorch and ONNX outputs for one actual image was approximately 7.6e-7.
- Download, loading, actual image inference, and face training status are recorded separately for the three models in `reports/checkpoint_status.json`.

### Smoke Run: 25 Source Images / 400 Variants

| Model | Actual Feature Array Shape | Status |
|---|---|---|
| farl | [400, 512] | Passed finite-value, raw/L2, and row/label alignment checks |
| lvface | [400, 512] | Passed finite-value, raw/L2, and row/label alignment checks |
| resnet50 | [400, 2048] | Passed finite-value, raw/L2, and row/label alignment checks |

![Comparison of actual smoke run processing conditions](smoke_comparison.png)

Similarities are recorded in three groups in `reports/smoke_validation.json`. They do not establish same-person probabilities, fairness, camera identification, attack success, or reliance on shortcuts.

### Pilot Run: 99 Source Images / 693 Variants

| Model | Actual Feature Array Shape | Status |
|---|---|---|
| farl | [693, 512] | Passed finite-value, raw/L2, and row/label alignment checks |
| lvface | [693, 512] | Passed finite-value, raw/L2, and row/label alignment checks |
| resnet50 | [693, 2048] | Passed finite-value, raw/L2, and row/label alignment checks |

![Comparison of actual pilot run processing conditions](pilot_comparison.png)

Similarities are recorded in three groups in `reports/pilot_validation.json`. They do not establish same-person probabilities, fairness, camera identification, attack success, or reliance on shortcuts.

## Resources and Feasibility

CPU: 13th Gen Intel(R) Core(TM) i7-1360P; GPU: Intel(R) Iris(R) Xe Graphics; memory: 15.72 GiB; CUDA available: False.

Measured face training throughput: 2.203 images/second (79 training images, batch size 2). Observed RSS: 0.49 GiB.
Extrapolating this throughput to 140,936 images gives **approximately 17.8 hours per epoch**, or **approximately 177.7 hours for 10 epochs**. Reading the sample PNG files is included in the measurement. Differences in archive I/O, validation, checkpoint saving, thermal throttling, and time to convergence are not accounted for, so this estimate does not guarantee completion within that time.
Main training on the current CPU is a separate undertaking from the small validation runs performed in this session. It requires either a dedicated GPU or sufficient CPU runtime. No resources were purchased and no remote runs were started automatically.

The proposed expansion to 100 identities × 20 images would require 14,000 feature vectors per model for the 7 core conditions. A simple extrapolation from pilot throughput estimates **approximately 3.9 hours** for feature extraction across all three models; reference image preparation, processing, visual review, and training would take additional time.
The proposal does not require a 2,000-image experiment. Phase 1 execution checks were expanded to 20 identities and 99 images. Resumable commands for a larger, separate run are provided in `docs/scale_up.md`. That larger example has not been executed; this does not imply that longer runs are infeasible.

## Rerunning the Pipeline and Phase 2 Handoff

Use the commands in `README.md` or `scripts/Run-Phase1.ps1`. Resume training with `--resume checkpoints/<run>/last.pt`. Rerunning the same feature extraction command reuses completed shards.

Hand off `data/manifests/sources.csv`, `*_sources.csv`, `*_variants.csv`, and each model's `features/<run>/<model>/{raw.npy,l2.npy,rows.csv,model.json}` files. Preserve `validation.json` and the checkpoint SHA alongside them.

The ResNet50 features reflect training on a small face dataset or subset and are not presented as a fully validated face recognition baseline. FaRL is a general face representation model, while LVFace is intended for face recognition. The Kaggle splits do not guarantee separation from the models' prior pretraining data.

The ResNet50 smoke and pilot runs use different checkpoints, so their features cannot be combined. For Phase 2, extract features using one consistent checkpoint. Source images and conditions shared by the two FaRL or LVFace runs must not be counted twice as independent samples.

![Workflow](workflow.png)

Requirement sources are detailed in `docs/requirements_matrix.md`. Implementation choices and their implications are documented in `docs/decisions.md`. Execution commands and exit codes are recorded in `reports/command_runs.jsonl`.
