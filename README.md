# ELEC872 Phase 1

Execution status and limitations: [reports/final_status.md](reports/final_status.md).
Requirement sources: [docs/requirements_matrix.md](docs/requirements_matrix.md).
Implementation choices beyond the proposal: [docs/decisions.md](docs/decisions.md).

Run the commands below in PowerShell from this directory. Existing files in
`tmp` have been preserved. Data and models come from public sources; no paid
resources are used. Korean originals are kept locally with `_kr` in their
filenames and are excluded from Git.

```powershell
./scripts/Setup-Environment.ps1
.\.venv-run\Scripts\python.exe -m pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv-run\Scripts\python.exe scripts/acquire.py
.\.venv-run\Scripts\python.exe scripts/environment.py
.\.venv-run\Scripts\python.exe -m phase1.data inventory
.\.venv-run\Scripts\python.exe scripts/decode_audit.py
.\.venv-run\Scripts\python.exe -m phase1.data prepare --run smoke --identities 5 --per-identity 5 --extras
.\.venv-run\Scripts\python.exe -m pytest -q tests
.\.venv-run\Scripts\python.exe -m phase1.train --run smoke_resnet --epochs 1
.\.venv-run\Scripts\python.exe -m phase1.train --run smoke_resnet --epochs 2 --resume checkpoints/smoke_resnet/last.pt
.\.venv-run\Scripts\python.exe -m phase1.extract --model farl --run smoke
.\.venv-run\Scripts\python.exe -m phase1.extract --model lvface --run smoke
.\.venv-run\Scripts\python.exe -m phase1.extract --model resnet50 --run smoke
.\.venv-run\Scripts\python.exe -m phase1.validate --run smoke
```

To repeat this session's small verification run in sequence:

```powershell
.\.venv-run\Scripts\python.exe scripts/execute_verified.py
```

This command preserves the prepared smoke manifests and runs the tests,
training/resume/continuous-training comparison, feature extraction for all
three models, and row alignment checks. It records a log and exit code for
each stage and stops at the first error, preserving details in
`logs/<stage>.log`.

The verified runtime is `.venv-run`. It shares existing global packages on a
**read-only basis**; no global packages were modified. The first attempt at a
standalone installation in `.venv` was interrupted. In this session,
`.venv-run/Lib/site-packages/project-shared.pth` reads helper dependencies
already installed there, so keep both directories. For a fresh installation,
the setup script above installs the required packages in the project virtual
environment, and this auxiliary path is unnecessary. Actual import paths and
versions are recorded in `reports/environment.json`.

`smoke --extras` produces 25 images × 16 conditions, including the 7 core
JPEG conditions. To run only the core conditions, use a new run name and omit
`--extras`. Review `reports/*_sources.png` and flag multiple faces or ambiguous
targets before using the images.

If a download is interrupted, rerun `scripts/acquire.py`; it resumes the
`.part` file using HTTP Range requests. Rerunning feature extraction verifies
and reuses completed shards. Use a new run directory if the input manifest or
checkpoint changes. Changing the extraction batch size within a run still
allows cached features to be matched by variant ID and reused, provided the
model and input manifest checks pass. The original smoke/pilot trainer resumes
from the **last completed epoch**, restoring the optimizer and identity map.
Its losses and accuracies verify small-scale execution and do not establish
face recognition performance.

Start or resume face training across the full upload with the separate
background launcher:

```powershell
.\scripts\Start-MainTraining.ps1
# Finish the current batch, save, and pause:
.\scripts\Stop-MainTraining.ps1
# Start resumes last.pt only if the 2-epoch target, including validation, is incomplete.
Get-Content reports/main_resnet_progress.json
```

Main training uses `phase1.main_train`. After excluding the 14 images flagged
in review, the original training split contains 480 identities with **140,925
training images / 35,459 internal validation images**. The configuration uses
ImageNetV2 initialization, trains the full backbone, and targets 2 epochs with
batch size 2, 2 CPU threads, and SGD at lr 0.001.
On 2026-10-26, the user reduced the target from 10 epochs to 2. At that point,
epoch 2 validation had already finished, and the latest model also contained
partial epoch 3 training. The epoch 2 validation results are preserved, but a
separate model checkpoint from that exact point was not retained. The current
`last.pt` must therefore not be described as a model trained for exactly
2 epochs. The 60 identities in the original validation split are excluded.
The exact manifest and exclusion reasons are recorded in
`data/manifests/main_sources.csv` and `reports/main_manifest.json`.
This does not establish that all images have been decoded or all faces and
identities manually reviewed. A decoding error records the source identifier
and stops the run; images are not silently skipped.

Progress and loss are recorded every 20 steps. Resume state is saved at step
10 and every 300 steps thereafter. `checkpoints/main_resnet/last.pt` includes
the mid-epoch sample position, weights, SGD momentum, and RNG state. After an
abrupt exit, only work since the last checkpoint must be repeated. If validation
is interrupted, that validation pass restarts from the beginning. Exact
agreement between mid-epoch resume and continuous training is verified in
`tests/test_main_training.py`.
Progress is recorded in `reports/main_resnet_progress.json`, epoch results in
`reports/main_resnet_training.json`, and launch logs/PIDs in
`logs/main_resnet_launch.json`. The progress JSON is a snapshot; check whether
its recorded PID is still running to detect an unexpected exit.
The launcher runs a separate, hidden process. While training, it temporarily
prevents automatic Windows idle sleep without changing persistent power
settings. The display may turn off. Training does not progress during Windows
shutdown, explicit sleep, or lid closure. CPU training can take several days;
completing the configured schedule does not establish recognition performance.

The first model initialization downloads official ImageNetV2 weights to
`weights/torch_cache`. Obtaining these weights does not count as completed
face training. The original Kaggle validation split is held out for embedding
comparisons on unseen identities and is excluded from the identity head's
closed-set validation. Embedding comparisons can use the avgpool output from
`load_resnet`; Phase 1 does not repeatedly tune against the final evaluation.

For the Phase 2 handoff, retain `sources.csv`, each run's source/variant CSVs,
and each model's `raw.npy`, `l2.npy`, `rows.csv`, `model.json`, and
`validation.json`. Use only artifacts that pass the same row alignment checks;
do not assume that different models share embedding coordinates.
The ResNet50 smoke and pilot runs use **different checkpoints**, so their
features must not be combined. Re-extract features using one consistent
checkpoint for Phase 2. FaRL and LVFace each retain their respective official
weights across both runs; overlapping source/condition pairs must not be
counted twice.

To run the measured pilot configuration (20 identities × 5 images, with
7 core JPEG conditions):

```powershell
.\.venv-run\Scripts\python.exe scripts/execute_pilot.py --identities 20 --per-identity 5
```

This pilot's ResNet50 also uses 2 epochs of subset training and is not a fully
validated recognition baseline. After visual review, the saved pilot contains
**20 identities, 99 source images, and 693 variants**. The final image with
multiple faces was excluded from the 100-image target; its identifier and the
reason are preserved separately. Rerunning with an existing pilot manifest
retains these review decisions. Face training uses batch size 2 and 2 threads;
feature extraction uses batch size 8 and 4 threads, selected from actual
measurements. The evidence for these settings is in
`reports/inference_benchmark.json`.
Commands for a larger 2,000-image run are provided separately in
[docs/scale_up.md](docs/scale_up.md). This example has not been executed and
is not included in the completed scope.
