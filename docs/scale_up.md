# Larger run example — not executed

The verified pilot is **99 sources / 20 identities / 693 core variants**.
The following 2,000-source example is **code-only configuration guidance**,
not an executed result or a mandatory proposal requirement. Measured model
throughput projects about 3.9 hours of three-model extraction, plus preparation,
review and training. Full identity-baseline training is a separate requirement.

Use a new run name so previously verified arrays/manifests remain immutable.

```powershell
.\.venv-run\Scripts\python.exe -m phase1.data prepare --run pilot_2000 --identities 100 --per-identity 20
# Review references before model training; contact-sheet generation covers the
# first 100 sources. Create additional pages with phase1.data.contact_sheet,
# record exclusions/review coverage, and finalize the manifest before training.
.\.venv-run\Scripts\python.exe -m phase1.train --manifest data/manifests/pilot_2000_sources.csv --run pilot_2000_resnet --epochs 2
.\.venv-run\Scripts\python.exe -m phase1.extract --run pilot_2000 --model farl --batch-size 8 --threads 4
.\.venv-run\Scripts\python.exe -m phase1.extract --run pilot_2000 --model lvface --batch-size 8 --threads 4
.\.venv-run\Scripts\python.exe -m phase1.extract --run pilot_2000 --model resnet50 --checkpoint checkpoints/pilot_2000_resnet/last.pt --batch-size 8 --threads 4
.\.venv-run\Scripts\python.exe -m phase1.validate --run pilot_2000
```

Training resume repeats an interrupted epoch from the last completed checkpoint:

```powershell
.\.venv-run\Scripts\python.exe -m phase1.train --manifest data/manifests/pilot_2000_sources.csv --run pilot_2000_resnet --epochs 2 --resume checkpoints/pilot_2000_resnet/last.pt
```

Feature extraction resumes by rerunning the same extraction command. Changed
input-manifest or checkpoint hashes are rejected: start a new run rather than
mixing coordinates or labels. Raw face files and prior artifacts are preserved.
No final evaluation split is used by this preparation command.
