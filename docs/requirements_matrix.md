# Phase 1 requirements and provenance

Primary source: `../proposal/ELEC872_Project_Proposal.pdf`, all three pages,
including Figure 1, timeline, and references [1]–[12]. The PDF is a research
source; implementation authorization comes from the user's pasted request.

## Proposal requirements (P)

| ID | Source and requirement | Implementation | Validation / evidence | Status |
|---|---|---|---|---|
| P1 | p3 Phase 1: download needed datasets | `scripts/acquire.py`; agreed Kaggle VGGFace2 upload | `reports/acquisition.json`, `reports/data_inventory.json` | See live status report |
| P2 | p2 §3; p3 Phase 1: obtain pretrained FaRL and LVFace | official FaRL ep16; official LVFace-T `.pt` and `.onnx` | checkpoint hashes; strict state load; real-image features | See live status report |
| P3 | p2 §3: own ResNet50 CNN baseline; p3: assess small-model training | `phase1/train.py`; identity classification; avgpool embeddings | actual backward/validation/save/resume logs; measured feasibility | See live status report |
| P4 | p3 Phase 1: set up GitHub repository | local Git repo; excluded raw data/weights/credentials | `git status`; remote association pending team URL | External dependency: team remote not supplied |
| P5 | p2 §3; p3 Phase 1: adapt Ramos image-artifact pipeline [6] | supplied Processor functions; core JPEG and configurable extras | processing manifests, pixel/sampling tests, real comparison PNG | See live status report |

Figure 1 depicts later encoder→embedding→recognition and GRL→classifier branches.
Only the encoder/embedding foundation is in this implementation. GRL, probing,
attacks, fairness analysis, synthetic-data comparison, and final-paper work remain
outside Phase 1. References [2], [6], [10], [12] motivate the data/pipeline/models;
[1], [3], [4], [5], [7]–[9], [11] were read as context, not extra execution tasks.

## User requirements (U)

| ID | User instruction | Code / method | Evidence / output |
|---|---|---|---|
| U1 | Actual files, execution, accurate completion labels | runnable CLI modules and measured reports | `README.md`, `reports/final_status.md` |
| U2 | Fixed three models; no substitutions | `phase1/models.py` | per-model feature `model.json`, hashes |
| U3 | Only hearfool/vggface2; preserve source | immutable ZIP, separate PNGs/manifests/features | `data/raw`, `data/reference`, `data/processed`, `features` |
| U4 | Recheck hardware; project environments, no paid resources | CPU runtime; local `.venv` | `reports/environment.json`, `requirements.lock.txt` |
| U5 | Full listing versus actual decoding | inventory, bounded decode audit and selected-image preparation report separate coverage | `reports/data_inventory.json`, `reports/decode_audit.json`, `reports/*_processing.json` |
| U6 | Source/person/split/path and separate labels | source and variant CSVs; separate `person_id`/`processing_label` | `data/manifests` |
| U7 | No same-source leakage; distinguish face and probe splits | photograph-group face split; identity-level probe split | integrity validation; `docs/decisions.md` |
| U8 | Inspect existing crops; no arbitrary largest-face selection | contact sheet; full extent RGB; review flags | `reports/smoke_sources.png`, review record |
| U9 | Independent IDENTITY + six JPEG variants | fresh reference copy per condition; upstream functions | tests, quantization/sampling metadata, comparison figure |
| U10 | Sharpen/resize/interpolation configurable; composite JPEG explicit | all provided Enum conditions; per-source RNG reset | operation order and parameter JSON in variant CSV |
| U11 | Actual FaRL weights; correct LVFace architecture; frozen/eval | strict visual loading; `vit_t`; official RGB normalization | feature model metadata and real outputs |
| U12 | Identity head/map; loss/backward/validation/save/resume | full ResNet50 fine-tuning implementation | `checkpoints/*`, training JSON, resume checks |
| U13 | Features/layer/checkpoint linkage; finite values; aligned rows | immutable feature shards; raw + L2 arrays and CSV | `features/*/*/validation.json`, integrity reports |
| U14 | 5 people × 5 sources; measured feasible expansion | training split only; no final evaluation tuning | smoke/pilot manifests, throughput and memory |
| U15 | Three similarity groups; conservative interpretation | descriptive cosine on within-model L2 features | validation JSON; no identity-probability claim |
| U16 | Run checks and record coverage/results/commands | pytest + data/model integration validation | `logs`, `reports/commands.md`, validation JSON |
| U17 | Resumable long tasks; keep existing work | atomic checkpoints and feature shards; existing `tmp` preserved | train `--resume`, extraction rerun |
| U18 | Korean final status; matrix, diagram and real figures | output documentation and actual comparison PNG | `reports/final_status.md`, `reports/workflow.svg` |

## Original timeline (verbatim dates)

Phase 1: **Oct 26**; Phase 2: **Nov 9**; Phase 3: **Nov 16**;
Phase 4: **Nov 30**; Phase 5: **Dec 14**. The paper is a 2026 student proposal;
these month/day strings are preserved, not silently shifted. Current work date:
2026-10-22 America/New_York. Feasibility is based on measured CPU runs, not the
historical dates alone.
