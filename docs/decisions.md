# Implementation decisions (not proposal requirements)

1. **Dataset selection.** The proposal mentions LFW and VGGFace2 alternatives.
   The user fixes hearfool/vggface2, overriding that choice. This upload must not
   be described as all 3.31M images/9,131 identities of original VGGFace2.
   Kaggle API metadata reports version1, last updated 2022-05-01; source ZIP SHA
   and metadata are retained. Acquisition is pinned to dataset version1.
2. **FaRL ep16 / LVFace-T.** FaRL's paper checkpoint and the smallest official
   LVFace make CPU verification feasible. FaRL is general facial representation
   learning (LAION-Face), LVFace is face recognition (Glint360K). Their training
   objectives and data differ; same dimensions do not imply shared coordinates.
3. **Reference images.** Decode the existing crop, apply EXIF orientation once,
   convert RGB, preserve full extent and losslessly save PNG. No new face
   detection, alignment or crop is introduced. This avoids condition-dependent
   geometry. Human visual contact-sheet inspection is documented separately;
   full-dataset target-identity/face-count correctness is not certified. Ambiguous
   targets need review/exclusion, never automatic largest-face selection. PNG
   cannot restore losses in original JPEGs; IDENTITY is not a pristine capture.
4. **Splits.** Original train/val is retained. Original val identities are
   `heldout_identity` for face training and `probe_test` for future probes, and
   remain untouched during smoke/pilot. Within each train identity, hash of
   person/filename-prefix groups photographs into face_train (~80%) / face_val
   (~20%). Multiple `_01`/`_02` crops of one numbered photo stay together. Probe
   train/val instead split train identities by stable hash (~80%/~20%). All
   processed variants inherit source splits. These roles are independent; a
   later probe protocol must choose the appropriate face-encoder training
   relation explicitly. Same-identity probe train/val overlap is forbidden here.
   Pretraining overlap with FaRL/LVFace is unknown; Kaggle splits do not prove
   disjointness from their historical pretraining. Exact duplicates within the
   selected set are checked; near-duplicate/full-dataset content overlap is not
   certified by filename groups.
5. **Sampling.** Seed 872; hash-ranked identities/photos; 4 face-training and 1
   within-identity face-validation photo per identity in the 25-image smoke.
   Distinct numbered photo groups are selected. Pilot scale is decided after
   actual throughput measurement. The suggested 100×20 is not in the proposal.
   The actual pilot target is 20×5; after reviewing candidates/replacements,
   99 sources across 20 identities are retained (one identity has four photos).
   All seven core conditions have exactly 99 variants. Excluded sources/reasons
   are retained in `data/manifests/review_exclusions.csv`; raw files are untouched.
6. **Processing.** The supplied fork and original processors share a recorded
   commit. Their functions are used directly from a vendored snapshot.
   IDENTITY + Q75/85/95 × 420/444 are independent reference-to-condition calls.
   Sharpen2/4, half/double resize, interpolation, MASK95 are optional modules.
   Sharpen/resize/interpolation include final JPEG95/444; MASK95 is JPEG95/444
   then central masking. Random interpolation uses an explicit per-source,
   per-condition seed rather than order-dependent global state. The realized
   size and operation representation are stored. Output PNGs preserve decoded
   JPEG pixels without a second encode. Quality means our encoder setting;
   labels are processing labels, not EXIF strings or physical camera models.
   The four optional interpolation conditions also have independently seeded
   scale draws. They are composite interpolation+rescale+JPEG conditions, not a
   controlled comparison of interpolation mode alone; use a common fixed scale
   in a new run if that isolated comparison is needed in Phase 2.
7. **Model preprocessing after processing.** Reference → independent processing
   → decoded pixels → model-specific resize/crop/normalization → feature.
   FaRL: RGB shortest-side224 bicubic, center224, official CLIP mean/std;
   projected CLS512 using only actual FaRL visual weights, strict load.
   LVFace-T: RGB OpenCV linear112, `(x/255-.5)/.5`, final512, exact `vit_t`
   architecture. `.onnx` supports frozen inference; `.pt` is retained for
   possible future training and architecture inspection. ResNet50: ImageNetV2
   initialization, shortest-side232 bilinear center224 and ImageNet mean/std;
   identity head is trained on the selected identity map; avgpool2048 retained.
   Dimensions must still be verified on actual outputs.
8. **Training.** ImageNet initialization is transfer initialization, not a face
   model. All backbone parameters plus the identity head are optimized with
   SGD (lr .001, momentum .9, weight decay 1e-4), batch2, CPU2 threads. No
   JPEG augmentation in the basic baseline: avoids confounding later processing
   probes. Two smoke epochs verify the mechanics, not useful recognition.
   Ten full-data epochs is a planning scenario, not a convergence guarantee or
   paper reproduction. Epoch-boundary resume validates the manifest/map and
   restores optimizer state. Mid-epoch interruption repeats that epoch.
9. **Features.** Raw float32 and rowwise L2 float32 are separate arrays. A
   CSV maps every row to source/person/condition/splits/model/hash/layer. Failed
   inputs fail explicitly rather than shifting labels. Immutable NPZ batches
   permit restart with input/weight metadata checks. Cross-model similarities
   are never computed. Descriptive groups are same-source variants,
   same-identity different photos and different identities, not calibrated
   probabilities or Phase 2 shortcut/fairness/attack findings.
   ResNet50 smoke and pilot use different face-trained checkpoints (five versus
   twenty identities). Their 2048-dimensional coordinates must not be pooled.
   Choose one checkpoint and re-extract all images under that same checkpoint
   before a Phase 2 probe. FaRL and LVFace keep their respective fixed official
   checkpoints across both runs; repeated source/condition rows can be deduplicated
   by variant ID plus checkpoint SHA, rather than counted as new independent data.
10. **Git and scope.** Local Git only because no team destination was supplied.
    Remote creation, publication and push require a separate instruction; adding
    a supplied existing team remote is part of repository setup. Weights, raw faces, feature arrays,
    credential files and generated images are ignored. No paid compute/API,
    global driver changes, DigiFace1M expansion, GRL or probe training.

11. **Current-resource adjustment.** Only about 1.3–1.6GiB physical memory was
    free during setup (despite 15.7GiB total). The first fully independent venv
    installation was stopped during its large PyTorch install. `.venv-run`
    reuses the existing same-version CPU PyTorch/torchvision read-only, plus
    project-local helper packages. No global package or driver was changed.
    OpenCV4.11 was selected to support the inherited NumPy1.26 runtime; local
    protobuf and h11/httpcore versions were aligned with their dependency requirements.
    The actual runtime paths and resolved versions are recorded. The full
    decode attempt was stopped before producing a completion report; only
    bounded audit/selected-source decode evidence is counted. Batch2 / CPU2
    threads is used for low-memory face training. A real frozen FaRL benchmark
    compared 2/2, 4/4, 4/8 and 8/8 thread/batch settings; CPU4/batch8 was selected
    for feature extraction. Actual measurements, scope and cold-case limitation
    are in `reports/inference_benchmark.json`. Full training feasibility is
    determined after measurement, not from total installed RAM alone.
# ResNet50 본 학습 실행 (2026-10-22 추가)

사용자의 본 학습 실행 지시에 따라 `phase1.main_train`을 추가했다. 기존 smoke/pilot
학습 코드와 결과는 보존한다. 원래 train의 480명 전체를 사용하며, 검토 제외 14장을
반영한 별도 고정 manifest는 학습 140,925장·내부 검증 35,459장이다. 원래 val 60명은
사용하지 않는다. 전체 수동 얼굴/신원 검토 및 전체 사전 디코딩을 완료했다는 뜻은 아니다.

ImageNetV2에서 새 480-class head와 전체 backbone을 학습한다. 기존 설정대로
10 epoch, batch2, CPU 2스레드, SGD lr0.001/momentum0.9/weight decay0.0001,
기존 resize232/center crop224/정규화, 추가 증강 없이 실행한다. 이 학습 일정의
선택은 구현 선택이며 프로포절의 최소 성능 보장이나 수렴 기준이 아니다.

300 step마다 원자적 last.pt를 저장하고 20 step마다 진행 JSON을 교체한다.
고정 epoch별 순열과 소비한 표본 수를 저장해 중간 재개 시 이미 처리한 사진을
다시 디코딩하지 않는다. optimizer momentum과 torch RNG도 복원한다.
재개 대조 테스트는 dropout을 포함한 소형 모델에서 가중치·momentum·RNG·손실의
정확 일치를 확인한다. 실제 ResNet50의 안전 중단·재개도 별도 기록한다.

백그라운드 실행 중에만 Windows의 유휴 절전을 방지하는 실행 상태 요청을 유지한다.
영구 전원 설정은 변경하지 않고, 표시장치 끄기와 사용자의 명시적 절전은 허용한다.
유료 서비스·원격 실행·드라이버 변경·전역 패키지 변경은 없다.
