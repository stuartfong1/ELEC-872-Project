# ELEC872 Phase 1

실제 실행 상태와 한계: [reports/final_status.md](reports/final_status.md).
요구사항 출처: [docs/requirements_matrix.md](docs/requirements_matrix.md).
프로포절 밖의 선택: [docs/decisions.md](docs/decisions.md).

PowerShell에서 이 폴더를 작업 위치로 사용합니다. 기존 `tmp` 자료는 보존했습니다.
데이터·모델은 공개 출처에서 확보하며 별도의 유료 자원은 사용하지 않습니다.

```powershell
./scripts/Setup-Environment.ps1
.\.venv-run\Scripts\python.exe -m pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv-run\Scripts\python.exe scripts/acquire.py
.\.venv-run\Scripts\python.exe scripts/environment.py
.\.venv-run\Scripts\python.exe -m phase1.data inventory
.\.venv-run\Scripts\python.exe scripts/decode_audit.py
.\.venv-run\Scripts\python.exe -m phase1.data prepare --run smoke --identities 5 --per-identity 5 --extras
.\.venv-run\Scripts\python.exe -m pytest -q
.\.venv-run\Scripts\python.exe -m phase1.train --run smoke_resnet --epochs 1
.\.venv-run\Scripts\python.exe -m phase1.train --run smoke_resnet --epochs 2 --resume checkpoints/smoke_resnet/last.pt
.\.venv-run\Scripts\python.exe -m phase1.extract --model farl --run smoke
.\.venv-run\Scripts\python.exe -m phase1.extract --model lvface --run smoke
.\.venv-run\Scripts\python.exe -m phase1.extract --model resnet50 --run smoke
.\.venv-run\Scripts\python.exe -m phase1.validate --run smoke
```

이 세션과 같은 소규모 검증을 순서대로 실행하려면:

```powershell
.\.venv-run\Scripts\python.exe scripts/execute_verified.py
```

이 명령은 이미 준비된 smoke 목록을 보존하고, 실제 테스트·학습/재개/연속학습
대조·세 모델 특징 추출·행 정렬 검증을 실행합니다. 각 단계의 로그와 종료코드를
남기며, 첫 오류에서 멈춰 `logs/<단계>.log`에 원인을 보존합니다.

검증 실행 환경은 `.venv-run`입니다. 전역 패키지를 **읽기 전용으로 공유**하며
전역 패키지는 변경하지 않았습니다. 첫 독립 설치 시도 `.venv`는 중단했고,
이 세션의 `.venv-run/Lib/site-packages/project-shared.pth`가 거기에 이미 설치된
보조 의존성을 읽습니다. 두 폴더를 함께 보존하세요. 새 설치에서는 위 setup
스크립트가 필요한 패키지를 프로젝트 venv에 설치하므로 이 보조 경로가 필요 없습니다.
실제 import 경로/버전은 `reports/environment.json`에 기록합니다.

`smoke --extras`는 25장 × 16조건입니다. 그중 핵심 JPEG 경로는 7조건입니다.
추가 조건을 제외한 독립 실행은 새 run 이름과 `--extras` 생략으로 구성합니다.
`reports/*_sources.png`를 검토해 여러 얼굴/불명확한 대상을 표시한 뒤 사용합니다.

다운로드 중단 시 `scripts/acquire.py`를 다시 실행합니다. `.part`에 Range를
사용해 이어받습니다. 특징 추출 재실행은 완료된 shard를 확인하고 재사용합니다.
원본 manifest/체크포인트가 달라지면 새 run 폴더를 사용해야 합니다.
같은 run에서 추출 배치 크기를 바꿔도 이미 저장된 특징을 variant ID로 다시
연결해 재사용합니다. 모델과 입력 manifest 검증이 일치해야만 재사용합니다.
학습 재개는 **완료한 epoch 경계** 기준이며 optimizer와 identity map을 복원합니다.
학습 결과의 loss/accuracy는 소규모 실행 검증이며 얼굴 인식 성능 입증이 아닙니다.

전체 업로드의 자체 얼굴 학습은 별도 백그라운드 실행기로 시작·재개합니다:

```powershell
.\scripts\Start-MainTraining.ps1
# 현재 배치를 마치고 저장한 다음 중단:
.\scripts\Stop-MainTraining.ps1
# 목표 2 epoch 미완료 시 last.pt에서 자동 재개하며, 검증까지 완료됐으면 시작하지 않습니다.
Get-Content reports/main_resnet_progress.json
```

본 학습은 `phase1.main_train`을 사용합니다. 원래 train 480명의 사진에서 기존
검토 제외 14장을 뺀 **140,925장 학습 / 35,459장 내부 검증**, ImageNetV2 초기화,
전체 backbone 학습, 목표 2 epoch, batch2, CPU 2스레드, SGD lr0.001입니다.
2026-10-26 사용자의 요청으로 기존 10 epoch에서 2 epoch로 제한했습니다.
요청 당시 epoch 2 검증은 이미 완료됐고, 마지막 모델은 epoch 3 일부도 학습한 상태입니다.
epoch 2의 검증 결과는 남아 있지만 해당 시점만의 모델 파일은 따로 보존되지 않았으므로,
현재 `last.pt`를 정확히 2 epoch 학습한 모델로 표시하면 안 됩니다.
원래 val 60명은 포함하지 않습니다. 정확한 목록과 제외 이유는
`data/manifests/main_sources.csv`, `reports/main_manifest.json`에 보존합니다.
전체 이미지 디코딩/신원 수동 검토 완료를 뜻하지 않으며, 디코딩 오류 발생 시
원본 식별자를 기록하고 중단합니다. 임의로 이미지를 건너뛰지 않습니다.

20 step마다 진행률·손실을, 첫 10 step과 이후 300 step마다 재개 상태를 저장합니다.
`checkpoints/main_resnet/last.pt`는 epoch 중간의 표본 위치·가중치·SGD momentum·
난수 상태를 포함합니다. 갑작스러운 종료 후에는 마지막 저장 이후만 다시 계산합니다.
검증 중 중단하면 해당 검증을 처음부터 다시 실행합니다. 중간 재개와 연속 실행의
일치는 `tests/test_main_training.py`에서 검증했습니다.
진행 기록은 `reports/main_resnet_progress.json`, epoch별 검증 결과는
`reports/main_resnet_training.json`, 실행별 로그/PID는 `logs/main_resnet_launch.json`입니다.
진행 JSON은 마지막 기록이므로 비정상 종료 여부는 기록된 PID의 실행 상태도 확인합니다.
숨겨진 별도 프로세스로 실행됩니다. 실행 중 Windows 자동 유휴 절전을 임시로
방지하며, 영구 전원 설정은 변경하지 않습니다. 화면은 꺼질 수 있습니다.
Windows 종료·명시적 절전·노트북 덮개 닫기 중에는 진행하지 않습니다.
CPU에서 며칠 이상 걸릴 수 있으며, 설정한 epoch의 완료 여부와 인식 성능은 구분합니다.

처음 모델 생성은 공식 ImageNetV2 가중치를 `weights/torch_cache`에 받습니다.
ImageNet 가중치 확보 자체를 얼굴 학습 완료로 취급하지 않습니다. 원래 Kaggle
val은 학습하지 않은 사람들의 임베딩 비교를 위한 보류 자료이며 identity head의
닫힌 집합 validation에 섞지 않습니다. 임베딩 비교는 `load_resnet`의 avgpool
출력으로 구성할 수 있으며 Phase 1에서 최종 평가를 반복하지 않습니다.

Phase 2 인계: `sources.csv`, 각 run의 source/variant CSV, 모델별 `raw.npy`,
`l2.npy`, `rows.csv`, `model.json`, `validation.json`. 동일 행 정렬 검증을 통과한
자료만 사용하고 모델 간 좌표계를 공유한다고 가정하지 않습니다.
ResNet50의 smoke와 pilot은 **서로 다른 체크포인트**이므로 특징을 합치지 마세요.
Phase 2에서는 한 체크포인트로 통일해 다시 추출해야 합니다. FaRL·LVFace는
각각 동일한 공식 가중치를 유지하며, 두 run의 겹치는 원본/조건은 중복 집계하지 않습니다.

실측 기반 확장 실행(20명 × 5장, JPEG 핵심 7조건):

```powershell
.\.venv-run\Scripts\python.exe scripts/execute_pilot.py --identities 20 --per-identity 5
```

이 실행의 ResNet50도 2 epoch 부분집합 학습이며, 완성된 인식 baseline이 아닙니다.
현재 저장된 pilot은 시각 검토 후 **20명·99장·693개 변형**입니다. 100장 목표에서
마지막 다중 얼굴 사진 1장을 제외했고, 제외 원본과 이유를 따로 보존했습니다.
기존 pilot manifest가 있으면 이 검토 결과를 유지해 재실행합니다.
얼굴 학습은 batch2/2스레드, 특징 추출은 실측으로 선택한 batch8/4스레드입니다.
성능 설정 선택 근거는 `reports/inference_benchmark.json`에 있습니다.
2,000장 확대 명령은 [docs/scale_up.md](docs/scale_up.md)에 별도로 제공합니다.
이는 미실행 예시이며 현재 완료 범위에 포함하지 않습니다.
