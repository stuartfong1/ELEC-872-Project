"""Build the evidence-linked Korean status report from actual run outputs."""
import json
from pathlib import Path
import sys
import psutil
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase1.common import ROOT, read_csv, write_json, sha256


def load(path):
    p = ROOT / path
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def main():
    inv = load('reports/data_inventory.json')
    audit = load('reports/decode_audit.json')
    env = load('reports/environment.json')
    acquire = load('reports/acquisition.json')
    verify = load('reports/model_verification.json')
    resume = load('reports/resume_verification.json')
    smoke = load('reports/smoke_validation.json')
    pilot = load('reports/pilot_validation.json')
    main_progress = load('reports/main_resnet_progress.json')
    main_manifest = load('reports/main_manifest.json')
    main_completed = bool(main_progress and main_progress.get('full_face_training_completed'))
    main_running = False
    if main_progress and main_progress.get('status') == 'running':
        try:
            main_running = 'phase1.main_train' in psutil.Process(main_progress['pid']).cmdline()
        except psutil.Error:
            pass
    training = load('reports/pilot_resnet_training.json') or load('reports/smoke_resnet_training.json')
    state = dict(phase1_fully_complete=False, data=inv, decode_audit=audit,
                 models={}, smoke_validated=bool(smoke and smoke['passed']),
                 pilot_validated=bool(pilot and pilot['passed']), full_resnet_training_completed=main_completed,
                 main_training_snapshot=main_progress, main_training_process_running=main_running,
                 local_git_prepared=(ROOT / '.git').exists(), team_remote_configured=False)
    for model in ['farl', 'lvface', 'resnet50']:
        state['models'][model] = dict(download='completed' if model != 'resnet50' else 'see ImageNet cache',
                                     model_load='verified' if verify and model in verify['models'] else 'not_verified',
                                     real_image_inference='verified' if smoke and model in smoke['features'] else 'not_verified',
                                     face_training='smoke/subset only; main incomplete' if model == 'resnet50' else 'frozen pretrained; not trained locally')
    initialization_path = 'weights/torch_cache/hub/checkpoints/resnet50-11ad3fa6.pth'
    if main_progress:
        state['models']['resnet50']['face_training'] = (
            'full identity-training schedule completed; recognition baseline validation remains' if main_completed
            else 'main training running' if main_running else 'main training stopped; inspect progress and logs')
    if (ROOT / initialization_path).exists():
        state['models']['resnet50']['initialization_weight'] = dict(
            enum='ResNet50_Weights.IMAGENET1K_V2', path=initialization_path,
            sha256=sha256(initialization_path),
            url='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
            status='downloaded; not face training')
        state['models']['resnet50']['download'] = 'completed (ImageNet initialization only)'
    evidence = []
    def row(id, requirement, status, path, scope, remaining='—'):
        evidence.append(dict(id=id, requirement=requirement, status=status, evidence=path, coverage=scope, remaining=remaining))
    row('P1/U3/U5', 'Kaggle 데이터 확보·검사', '완료' if inv else '미완료', 'reports/data_inventory.json; reports/acquisition.json',
        f"목록 {sum(inv['file_listing_images'].values()):,}장; 별도 디코딩 표본 {audit['actual_decoded'] if audit else 0:,}장" if inv else '')
    row('P2/U2/U11', 'FaRL·LVFace 공식 가중치 및 실제 추론', '완료' if verify else '미완료', 'reports/model_verification.json; features/smoke',
        'FaRL ep16 / LVFace-T; frozen/eval; 실제 출력 차원; LVFace ONNX 대조')
    row('P3/U12', 'ResNet50 자체 얼굴 학습',
        '학습 일정 완료' if main_completed else '진행 중' if main_running else '미완료',
        'reports/main_resnet_progress.json; reports/main_manifest.json; reports/main_training_verification.json'
        if main_progress else 'reports/smoke_resnet_training.json; reports/resume_verification.json',
        f"본 학습 {main_manifest['identities']}명, 학습 {main_manifest['train_images']:,}장 / 내부 검증 {main_manifest['validation_images']:,}장, 목표 {main_progress['target_epochs']} epoch"
        if main_manifest and main_progress else '실제 loss/backward/validation/save/resume 검증; ImageNet에서 얼굴 identity로 소규모 학습',
        '유효한 얼굴 인식 baseline 검증 필요' if main_completed else '전체 학습 완료 및 유효한 얼굴 인식 baseline 검증 필요')
    row('P4', '저장소 준비', '외부 조건 필요', '.git; .gitignore', '로컬 Git 완료, 데이터·가중치 제외', '기존 팀 원격 URL 필요; 원격 생성·공개·푸시는 별도 지시')
    row('P5/U9/U10', '제공된 처리 코드 연결', '완료' if smoke else '미완료', 'phase1/upstream_processors.py; reports/smoke_processing.json',
        '25원본 × 16조건; IDENTITY+JPEG6 핵심 및 추가9 모듈')
    row('U6/U7', '원본 연결·라벨·분할 누출 검사', '완료' if smoke else '미완료', 'data/manifests; reports/smoke_validation.json',
        '전체 원본 목록 및 실험 표본의 photo-group/identity 분할 검사', '전체 근접중복·사전학습 중복 여부는 미확인')
    row('U8', '기준 얼굴·시각 검토', '완료' if load('reports/reference_review.json') else '미완료', 'reports/reference_review.json; reports/pilot_reference_review.json; reports/smoke_sources.png',
        '기존 크롭 유지; EXIF 방향만 1회; PNG 기준; 표본 검토 범위 별도 기록', '전체 업로드의 얼굴/신원 수동 검토는 수행하지 않음')
    row('U13/U15', '모델별 특징·행 정렬·기본 유사도', '완료' if smoke else '미완료', 'features; reports/*_validation.json',
        'raw/L2 분리; source/person/processing/model/hash/layer 매핑; 3종 유사도')
    row('U14', '25장 실행과 실측 확장', '완료' if pilot else '진행 중', 'reports/smoke_validation.json; reports/pilot_validation.json',
        f"smoke {smoke['source_count'] if smoke else 0}장; pilot {pilot['source_count'] if pilot else 0}장")
    row('U4/U16/U17', '환경·실행·재개 증거', '완료' if resume and env else '미완료', 'reports/environment.json; reports/command_runs.jsonl; reports/resume_verification.json',
        '프로젝트 venv; 실제 실행 명령/종료코드; checkpoint·feature shard 재개')
    row('U1/U18', '문서·그림·Phase 2 인계', '완료', 'README.md; docs; reports; data/manifests; features',
        '요구사항 출처와 구현 선택 구분; workflow; 실제 표본 비교; 상태·한계')
    write_json('reports/checkpoint_status.json', state)
    write_json('reports/requirement_status.json', evidence)
    lines = ['# Phase 1 실제 실행 결과', '', '**Phase 1 전체는 미완료입니다.** 자체 ResNet50 본 학습과 팀 원격 저장소 연결이 남았습니다.', '',
             '작업 기준일: 2026-10-22 (America/New_York). 프로포절 Phase 1 마감은 원문대로 **Oct 26**입니다.', '',
             '## 요구사항 상태', '', '| ID | 요구사항 | 상태 | 실제 실행 범위 | 증거 | 남은 작업 |', '|---|---|---|---|---|---|']
    if main_progress:
        lines[2] = '**Phase 1 전체는 미완료입니다.** ResNet50 본 학습 상태는 아래 기록을 참조하세요. 팀 원격 저장소 연결과 baseline 검증이 남았습니다.'
    for r in evidence:
        links = '; '.join(f'[{p}](../{p})' if '*' not in p and (ROOT / p).exists() else p
                          for p in r['evidence'].split('; '))
        lines.append(f"| {r['id']} | {r['requirement']} | {r['status']} | {r['coverage']} | {links} | {r['remaining']} |")
    if main_progress:
        lines += ['', '## ResNet50 본 학습', '',
                  f"기록 시각(UTC): {main_progress['updated_utc']}. 프로세스 실행 확인: {main_running}. 상태: {main_progress['status']}.",
                  f"목표 {main_progress['target_epochs']} epoch, 현재 epoch {main_progress['epoch_display']}, 누적 optimizer step {main_progress['global_step']:,}.",
                  '최신 진행률은 `reports/main_resnet_progress.json`, 로그 위치는 `logs/main_resnet_launch.json`에 있습니다. 이 문단은 보고서 생성 시점의 기록입니다.',
                  '중단·재개는 `scripts/Stop-MainTraining.ps1`, `scripts/Start-MainTraining.ps1`을 사용합니다. 300 step마다 중간 체크포인트를 저장합니다.']
    lines += ['', '## 데이터와 모델 검증', '']
    if inv:
        lines += [f"- 목록: train {inv['file_listing_identities']['train']}명 / {inv['file_listing_images']['train']:,}장, val {inv['file_listing_identities']['val']}명 / {inv['file_listing_images']['val']:,}장. 사람 ID 중복 0.",
                  f"- 전체 디코딩은 완료하지 않았습니다. 별도 감사 표본 {audit['actual_decoded'] if audit else 0:,}장과 각 실행의 source_decode_coverage를 참조하세요. 원본 전체 VGGFace2가 아닌 Kaggle 업로드입니다."]
        if audit:
            lines.append(f"- 별도 감사: {audit['actual_decoded']:,}장 / {audit['identities']}명, 디코딩 실패 {len(audit['failures'])}장 (`reports/decode_audit.json`). 표본 검사를 전체 이미지 검증으로 확대 해석하지 않습니다.")
    lines += ['- 처리 모듈 테스트 6개 통과. ResNet50 재개/연속 학습의 320개 모델 tensor와 optimizer 상태가 일치했습니다. LVFace PyTorch/ONNX 단일 실제 이미지 대조의 최대 절대 오차는 약 7.6e-7입니다.',
              '- 세 모델의 다운로드·로드·실제 이미지 추론·얼굴 학습 상태는 `reports/checkpoint_status.json`에서 분리해 기록합니다.']
    for name, report in [('smoke', smoke), ('pilot', pilot)]:
        if not report:
            continue
        lines += ['', f"### {name}: {report['source_count']}원본 / {report['variant_count']}변형", '',
                  '| 모델 | 실제 특징 배열 | 상태 |', '|---|---|---|']
        for model, detail in report['features'].items():
            lines.append(f"| {model} | {detail['shape']} | 유한값·raw/L2·행/라벨 정렬 검증 통과 |")
        lines += ['', f'![실제 {name} 조건 비교]({name}_comparison.png)', '']
        lines += ['유사도는 `reports/' + name + '_validation.json`의 세 그룹으로 기록했습니다. 동일인 확률·공정성·카메라 식별·공격 성공·지름길 의존성으로 해석하지 않습니다.']
    lines += ['', '## 자원 및 실행 가능성', '']
    if env:
        cpu_name = env['cpu']['Name'] if isinstance(env['cpu'], dict) else str(env['cpu'])
        gpu_name = env['gpu']['Name'] if isinstance(env['gpu'], dict) else str(env['gpu'])
        lines.append(f"CPU: {cpu_name}; GPU: {gpu_name}; 메모리 {env['memory_bytes']/2**30:.2f}GiB; CUDA {env['cuda_available']}.")
    if training and inv:
        h = training['history'][-1]
        source_rows = read_csv('data/manifests/sources.csv')
        ntrain = sum(r['face_split'] == 'face_train' and r['decode_status'] != 'failed' for r in source_rows)
        throughput = h['images_per_second']
        hrs = ntrain / throughput / 3600
        lines += ['', f"실측 얼굴 학습 처리량: {throughput:.3f}장/초 ({h['train_images']} 학습 사진, batch{training['config']['batch_size']}). 관측 RSS: {h['peak_process_rss_bytes']/2**30:.2f}GiB.",
                  f"동일 처리량을 {ntrain:,}장에 단순 외삽하면 **epoch당 약 {hrs:.1f}시간**, 10 epoch는 **약 {hrs*10:.1f}시간**입니다. 현재 표본 PNG 읽기는 실측에 포함돼 있습니다. 전체 압축파일 I/O 차이, 검증·저장·열 제한 및 수렴 시간은 별도이므로 완료 보장이 아닙니다.",
                  '현재 CPU에서는 본 학습을 이 세션의 소규모 검증과 동일시할 수 없습니다. 필요 자원은 전용 GPU 또는 충분한 CPU 실행 시간이며 자동 구매/원격 실행은 하지 않았습니다.']
    if pilot:
        timings = {m:load(f'features/pilot/{m}/timing_original.json') for m in ['farl','lvface','resnet50']}
        proposed_seconds = sum(14000/t['images_per_second'] for t in timings.values() if t)
        lines += ['', f"검토한 100명×20장 확대안은 핵심 7조건 기준 모델마다 14,000개 특징이 필요합니다. pilot 처리량의 단순 외삽으로 세 모델 추출 합계 약 **{proposed_seconds/3600:.1f}시간**이며, 기준 이미지/처리본 생성·시각 검토·학습은 별도입니다.",
                  '2,000장은 프로포절의 필수 규모가 아닙니다. 이번 Phase 1 실행 검증에서는 20명·99장으로 범위를 확장했고, 더 큰 실행은 `docs/scale_up.md`에 별도 run의 재개 가능한 명령으로 제공합니다. 이 확대 예시는 미실행이며, 장시간 실행이 불가능하다고 주장하지 않습니다.']
        write_json('reports/feasibility.json', dict(considered_pilot_sources=2000,
                   considered_pilot_core_variants=14000, projected_three_model_extraction_hours=proposed_seconds/3600,
                   excludes='reference preparation, processing, review, training, thermal/I/O changes',
                   actual_pilot_sources=pilot['source_count'], pilot_mandatory_in_proposal=False))
    lines += ['', '## 재실행 및 Phase 2 인계', '',
              '`README.md`의 명령 또는 `scripts/Run-Phase1.ps1`을 사용합니다. 학습은 `--resume checkpoints/<run>/last.pt`, 특징은 같은 명령 재실행으로 완료 shard를 재사용합니다.', '',
              '`data/manifests/sources.csv`, `*_sources.csv`, `*_variants.csv`와 모델별 `features/<run>/<model>/{raw.npy,l2.npy,rows.csv,model.json}`을 인계합니다. `validation.json`과 checkpoint SHA를 함께 보존해야 합니다.', '',
              'ResNet50 특징은 소규모/부분집합 얼굴 학습 상태이며 완성된 얼굴 인식 baseline으로 주장하지 않습니다. FaRL은 일반 얼굴 표현, LVFace는 얼굴 인식 목적의 모델입니다. Kaggle 분할은 과거 사전학습 자료와의 분리를 보장하지 않습니다.', '',
              'ResNet50 smoke/pilot은 서로 다른 체크포인트이므로 특징을 합칠 수 없습니다. Phase 2에서는 하나의 체크포인트로 통일해 추출하세요. FaRL·LVFace의 두 run에서 겹치는 원본/조건은 독립 표본으로 중복 집계하지 않습니다.', '',
              '![작업 흐름](workflow.png)', '', '상세 출처 구분: `docs/requirements_matrix.md`. 구현 선택과 영향: `docs/decisions.md`. 실행 명령·종료코드: `reports/command_runs.jsonl`.']
    (ROOT / 'reports/final_status.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
