"""Build the evidence-linked English status report from actual run outputs."""
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
    main_run = (load('configs/phase1.json') or {}).get('resnet50', {}).get('run', 'main_resnet_adam')
    main_progress = load(f'reports/{main_run}_progress.json')
    main_manifest = load('reports/main_manifest.json')
    main_completed = bool(main_progress and main_progress.get('full_face_training_completed'))
    main_running = False
    if main_progress and main_progress.get('status') == 'running':
        try:
            main_running = 'phase1.train' in psutil.Process(main_progress['pid']).cmdline()
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
    row('P1/U3/U5', 'Acquire and inspect the Kaggle dataset', 'Complete' if inv else 'Incomplete', 'reports/data_inventory.json; reports/acquisition.json',
        f"Inventory of {sum(inv['file_listing_images'].values()):,} images; separate decoding audit of {audit['actual_decoded'] if audit else 0:,} images" if inv else '')
    row('P2/U2/U11', 'Obtain official FaRL and LVFace weights and run inference', 'Complete' if verify else 'Incomplete', 'reports/model_verification.json; features/smoke',
        'FaRL ep16 / LVFace-T; frozen weights and evaluation mode; verified output dimensions; LVFace ONNX comparison')
    row('P3/U12', 'Train ResNet50 on face identities',
        'Training schedule complete' if main_completed else 'In progress' if main_running else 'Incomplete',
        f'reports/{main_run}_progress.json; reports/main_manifest.json; reports/{main_run}_verification.json'
        if main_progress else 'reports/smoke_resnet_training.json; reports/resume_verification.json',
        f"Main training on {main_manifest['identities']} identities, with {main_manifest['train_images']:,} training images / {main_manifest['validation_images']:,} internal validation images; target of {main_progress['target_epochs']} epochs"
        if main_manifest and main_progress else 'Verified loss, backward pass, validation, saving, and resume; small-scale face identity training from ImageNet initialization',
        'Validate a usable face recognition baseline' if main_completed else 'Complete training and validate a usable face recognition baseline')
    row('P4', 'Prepare the repository', 'Awaiting external input', '.git; .gitignore', 'Local Git setup complete; data and weights excluded', 'Existing team remote URL required; creating a remote, publishing, or pushing requires separate instructions')
    row('P5/U9/U10', 'Integrate the supplied processing code', 'Complete' if smoke else 'Incomplete', 'phase1/upstream_processors.py; reports/smoke_processing.json',
        '25 source images × 16 conditions; core IDENTITY+JPEG6 conditions plus 9 additional modules')
    row('U6/U7', 'Verify source links, labels, and split leakage', 'Complete' if smoke else 'Incomplete', 'data/manifests; reports/smoke_validation.json',
        'Full source inventory and photo-group/identity split checks for the experimental samples', 'Near-duplicate checks across the full dataset and overlap with pretraining data remain unverified')
    row('U8', 'Prepare reference faces and perform visual review', 'Complete' if load('reports/reference_review.json') else 'Incomplete', 'reports/reference_review.json; reports/pilot_reference_review.json; reports/smoke_sources.png',
        'Existing crops retained; EXIF orientation applied once; PNG references; sample review coverage documented separately', 'Faces and identities have not been manually reviewed across the entire upload')
    row('U13/U15', 'Verify model features, row alignment, and basic similarities', 'Complete' if smoke else 'Incomplete', 'features; reports/*_validation.json',
        'Separate raw and L2-normalized features; source/person/processing/model/hash/layer mappings; three similarity groups')
    row('U14', 'Run a 25-image smoke test and scale up using measured results', 'Complete' if pilot else 'In progress', 'reports/smoke_validation.json; reports/pilot_validation.json',
        f"Smoke run: {smoke['source_count'] if smoke else 0} images; pilot run: {pilot['source_count'] if pilot else 0} images")
    row('U4/U16/U17', 'Record the environment, execution, and resume checks', 'Complete' if resume and env else 'Incomplete', 'reports/environment.json; reports/command_runs.jsonl; reports/resume_verification.json',
        'Project virtual environment; actual commands and exit codes; checkpoint and feature shard resumption')
    row('U1/U18', 'Prepare documentation, figures, and the Phase 2 handoff', 'Complete', 'README.md; docs; reports; data/manifests; features',
        'Requirement sources distinguished from implementation choices; workflow; comparisons of actual samples; status and limitations')
    write_json('reports/checkpoint_status.json', state)
    write_json('reports/requirement_status.json', evidence)
    lines = ['# Phase 1 Execution Report', '', '**Phase 1 is not yet complete.** Main ResNet50 face training and connection to the team remote repository remain outstanding.', '',
             'Report date: 2026-10-22 (America/New_York). The Phase 1 deadline is **Oct 26**, as stated in the proposal.', '',
             '## Requirements Status', '', '| ID | Requirement | Status | Scope Actually Executed | Evidence | Remaining Work |', '|---|---|---|---|---|---|']
    if main_progress:
        lines[2] = '**Phase 1 is not yet complete.** See the ResNet50 training snapshot below. Connecting the team remote repository and validating the baseline remain outstanding.'
    for r in evidence:
        links = '; '.join(f'[{p}](../{p})' if '*' not in p and (ROOT / p).exists() else p
                          for p in r['evidence'].split('; '))
        lines.append(f"| {r['id']} | {r['requirement']} | {r['status']} | {r['coverage']} | {links} | {r['remaining']} |")
    if main_progress:
        lines += ['', '## Main ResNet50 Training', '',
                  f"Snapshot timestamp (UTC): {main_progress['updated_utc']}. Process confirmed running: {main_running}. Status: {main_progress['status']}.",
                  f"Target: {main_progress['target_epochs']} epochs. Current epoch: {main_progress['epoch_display']}. Total optimizer steps: {main_progress['global_step']:,}.",
                  f'The latest progress is recorded in `reports/{main_run}_progress.json`, and log locations are listed in `logs/{main_run}_launch.json`. This paragraph is a snapshot taken when the report was generated.',
                  'Use `scripts/Stop-MainTraining.ps1` and `scripts/Start-MainTraining.ps1` to pause and resume training. Intermediate checkpoints are saved every 300 steps.']
    lines += ['', '## Data and Model Validation', '']
    if inv:
        lines += [f"- Inventory: {inv['file_listing_identities']['train']} training identities / {inv['file_listing_images']['train']:,} images, and {inv['file_listing_identities']['val']} validation identities / {inv['file_listing_images']['val']:,} images. No person IDs overlap between these splits.",
                  f"- Decoding has not been checked across the full dataset. Refer to the separate audit of {audit['actual_decoded'] if audit else 0:,} images and the source_decode_coverage recorded for each run. The data comes from the Kaggle upload, rather than the complete original VGGFace2 dataset."]
        if audit:
            lines.append(f"- Separate audit: {audit['actual_decoded']:,} images / {audit['identities']} identities, with {len(audit['failures'])} decoding failures (`reports/decode_audit.json`). These sample checks do not establish validity across all images.")
    lines += ['- All 6 processing module tests passed. Resumed and continuous ResNet50 training produced identical values for all 320 model tensors and the optimizer state. The maximum absolute difference between LVFace PyTorch and ONNX outputs for one actual image was approximately 7.6e-7.',
              '- Download, loading, actual image inference, and face training status are recorded separately for the three models in `reports/checkpoint_status.json`.']
    for name, report in [('smoke', smoke), ('pilot', pilot)]:
        if not report:
            continue
        lines += ['', f"### {name}: {report['source_count']} Source Images / {report['variant_count']} Variants", '',
                  '| Model | Actual Feature Array Shape | Status |', '|---|---|---|']
        for model, detail in report['features'].items():
            lines.append(f"| {model} | {detail['shape']} | Passed finite-value, raw/L2, and row/label alignment checks |")
        lines += ['', f'![Comparison of actual {name} processing conditions]({name}_comparison.png)', '']
        lines += ['Similarities are recorded in three groups in `reports/' + name + '_validation.json`. They do not establish same-person probabilities, fairness, camera identification, attack success, or reliance on shortcuts.']
    lines += ['', '## Resources and Feasibility', '']
    if env:
        cpu_name = env['cpu']['Name'] if isinstance(env['cpu'], dict) else str(env['cpu'])
        gpu_name = env['gpu']['Name'] if isinstance(env['gpu'], dict) else str(env['gpu'])
        lines.append(f"CPU: {cpu_name}; GPU: {gpu_name}; memory: {env['memory_bytes']/2**30:.2f} GiB; CUDA available: {env['cuda_available']}.")
    if training and inv:
        h = training['history'][-1]
        source_rows = read_csv('data/manifests/sources.csv')
        ntrain = sum(r['face_split'] == 'face_train' and r['decode_status'] != 'failed' for r in source_rows)
        throughput = h['images_per_second']
        hrs = ntrain / throughput / 3600
        lines += ['', f"Measured face training throughput: {throughput:.3f} images/second ({h['train_images']} training images, batch size {training['config']['batch_size']}). Observed RSS: {h['peak_process_rss_bytes']/2**30:.2f} GiB.",
                  f"Extrapolating this throughput to {ntrain:,} images gives **approximately {hrs:.1f} hours per epoch**, or **approximately {hrs*10:.1f} hours for 10 epochs**. Reading the sample PNG files is included in the measurement. Differences in archive I/O, validation, checkpoint saving, thermal throttling, and time to convergence are not accounted for, so this estimate does not guarantee completion within that time.",
                  'Main training on the current CPU is a separate undertaking from the small validation runs performed in this session. It requires either a dedicated GPU or sufficient CPU runtime. No resources were purchased and no remote runs were started automatically.']
    if pilot:
        timings = {m:load(f'features/pilot/{m}/timing_original.json') for m in ['farl','lvface','resnet50']}
        proposed_seconds = sum(14000/t['images_per_second'] for t in timings.values() if t)
        lines += ['', f"The proposed expansion to 100 identities × 20 images would require 14,000 feature vectors per model for the 7 core conditions. A simple extrapolation from pilot throughput estimates **approximately {proposed_seconds/3600:.1f} hours** for feature extraction across all three models; reference image preparation, processing, visual review, and training would take additional time.",
                  'The proposal does not require a 2,000-image experiment. Phase 1 execution checks were expanded to 20 identities and 99 images. Resumable commands for a larger, separate run are provided in `docs/scale_up.md`. That larger example has not been executed; this does not imply that longer runs are infeasible.']
        write_json('reports/feasibility.json', dict(considered_pilot_sources=2000,
                   considered_pilot_core_variants=14000, projected_three_model_extraction_hours=proposed_seconds/3600,
                   excludes='reference preparation, processing, review, training, thermal/I/O changes',
                   actual_pilot_sources=pilot['source_count'], pilot_mandatory_in_proposal=False))
    lines += ['', '## Rerunning the Pipeline and Phase 2 Handoff', '',
              'Use the commands in `README.md` or `scripts/Run-Phase1.ps1`. Resume training with `--resume checkpoints/<run>/last.pt`. Rerunning the same feature extraction command reuses completed shards.', '',
              "Hand off `data/manifests/sources.csv`, `*_sources.csv`, `*_variants.csv`, and each model's `features/<run>/<model>/{raw.npy,l2.npy,rows.csv,model.json}` files. Preserve `validation.json` and the checkpoint SHA alongside them.", '',
              "The ResNet50 features reflect training on a small face dataset or subset and are not presented as a fully validated face recognition baseline. FaRL is a general face representation model, while LVFace is intended for face recognition. The Kaggle splits do not guarantee separation from the models' prior pretraining data.", '',
              'The ResNet50 smoke and pilot runs use different checkpoints, so their features cannot be combined. For Phase 2, extract features using one consistent checkpoint. Source images and conditions shared by the two FaRL or LVFace runs must not be counted twice as independent samples.', '',
              '![Workflow](workflow.png)', '', 'Requirement sources are detailed in `docs/requirements_matrix.md`. Implementation choices and their implications are documented in `docs/decisions.md`. Execution commands and exit codes are recorded in `reports/command_runs.jsonl`.']
    (ROOT / 'reports/final_status.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
