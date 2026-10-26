from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from argparse import Namespace
import json
import runpy
import subprocess
from phase1.common import ROOT, sha256, write_json
from phase1.extract import extract
from scripts.execute_verified import stage


def feature_resume():
    before = sha256('features/smoke/farl/raw.npy')
    extract(Namespace(model='farl', run='smoke', backend='torch', batch_size=16,
                      threads=4, checkpoint='checkpoints/smoke_resnet/last.pt'))
    after = sha256('features/smoke/farl/raw.npy')
    validation = json.loads((ROOT / 'features/smoke/farl/validation.json').read_text())
    assert before == after and validation['computed_rows'] == 0 and validation['cached_rows'] == 400
    write_json('reports/feature_resume_verification.json', dict(passed=True, raw_sha256_before=before,
               raw_sha256_after=after, computed_rows=0, cached_rows=400,
               changed_batch_size_from=8, changed_batch_size_to=16, row_order_unchanged=True))


def source_and_git():
    copies = [('phase1/upstream_processors.py', 'third_party/processors_fork/feature_extraction/processors.py'),
              ('phase1/upstream_processors.py', 'third_party/processors_original/feature_extraction/processors.py'),
              ('phase1/clip_model.py', 'third_party/CLIP/clip/model.py')]
    source_checks = []
    for destination, source in copies:
        a,b = sha256(destination), sha256(source)
        assert a == b
        source_checks.append(dict(destination=destination, source=source, sha256=a, identical=True))
    private_paths = ['data/raw/vggface2.zip', 'weights/LVFace-T_Glint360K.pt',
                     'features/smoke/farl/raw.npy', 'checkpoints/smoke_resnet/last.pt', '.env', 'kaggle.json']
    for path in private_paths:
        subprocess.run(['git', 'check-ignore', '--quiet', '--', path], cwd=ROOT, check=True)
    remotes = subprocess.check_output(['git', 'remote', '-v'], cwd=ROOT, text=True).strip()
    write_json('reports/source_and_repository_verification.json', dict(passed=True, source_copies=source_checks,
               ignored_private_paths=private_paths, remote_output=remotes, published_or_pushed=False))


stage('verify-feature-resume', feature_resume, {'scope': '400 cached FaRL rows; changed batch size; exact SHA match'})
stage('verify-backbone-update', lambda: runpy.run_path(str(ROOT / 'scripts/verify_backbone_update.py'), run_name='__main__'), {})
stage('verify-unseen-embeddings', lambda: runpy.run_path(str(ROOT / 'scripts/verify_unseen_embeddings.py'), run_name='__main__'), {})
stage('verify-source-and-git', source_and_git, {})
stage('environment-final', lambda: runpy.run_path(str(ROOT / 'scripts/environment.py'), run_name='__main__'), {})
