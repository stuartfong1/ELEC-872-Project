from pathlib import Path
import runpy
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from argparse import Namespace
from scripts.execute_verified import stage
from phase1.common import ROOT
from phase1.extract import extract
from phase1.validate import validate

parser = argparse.ArgumentParser()
parser.add_argument('--run', default='smoke')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--checkpoint', default='checkpoints/smoke_resnet/last.pt')
args = parser.parse_args()
for model in ['farl', 'lvface', 'resnet50']:
    params = dict(model=model, run=args.run, backend='torch', batch_size=args.batch_size,
                  threads=args.threads, checkpoint=args.checkpoint)
    stage('extract-' + args.run + '-' + model, lambda p=params: extract(Namespace(**p)), params)
stage('validate-' + args.run, lambda: validate(args.run), {'run': args.run})
if args.run == 'smoke':
    stage('verify-models', lambda: runpy.run_path(str(ROOT / 'scripts/verify_models.py'), run_name='__main__'), {})
    stage('environment', lambda: runpy.run_path(str(ROOT / 'scripts/environment.py'), run_name='__main__'), {})
