"""Repair the initial shared logger's fixed entrypoint, preserving originals.
Actual invocations are taken from this session's explicit launch commands.
No outcomes, timing measurements, parameters or logs are changed.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from phase1.common import ROOT, write_json

path = ROOT / 'reports/command_runs.jsonl'
records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
corrections = []
for record in records:
    command = record.get('command', [])
    if len(command) != 2 or command[1] != 'scripts/execute_verified.py':
        continue
    name = record['name']
    corrected = None
    if name in ('tests-final', 'train-pilot', 'validate-pilot') or name.startswith('extract-pilot-'):
        corrected = [command[0], 'scripts/execute_pilot.py', '--identities', '20', '--per-identity', '5']
    elif record['utc_started'] >= '2026-10-22T22:41:00' and (
        name.startswith('extract-smoke-') or name in ('validate-smoke', 'verify-models', 'environment')):
        corrected = [command[0], 'scripts/execute_features.py', '--run', 'smoke', '--batch-size', '8', '--threads', '4']
    if corrected:
        record['recorded_command_original'] = command
        record['command'] = corrected
        record['command_correction_reason'] = 'shared logger initially used a fixed entrypoint; corrected to observed launch invocation'
        corrections.append(dict(name=name, utc_started=record['utc_started'], original=command, corrected=corrected))
path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
write_json('reports/execution_ledger_corrections.json', corrections)
print(json.dumps({'corrected_records': len(corrections), 'original_values_preserved': True}))
