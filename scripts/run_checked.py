"""Run a CLI command with durable command, return code, timing and log evidence.
Example: python scripts/run_checked.py smoke-tests -m pytest -q
"""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
name, *arguments = sys.argv[1:]
if not arguments:
    raise SystemExit('supply a record name and Python arguments')
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
start = time.perf_counter()
log = root / 'logs' / (name + '.log')
with log.open('w', encoding='utf-8') as f:
    process = subprocess.run([sys.executable, *arguments], cwd=root, stdout=f, stderr=subprocess.STDOUT,
                             env={**__import__('os').environ, 'PYTHONUTF8': '1'})
record = dict(name=name, utc_started=stamp, command=[sys.executable, *arguments],
              returncode=process.returncode, seconds=time.perf_counter() - start,
              log=str(log.relative_to(root)), outcome='passed' if process.returncode == 0 else 'failed')
with (root / 'reports/command_runs.jsonl').open('a', encoding='utf-8') as f:
    f.write(json.dumps(record) + '\n')
print(json.dumps(record))
raise SystemExit(process.returncode)
