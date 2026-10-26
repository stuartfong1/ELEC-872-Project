import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def read_csv(path):
    with (ROOT / path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Refuse to write empty manifest without schema')
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with (ROOT / path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()
