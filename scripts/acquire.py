"""Acquire only the agreed dataset and pinned official model/source artifacts.
Run with the project Python, from repository root. Downloads resume via Range.
"""
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import subprocess
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
HF_REVISION = 'b12702ab1f5c721748e054a66dc90e1edd1f0724'
SOURCE_COMMITS = {
    'FacePerceiver/FaRL': '3fb46941cb4444183442eb09748c310114bcb846',
    'bytedance/LVFace': '80bd20c8bb78e56eb708941180d800a27c396a6b',
    'openai/CLIP': 'd05afc436d78f1c48dc0dbf8e5980a9d471f35f6',
    'cracker321/visual-encoder-traces': '7c4859213fbde95a410307fa62cb0e2aeef12593',
    'ryan-caesar-ramos/visual-encoder-traces': '7c4859213fbde95a410307fa62cb0e2aeef12593',
}


def download(url, path):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                "sha256": hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(), "reused": True}
    part = path.with_suffix(path.suffix + '.part')
    start = time.perf_counter()
    for attempt in range(5):
        offset = part.stat().st_size if part.exists() else 0
        try:
            with requests.get(url, headers={'Range': f'bytes={offset}-'} if offset else {},
                              stream=True, timeout=(30, 90)) as r:
                r.raise_for_status()
                mode = 'ab' if offset and r.status_code == 206 else 'wb'
                expected = int(r.headers.get('content-length', 0))
                received = 0
                with part.open(mode) as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
                        received += len(chunk)
                if expected and received != expected:
                    raise IOError(f'incomplete response: {received}/{expected}')
            part.replace(path)
            return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                    "sha256": hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
                    "seconds": time.perf_counter() - start, "url": url}
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2)


def source(repo, target):
    dest = ROOT / 'third_party' / target
    if not dest.exists():
        subprocess.run(['git', 'clone', '--depth', '1', f'https://github.com/{repo}.git', str(dest)], check=True)
    commit = subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != SOURCE_COMMITS[repo]:
        subprocess.run(['git', '-C', str(dest), 'fetch', '--depth', '1', 'origin', SOURCE_COMMITS[repo]], check=True)
        subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', SOURCE_COMMITS[repo]], check=True)
        commit = SOURCE_COMMITS[repo]
    return {'repository': repo, 'commit': commit, 'path': str(dest.relative_to(ROOT))}


def main():
    specs = [
        ('dataset', 'https://www.kaggle.com/api/v1/datasets/download/hearfool/vggface2?datasetVersionNumber=1', 'data/raw/vggface2.zip'),
        ('farl', 'https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep16.pth', 'weights/FaRL-Base-Patch16-LAIONFace20M-ep16.pth'),
    ]
    rev = HF_REVISION
    for ext in ['pt', 'onnx']:
        fn = f'LVFace-T_Glint360K/LVFace-T_Glint360K.{ext}'
        specs.append((f'lvface_{ext}', f'https://huggingface.co/bytedance-research/LVFace/resolve/{rev}/{fn}', f'weights/LVFace-T_Glint360K.{ext}'))
    out = ROOT / 'reports/acquisition.json'
    previous = json.loads(out.read_text()) if out.exists() else {}
    result = {'hf_revision': rev, 'downloads': {}, 'sources': []}
    with cf.ThreadPoolExecutor(max_workers=5) as pool:
        pending = {pool.submit(download, url, path): name for name, url, path in specs}
        pending[pool.submit(lambda: [source(r, n) for r, n in [
            ('FacePerceiver/FaRL', 'FaRL'), ('bytedance/LVFace', 'LVFace'),
            ('openai/CLIP', 'CLIP'), ('cracker321/visual-encoder-traces', 'processors_fork'),
            ('ryan-caesar-ramos/visual-encoder-traces', 'processors_original')]])] = 'sources'
        for fut in cf.as_completed(pending):
            name = pending[fut]
            try:
                value = fut.result()
                if name == 'sources':
                    result['sources'] = value
                else:
                    old = previous.get('downloads', {}).get(name, {})
                    result['downloads'][name] = {'status': 'downloaded', **old, **value}
                print(name, 'OK', flush=True)
            except Exception as e:
                result['downloads'][name] = {'status': 'failed', 'error': str(e)}
                print(name, str(e), flush=True)
            out.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
