"""CUDA-only contract, resume compatibility, and hardware-gated training tests."""
import argparse
import hashlib
import pytest
import torch
from torch import nn
from phase1 import train as training
from phase1.common import write_csv


def test_cuda_required_before_data_or_checkpoint_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'ROOT', tmp_path)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='no CPU fallback'):
        training.train(training.parser().parse_args([]))
    assert not list(tmp_path.iterdir())


def test_no_cpu_execution_option():
    with pytest.raises(SystemExit):
        training.parser().parse_args(['--device', 'cpu'])
    args = training.parser().parse_args([])
    assert args.gpu == 0 and args.optimizer == 'adam'
    assert not hasattr(args, 'device')


def test_obsolete_device_argument_is_rejected():
    with pytest.raises(ValueError, match='CUDA-only'):
        training.train(argparse.Namespace(device='cpu'))


def test_feature_extraction_also_requires_cuda(monkeypatch):
    from phase1.extract import extract
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='no CPU fallback'):
        extract(argparse.Namespace(gpu=0))


def test_cuda_device_selection(monkeypatch):
    selected = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'set_device', selected.append)
    assert str(training.require_cuda(1)) == 'cuda:1'
    assert selected == [1]
    with pytest.raises(ValueError, match='unavailable'):
        training.require_cuda(2)


def checkpoint_metadata():
    return dict(identity_map={'a': 0}, manifest_sha256='manifest', batch_size=2,
                initialization='imagenet', config=dict(lr=.001), epoch=1, offset=0)


def test_legacy_sgd_is_not_loaded_into_adam():
    args = training.parser().parse_args([])
    with pytest.raises(ValueError, match='optimizer differs'):
        training.validate_resume(checkpoint_metadata(), args, {'a': 0}, 'manifest')
    args.optimizer = 'sgd'
    training.validate_resume(checkpoint_metadata(), args, {'a': 0}, 'manifest')


@pytest.mark.parametrize('field,value', [('identity_map', {'b': 0}), ('manifest_sha256', 'other'),
                                        ('batch_size', 8), ('initialization', 'random'),
                                        ('sampler', 'unknown')])
def test_resume_rejects_changed_experiment(field, value):
    args = training.parser().parse_args(['--optimizer', 'sgd'])
    ck = checkpoint_metadata()
    ck[field] = value
    with pytest.raises(ValueError):
        training.validate_resume(ck, args, {'a': 0}, 'manifest')


def test_weight_initialization_and_resume_are_exclusive():
    with pytest.raises(SystemExit):
        training.parser().parse_args(['--resume', 'old.pt', '--initialize-from', 'old.pt'])


@pytest.fixture
def toy_cuda(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('Actual NVIDIA CUDA GPU required; no CPU substitute')
    monkeypatch.setattr(training, 'ROOT', tmp_path)
    rows = [dict(source_id=str(i), original_split='train', person_id=str(i % 2),
                 face_split='face_train' if i < 10 else 'face_val', photo_group=str(i)) for i in range(14)]
    manifest = tmp_path / 'sources.csv'
    write_csv(manifest, rows)

    class ToyDataset:
        def __init__(self, source_rows, mapping):
            self.rows, self.mapping = source_rows, mapping
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            row = self.rows[index]
            return torch.tensor([int(row['source_id']) / 14., 1.]), self.mapping[row['person_id']]

    monkeypatch.setattr(training, 'FaceDataset', ToyDataset)
    monkeypatch.setattr(training, 'new_resnet', lambda classes, initialization:
        nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Dropout(.3), nn.Linear(8, classes)))

    def args(run, extra=()):
        return training.parser().parse_args(['--manifest', str(manifest), '--run', run, '--epochs', '2',
            '--initialization', 'random', '--threads', '1', '--checkpoint-every', '2', '--progress-every', '2', *extra])
    return tmp_path, args


@pytest.mark.parametrize('optimizer', ['adam', 'sgd'])
def test_cuda_mid_epoch_resume_matches_continuous(toy_cuda, optimizer):
    root, args = toy_cuda
    extra = ['--optimizer', optimizer]
    training.train(args('continuous', extra))
    training.train(args('resumed', extra + ['--max-steps', '3']))
    path = root / 'checkpoints/resumed/last.pt'
    partial = torch.load(path, weights_only=True, map_location='cpu')
    assert partial['epoch'] == 0 and partial['offset'] == 6
    assert partial['device'].startswith('cuda:') and partial['cuda_rng_state'] is not None
    training.train(args('resumed', extra + ['--resume', str(path)]))
    a = torch.load(root / 'checkpoints/continuous/last.pt', weights_only=True, map_location='cpu')
    b = torch.load(path, weights_only=True, map_location='cpu')
    assert a['epoch'] == b['epoch'] == 2 and a['global_step'] == b['global_step'] == 10
    assert (root / 'reports/resumed_training.json').is_file()
    for key in a['model']:
        assert torch.equal(a['model'][key], b['model'][key]), key
    for key, state in a['optimizer']['state'].items():
        for name, value in state.items():
            other = b['optimizer']['state'][key][name]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    assert torch.equal(a['torch_rng_state'], b['torch_rng_state'])
    assert torch.equal(a['cuda_rng_state'], b['cuda_rng_state'])
    for h1, h2 in zip(a['history'], b['history']):
        for field in ('train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy'):
            assert h1[field] == h2[field]


def test_cuda_initialize_weights_resets_optimizer_and_counters(toy_cuda):
    root, args = toy_cuda
    training.train(args('old_sgd', ['--optimizer', 'sgd', '--max-steps', '3']))
    old = root / 'checkpoints/old_sgd/last.pt'
    old_hash = hashlib.sha256(old.read_bytes()).hexdigest()
    training.train(args('new_adam', ['--initialize-from', str(old), '--max-steps', '1']))
    ck = torch.load(root / 'checkpoints/new_adam/last.pt', weights_only=True, map_location='cpu')
    assert ck['optimizer_name'] == 'adam' and ck['global_step'] == 1 and ck['offset'] == 2
    assert ck['initialized_from']['sha256'] == old_hash
    assert ck['initialized_from']['optimizer_reset']
    assert hashlib.sha256(old.read_bytes()).hexdigest() == old_hash


def test_cuda_legacy_epoch_boundary_checkpoint(toy_cuda):
    root, args = toy_cuda
    one = args('legacy', ['--optimizer', 'sgd'])
    one.epochs = 1
    training.train(one)
    path = root / 'checkpoints/legacy/last.pt'
    ck = torch.load(path, weights_only=True, map_location='cpu')
    for field in ['offset', 'global_step', 'partial_stats', 'sampler', 'cuda_rng_state', 'optimizer_name']:
        ck.pop(field, None)
    ck['config'].pop('optimizer', None)
    torch.save(ck, path)
    training.train(args('legacy', ['--optimizer', 'sgd', '--resume', str(path)]))
    result = torch.load(path, weights_only=True, map_location='cpu')
    assert result['epoch'] == 2 and result['global_step'] == 10


def test_cuda_completed_checkpoint_is_preserved(toy_cuda):
    root, args = toy_cuda
    training.train(args('done'))
    path = root / 'checkpoints/done/last.pt'
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    training.train(args('done', ['--resume', str(path)]))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
