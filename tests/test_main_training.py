"""Check exact mid-epoch restart, including stochastic model RNG and SGD momentum."""
import torch
from torch import nn
from phase1 import main_train
from phase1.common import write_csv


def test_mid_epoch_resume_matches_continuous(tmp_path, monkeypatch):
    monkeypatch.setattr(main_train, 'ROOT', tmp_path)
    rows = [dict(source_id=str(i), original_split='train', person_id=str(i % 2),
                 face_split='face_train' if i < 10 else 'face_val', photo_group=str(i))
            for i in range(14)]
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

    monkeypatch.setattr(main_train, 'MainFaceDataset', ToyDataset)
    monkeypatch.setattr(main_train, 'new_resnet', lambda classes, initialization:
                        nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Dropout(.3), nn.Linear(8, classes)))

    def arguments(run, extra=()):
        return main_train.parser().parse_args([
            '--manifest', str(manifest), '--run', run, '--epochs', '2',
            '--initialization', 'random', '--threads', '1',
            '--checkpoint-every', '2', '--progress-every', '2', *extra])

    main_train.train(arguments('continuous'))
    main_train.train(arguments('resumed', ['--max-steps', '3']))
    resume = tmp_path / 'checkpoints/resumed/last.pt'
    partial = torch.load(resume, weights_only=True)
    assert partial['epoch'] == 0 and partial['offset'] == 6
    main_train.train(arguments('resumed', ['--resume', str(resume)]))
    a = torch.load(tmp_path / 'checkpoints/continuous/last.pt', weights_only=True)
    b = torch.load(resume, weights_only=True)
    assert a['epoch'] == b['epoch'] == 2 and a['global_step'] == b['global_step'] == 10
    for key, value in a['model'].items():
        assert torch.equal(value, b['model'][key]), key
    for key, value in a['optimizer']['state'].items():
        assert torch.equal(value['momentum_buffer'], b['optimizer']['state'][key]['momentum_buffer'])
    assert torch.equal(a['torch_rng_state'], b['torch_rng_state'])
    for h1, h2 in zip(a['history'], b['history']):
        for field in ('train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy'):
            assert h1[field] == h2[field]
