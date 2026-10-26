"""Own face-identity training of the proposal's ResNet50; CPU/GPU and resume."""
import argparse
import io
import json
import random
import time
import zipfile
import numpy as np
import psutil
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from .common import ROOT, read_csv, write_json, sha256
from .models import new_resnet, resnet_transform


class FaceDataset(Dataset):
    def __init__(self, rows, mapping):
        self.rows, self.mapping = rows, mapping
        self.transform = resnet_transform()
        self.archives = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        ref = row.get('reference_path')
        if ref and (ROOT / ref).exists():
            with Image.open(ROOT / ref) as im:
                image = im.convert('RGB')
        else:
            archive = row['archive']
            if archive not in self.archives:
                self.archives[archive] = zipfile.ZipFile(ROOT / archive)
            with Image.open(io.BytesIO(self.archives[archive].read(row['archive_member']))) as im:
                image = ImageOps.exif_transpose(im).convert('RGB')
        return self.transform(image), self.mapping[row['person_id']]


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(872)
    random.seed(872)
    np.random.seed(872)
    rows = read_csv(args.manifest)
    train_rows = [r for r in rows if r['face_split'] == 'face_train' and r['decode_status'] != 'failed']
    val_rows = [r for r in rows if r['face_split'] == 'face_val' and r['decode_status'] != 'failed']
    mapping = {p: i for i, p in enumerate(sorted({r['person_id'] for r in train_rows}))}
    if not train_rows or not val_rows or not {r['person_id'] for r in val_rows} <= set(mapping):
        raise ValueError('Need within-identity train and validation images')
    assert not ({r['photo_group'] for r in train_rows} & {r['photo_group'] for r in val_rows})
    output = ROOT / 'checkpoints' / args.run
    output.mkdir(parents=True, exist_ok=True)
    write_json(f'checkpoints/{args.run}/identity_map.json', mapping)
    device = torch.device(args.device)
    manifest_sha = sha256(args.manifest)
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(ROOT / args.resume, map_location='cpu', weights_only=True)
        if checkpoint['identity_map'] != mapping or checkpoint['manifest_sha256'] != manifest_sha:
            raise ValueError('Resume manifest or identity mapping changed')
        if checkpoint['batch_size'] != args.batch_size or checkpoint['initialization'] != args.initialization:
            raise ValueError('Resume training settings changed')
    model = new_resnet(len(mapping), 'random' if checkpoint else args.initialization).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9, weight_decay=1e-4)
    start_epoch, history = 0, []
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch, history = checkpoint['epoch'], checkpoint['history']
    train_data = FaceDataset(train_rows, mapping)
    val_loader = DataLoader(FaceDataset(val_rows, mapping), batch_size=args.batch_size, shuffle=False, num_workers=0)
    overall = time.perf_counter()
    peak_rss = 0
    for epoch in range(start_epoch, args.epochs):
        # Epoch seed makes uninterrupted and epoch-boundary resumed runs identical.
        torch.manual_seed(872 + epoch)
        generator = torch.Generator().manual_seed(872 + epoch)
        loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                            generator=generator, num_workers=0)
        model.train()
        start = time.perf_counter()
        loss_sum, total, correct, backward_steps = 0., 0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            # BatchNorm remains numerically valid if a final singleton occurs.
            if len(images) == 1:
                for m in model.modules():
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        m.eval()
            scores = model(images)
            loss = torch.nn.functional.cross_entropy(scores, labels)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            backward_steps += 1
            loss_sum += loss.item() * len(labels)
            correct += (scores.argmax(1) == labels).sum().item()
            total += len(labels)
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        train_seconds = time.perf_counter() - start
        model.eval()
        val_loss, val_correct, val_count = 0., 0, 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                scores = model(images)
                val_loss += torch.nn.functional.cross_entropy(scores, labels).item() * len(labels)
                val_correct += (scores.argmax(1) == labels).sum().item()
                val_count += len(labels)
        record = dict(epoch=epoch + 1, train_loss=loss_sum / total, train_accuracy=correct / total,
                      validation_loss=val_loss / val_count, validation_accuracy=val_correct / val_count,
                      train_images=total, validation_images=val_count, backward_steps=backward_steps,
                      train_seconds=train_seconds, images_per_second=total / train_seconds,
                      peak_process_rss_bytes=peak_rss, resumed_from_epoch=start_epoch)
        history.append(record)
        ck = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch + 1,
                  identity_map=mapping, manifest=args.manifest, manifest_sha256=manifest_sha,
                  initialization=args.initialization, history=history, batch_size=args.batch_size,
                  training_status='smoke_face_training_only' if args.run.startswith('smoke') else 'subset_face_training',
                  torch_rng_state=torch.get_rng_state(), config=vars(args))
        temp = output / 'last.pt.tmp'
        torch.save(ck, temp)
        temp.replace(output / 'last.pt')
        write_json(f'reports/{args.run}_training.json', dict(history=history, config=vars(args),
                   identity_map=mapping, wall_seconds_this_invocation=time.perf_counter() - overall,
                   status=ck['training_status'], full_face_training_completed=False))
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='data/manifests/smoke_sources.csv')
    p.add_argument('--run', default='smoke_resnet')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--resume')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='cpu')
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--initialization', choices=['imagenet', 'random'], default='imagenet')
    train(p.parse_args())
