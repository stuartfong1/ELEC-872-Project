"""Fixed proposal architectures. No generic CLIP or substitute face model weights."""
import importlib.util
import os
import sys
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from .common import ROOT, sha256

os.environ.setdefault('TORCH_HOME', str(ROOT / 'weights/torch_cache'))
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def freeze(model):
    model.eval()
    model.requires_grad_(False)
    return model


def load_farl():
    from .clip_model import VisionTransformer
    path = 'weights/FaRL-Base-Patch16-LAIONFace20M-ep16.pth'
    checkpoint = torch.load(ROOT / path, map_location='cpu', weights_only=True, mmap=True)
    state = checkpoint['state_dict']
    visual = {k.removeprefix('visual.'): v for k, v in state.items() if k.startswith('visual.')}
    # Construct only official CLIP visual architecture; every inference parameter
    # must be supplied by FaRL, with strict state matching (no random fallbacks).
    with torch.device('meta'):
        model = VisionTransformer(input_resolution=224, patch_size=16, width=768,
                                  layers=12, heads=12, output_dim=512)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(visual))
    extra = sorted(set(visual) - set(expected))
    if missing:
        raise ValueError(f'FaRL visual tensors missing: {missing}')
    model.load_state_dict({k: visual[k] for k in expected}, strict=True, assign=True)
    assert torch.equal(model.conv1.weight, visual['conv1.weight'].float())
    preprocessing = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                               T.CenterCrop(224), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])
    meta = dict(model='FaRL', architecture='ViT-B/16', checkpoint=path, checkpoint_sha256=sha256(path),
                feature_layer='visual CLS after ln_post and projection', input_size=224, color='RGB',
                normalization={'mean': CLIP_MEAN, 'std': CLIP_STD}, loaded_visual_tensors=len(expected),
                missing_visual_tensors=missing, ignored_visual_training_tensors=extra,
                role='general facial representation; not a dedicated identity recognizer',
                frozen=True, evaluation_mode=True, generic_clip_weights_loaded=False)
    return freeze(model.float()), preprocessing, meta


def lvface_preprocess(image):
    import cv2
    # PIL already supplies RGB; equivalent to official cv2 BGR->RGB path.
    a = cv2.resize(np.asarray(image.convert('RGB')), (112, 112), interpolation=cv2.INTER_LINEAR)
    a = ((a.transpose(2, 0, 1).astype(np.float32) / 255.) - .5) / .5
    return torch.from_numpy(a.copy())


def load_lvface(backend='torch'):
    pt = 'weights/LVFace-T_Glint360K.pt'
    onnx = 'weights/LVFace-T_Glint360K.onnx'
    meta = dict(model='LVFace', architecture='vit_t: 112/patch9/embed256/depth12/heads8/output512',
                checkpoint=pt if backend == 'torch' else onnx,
                checkpoint_sha256=sha256(pt if backend == 'torch' else onnx),
                feature_layer='final embedding', input_size=112, color='RGB', normalization='(x/255-0.5)/0.5',
                resize='OpenCV INTER_LINEAR', frozen=True, evaluation_mode=True,
                role='face recognition trained on Glint360K', backend=backend)
    if backend == 'onnx':
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        session = ort.InferenceSession(str(ROOT / onnx), sess_options=opts, providers=['CPUExecutionProvider'])
        meta['onnx_input_shape'] = session.get_inputs()[0].shape
        return session, lvface_preprocess, meta
    sys.path.insert(0, str(ROOT / 'third_party/LVFace'))
    from backbones import get_model
    model = get_model('vit_t', num_features=512, fp16=False)
    state = torch.load(ROOT / pt, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    meta['strict_state_match'] = True
    return freeze(model.float()), lvface_preprocess, meta


def resnet_transform():
    return T.Compose([T.Resize(232, interpolation=T.InterpolationMode.BILINEAR),
                      T.CenterCrop(224), T.ToTensor(),
                      T.Normalize((.485, .456, .406), (.229, .224, .225))])


def new_resnet(classes, initialization='imagenet'):
    weights = ResNet50_Weights.IMAGENET1K_V2 if initialization == 'imagenet' else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, classes)
    return model


def load_resnet(checkpoint):
    ck = torch.load(ROOT / checkpoint, map_location='cpu', weights_only=True)
    model = new_resnet(len(ck['identity_map']), initialization='random')
    model.load_state_dict(ck['model'], strict=True)
    model.fc = nn.Identity()
    meta = dict(model='ResNet50', architecture='torchvision ResNet50 V1.5', checkpoint=checkpoint,
                checkpoint_sha256=sha256(checkpoint), feature_layer='flattened avgpool before identity fc',
                input_size=224, color='RGB', normalization='ImageNet mean/std', resize='232 then center224',
                training_status=ck['training_status'], epochs=ck['epoch'], identities=len(ck['identity_map']),
                initialization=ck['initialization'], frozen=True, evaluation_mode=True)
    return freeze(model), resnet_transform(), meta
