"""Use the supplied upstream Processor functions; preserve their exact operations.

Every call receives a fresh RGB reference copy. RNG is recreated per source and
condition, making interpolation independent of iteration order and resumable.
Only ownership/lifetime and reproducible RNG handling differ from upstream.
"""
import numpy as np
from PIL import Image
from .upstream_processors import Processor
from .common import digest

CORE = ['IDENTITY'] + [f'JPEG{q}_{s}' for q in (75, 85, 95) for s in ('420', '444')]
EXTRA = ['SHARPEN_2', 'SHARPEN_4', 'RESIZE_HALF', 'RESIZE_DOUBLE',
         'INTERPOLATION_BILINEAR', 'INTERPOLATION_BICUBIC',
         'INTERPOLATION_LANCZOS', 'INTERPOLATION_BOX', 'MASK_95']


def apply(reference, name, source_id, seed=872):
    process = Processor[name].get_processing()
    rng_seed = int(digest(f'{seed}/{source_id}/{name}')[:8], 16)
    parts = getattr(process, 'transforms', [process])
    operations = []
    for part in parts:
        if hasattr(part, 'rng'):
            part.rng = np.random.default_rng(rng_seed)
        operations.append(repr(part))
    # Calling the actual provided function, rather than recreating JPEG behavior.
    result = process(reference.copy())
    result.load()
    # Re-encoding is forbidden: PNG stores already-decoded JPEG pixels losslessly.
    if getattr(result, 'format', None) == 'JPEG':
        from PIL.JpegImagePlugin import get_sampling
        sampling = get_sampling(result)
        qtables = result.quantization
    else:
        sampling, qtables = None, None
    return result.convert('RGB').copy(), {
        'operations': operations, 'rng_seed': rng_seed,
        'jpeg_sampling_code': sampling, 'quantization_tables': qtables,
        'output_size': list(result.size), 'storage': 'PNG of decoded result; no extra JPEG',
    }
