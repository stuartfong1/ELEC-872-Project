import json
import numpy as np
from PIL import Image
from phase1.processing import CORE, EXTRA, apply


def test_identity_and_constant_jpeg():
    reference = Image.new('RGB', (64, 64), (128, 128, 128))
    before = np.asarray(reference).copy()
    for condition in CORE:
        result, meta = apply(reference, condition, 'constant')
        assert result.mode == 'RGB' and result.size == reference.size
        assert np.array_equal(np.asarray(reference), before)
        if condition == 'IDENTITY':
            assert np.array_equal(np.asarray(result), before)
        else:
            assert meta['jpeg_sampling_code'] == (2 if condition.endswith('420') else 0)
    # No assertion that distinct JPEG settings must yield distinct pixels.


def test_independent_reproducible_conditions():
    ref = Image.fromarray(np.random.default_rng(872).integers(0, 256, (73, 91, 3), dtype=np.uint8))
    original = np.asarray(ref).copy()
    for condition in CORE + EXTRA:
        first, meta1 = apply(ref, condition, 'source')
        _ = apply(ref, 'MASK_95', 'other_source')
        second, meta2 = apply(ref, condition, 'source')
        assert np.array_equal(np.asarray(first), np.asarray(second))
        assert meta1 == meta2
        assert np.array_equal(np.asarray(ref), original)


def test_jpeg_matches_actual_upstream_function():
    from phase1.upstream_processors import Processor
    ref = Image.fromarray(np.random.default_rng(872).integers(0, 256, (100, 120, 3), dtype=np.uint8))
    for name in CORE:
        expected = Processor[name].get_processing()(ref.copy()).convert('RGB')
        actual, _ = apply(ref, name, 'test')
        assert np.array_equal(np.asarray(actual), np.asarray(expected))


def test_resize_order_and_dimensions():
    ref = Image.new('RGB', (100, 80), 'blue')
    half, metadata = apply(ref, 'RESIZE_HALF', 'test')
    assert half.size == (50, 40)
    assert metadata['operations'][0].startswith('Rescale')
    assert metadata['operations'][1] == 'JPEGCompression(quality=95, subsampling=4:4:4)'


def test_corrupt_image_does_not_decode(tmp_path):
    import pytest
    p = tmp_path / 'bad.jpg'
    p.write_bytes(b'not an image')
    with pytest.raises(Exception):
        Image.open(p).load()


def test_generated_png_cache_is_immutable(tmp_path):
    import pytest
    from phase1.data import save_immutable_png
    path = tmp_path / 'reference.png'
    image = Image.new('RGB', (16, 16), 'red')
    save_immutable_png(image, path)
    original_bytes = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns
    save_immutable_png(image, path)
    assert path.stat().st_mtime_ns == original_mtime
    with pytest.raises(ValueError, match='Existing generated PNG differs'):
        save_immutable_png(Image.new('RGB', (16, 16), 'blue'), path)
    assert path.read_bytes() == original_bytes
