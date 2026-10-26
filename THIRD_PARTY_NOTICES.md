# Source provenance

The exact fetched commits and download URLs/hashes are recorded in
`reports/acquisition.json`. No generic CLIP pretrained weights are downloaded.

- `phase1/upstream_processors.py`: unmodified `feature_extraction/processors.py`
  from cracker321/visual-encoder-traces, commit
  `7c4859213fbde95a410307fa62cb0e2aeef12593`; same commit as the original
  ryan-caesar-ramos/visual-encoder-traces repository. The supplied user snippet
  matches this API. Wrapper code is in `phase1/processing.py`.
- `phase1/clip_model.py`: OpenAI CLIP architecture source, commit
  `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`; MIT license in
  `licenses/LICENSE_CLIP.txt`. Only its ViT-B/16 architecture is used with official
  FaRL weights, not generic CLIP weights.
- `third_party/LVFace`: official Bytedance repository, commit
  `80bd20c8bb78e56eb708941180d800a27c396a6b`; license in
  `licenses/LICENSE_LVFace.txt`. `vit_t` must be specified; its example's `r50`
  default is not used. Downloaded by `scripts/acquire.py`, excluded from Git.
- FaRL checkpoint: official ep16, SHA-256
  `67b4d6bcad0cd4bdaa302741e05e92eb82ecb334ef92446555b3abd3e3cd3862`.
- LVFace checkpoints: official Hugging Face revision
  `b12702ab1f5c721748e054a66dc90e1edd1f0724` (T / Glint360K).
- ResNet50 implementation: torchvision, version recorded by the actual
  environment. Initialization uses the official IMAGENET1K_V2 enum; its weight
  file remains separate from locally face-trained checkpoints.

Official references:
[FaRL](https://github.com/FacePerceiver/FaRL),
[LVFace](https://github.com/bytedance/LVFace),
[LVFace weights](https://huggingface.co/bytedance-research/LVFace),
[torchvision ResNet50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html),
[original processors](https://github.com/ryan-caesar-ramos/visual-encoder-traces),
[team fork processors](https://github.com/cracker321/visual-encoder-traces/blob/main/feature_extraction/processors.py),
[agreed Kaggle upload](https://www.kaggle.com/datasets/hearfool/vggface2/data).
