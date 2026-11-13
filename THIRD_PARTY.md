# Third-party material

## QSIV

[QSIV](https://huggingface.co/djinn-san/QSIV), copyright 2026 the QSIV authors, is the local decision runtime and
model this package drives. Its code and weights are not bundled in this package. `qsiv-runtime setup` fetches
revision `f1ee842daf758c0ab535cf7d281c85439d6e4b25`; the downloaded model directory retains its complete
Apache-2.0
[LICENSE](https://huggingface.co/djinn-san/QSIV/blob/f1ee842daf758c0ab535cf7d281c85439d6e4b25/LICENSE)
and [NOTICE](https://huggingface.co/djinn-san/QSIV/blob/f1ee842daf758c0ab535cf7d281c85439d6e4b25/NOTICE).

QSIV's NOTICE credits the Qwen team / Alibaba Cloud for Qwen3.5-0.8B-Base and its tokenizer, as well as the Gated
DeltaNet and flash-linear-attention work. Those notices and dependency licenses remain applicable. Setup changes
only the downloaded runtime's NumPy dependency pin to match the tested environment; it does not change QSIV
inference code or weights.

## agemem

The service, setup and doctor code is adapted from [agemem](https://github.com/illescasDaniel/agemem) (MIT, same
author). The pinned GPU environment lock file `src/qsiv_runtime/data/qsiv-linux-py312.lock` is a trimmed copy of
agemem's.

## Other material

Dependencies installed into the separate GPU environment (PyTorch, Transformers, ...) retain their own licenses;
the notices above do not relicense them.
