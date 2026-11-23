# LM_Dec_Long_HybridMix_LocalGlobal_W64

HybridMix_LocalGlobal_W64 - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Reviewed**
- Failure class: Experiment Incomplete: not run yet

## Provenance

- Source repository: (new) hybrid: GRU + local/global attention stack
- Upstream / license: lucidrains/x-transformers (MIT), pip 2.31.7 + torch.nn

## Differences from the source ([Unified] protocol)

- hyb_mix with window 64: GRU, 3 window-64 RoPE layers, 3 global NoPE layers

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- No results yet.

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check hyb_mix64
python run.py lm02 hyb_mix64      # enwik8 is downloaded to data/ on first use
```
