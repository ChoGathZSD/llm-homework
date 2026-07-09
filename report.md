# Part B Report: LoRA SVG Logo Fine-tuning

## Summary

I fine-tuned Gemma 3 270M with LoRA on the 219 published detailed-prompt -> SVG pairs and evaluated on the 17-example validation set. The main improvement is format direction: the base model usually continues in natural language, while the fine-tuned adapter begins outputs with SVG-like markup. However, most generated SVGs are still incomplete under the fast self-evaluation decoding limit, so the absolute reward remains low.

| Model | Mean reward | Median | Min | Max |
|---|---:|---:|---:|---:|
| Base Gemma 3 270M | 1.5844 | 1.5847 | 1.0476 | 1.9125 |
| LoRA adapter | 2.0480 | 2.1125 | 1.7875 | 2.1125 |

Mean delta: +0.4636 reward points, or +29.26% relative to the base model.

## Reward Design

The reward in `reward.py` is a weighted proxy for whether a generated logo is a usable SVG:

- Validity: XML parseability, a single `<svg>...</svg>` envelope, no markdown or extra prose.
- Structure: expected SVG tags, `viewBox`, reasonable number and diversity of vector elements.
- Geometry: coordinates mostly inside or near the 256 x 256 canvas, no collapsed or extreme geometry.
- Palette: at least two visible colors, moderate palette size, contrast and saturation.
- Prompt alignment: heuristic matches between prompt color/shape words and generated SVG colors/tags.
- Anti-degeneration: penalizes empty, tiny, overlong, repeated, or externally referenced outputs.

The reward deliberately gives validity the largest weight because an invalid SVG cannot be visually judged reliably. Prompt alignment is included but kept lower because keyword heuristics are much weaker than actual visual assessment.

## Training Setup

I used the `transformers + PEFT` route rather than ms-swift. The final run used:

- Base model: Gemma 3 270M, locally cached under `models/gemma-3-270m`.
- LoRA rank: 8, alpha: 16, dropout: 0.05.
- Target modules: q/k/v/o projections and MLP up/down/gate projections.
- Batch size: 1.
- Gradient accumulation: 1.
- Epochs: 2.
- Max sequence length: 2048.
- Precision: bfloat16.
- Optimizer: AdamW.
- Learning rate: 2e-4 with cosine schedule.

An initial fp16 run produced NaN loss, so I switched to bfloat16 after a one-step smoke test showed finite loss and gradients. The final training run was numerically healthy: validation loss decreased from 0.7248 after epoch 1 to 0.6928 after epoch 2.

## Results and Interpretation

The base model mostly produced natural-language continuations instead of SVG. For example, one base output began with descriptive prose about a pencil and then repeated phrases. This explains its near-zero validity and structure scores.

The LoRA model learned the broad output format. A typical adapter output begins:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><rect ...
```

This is a real improvement over the base model, but the generated SVGs were usually truncated before `</svg>` during self-evaluation, causing XML parse errors such as "unclosed token". Therefore the adapter receives only a small reward increase even though it has learned the initial SVG format.

This is a useful Goodhart-style observation: the model improved on a proxy-relevant behavior, namely starting valid SVG markup, but did not yet produce complete valid SVG documents under the chosen decoding budget. The reward correctly penalizes this because incomplete SVGs are not usable logos.

## Limitations

The result is not visually strong. The adapter does not reliably close SVG tags, and most validation outputs remain invalid XML. The absolute score is low, so I would not claim the model has learned robust SVG drawing. The most defensible claim is narrower: LoRA moved the model from natural-language continuation toward SVG-format continuation.

The self-evaluation decoding was capped with a per-example time limit to keep evaluation reproducible on an 8 GB local GPU. This likely suppresses the score of outputs that might eventually close after more tokens, but it also reflects a practical constraint: a useful small model should produce compact SVGs quickly.

## Next Experiments

The most promising next step would be to train with shorter target SVGs or add a reward/data preference for compact, closed SVGs. A small curated subset of simple logos may help the 270M model learn complete documents before trying more complex examples. I would also test generation with a stricter stop condition and a shorter target style, rather than simply increasing max tokens.
