# All You Need Are Waves
### Frequency as Shared Substrate for Multi-Paradigm AI Integration

[![DOI](https://doi.org/10.5281/zenodo.20287329)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

---

## The Idea

Current AI systems operate across incompatible paradigms — LLMs, symbolic reasoners, world models, perceptual encoders — each with representational logics that resist direct integration. Existing approaches (adapter layers, multimodal latent spaces, orchestration frameworks) either don't scale or require retraining when new paradigms are added.

This project proposes a different integration substrate: **frequency space**.

Rather than translating between paradigms, each paradigm independently encodes its representations as frequency signatures into a shared spectral space. Integration happens through a spectral architecture operating on the superposed signal. The core structural claim: frequency space has **fixed orthogonality properties independent of any training objective** — unlike trained latent spaces, whose geometry is paradigm-dependent by construction.

The title is a deliberate reference to Vaswani et al. (2017). Where *Attention Is All You Need* proposed attention as the sufficient mechanism for language modeling, this paper proposes that the next integration problem may require a different substrate — one grounded in the physics of waves.

---

## This Repository

| File | What it is |
|------|------------|
| `experiment/ufc_separability_experiment.py` | Core separability test: noise, scale, resolution |
| `experiment/ufc_multirun_validation.py` | 10-seed validation — results not initialization artifacts |
| `experiment/ufc_wavetype_experiment.py` | Wave type comparison + wavelength-reliability relationship |
| `paper/` | Links to full paper and working notes on Zenodo |

---

## Run the Experiments

**Requirements:** Python 3.8+, NumPy, SciPy, PyWavelets

```bash
pip install numpy scipy pywavelets
python experiment/ufc_separability_experiment.py
python experiment/ufc_multirun_validation.py
python experiment/ufc_wavetype_experiment.py
```

No GPU required. All three run in under 30 seconds.

---

## Results Summary

### Separability

```
Baseline (no noise):  sim = 1.000000 ± 0.00  across ALL seeds
Noise threshold:
  σ ≤ 0.01  →  holds   (>0.997)
  σ = 0.05  →  degraded (~0.96)
  σ ≥ 0.10  →  fails
```

### Wave types (FFT / DCT / Haar)

```
All three orthogonal bases perform equivalently at σ=0.05:
  FFT: Sym 0.977  Tok 0.944
  DCT: Sym 0.977  Tok 0.944
  Haar: Sym 0.977  Tok 0.944

→ Orthogonality is the relevant property, not waveform shape.
```

### Wavelength vs. reliability

```
Low band vs. high band (same paradigm, varying noise):
  σ=0.10  →  low band advantage: +0.003
  σ=0.20  →  low band advantage: +0.012
  σ=0.50  →  low band advantage: +0.029

→ Lower frequencies are more noise-robust.
  Band allocation has a direction: symbolic/relational paradigms
  in low bands, dense embeddings in high bands.
```

**What this means:** Separability holds under ideal conditions and low noise, robust across initializations. Wave type is not significant. Lower frequency bands are measurably more noise-robust — turning band allocation from a free parameter into a design principle.

---

## What This Is Not

This repository does not demonstrate:
- Performance improvements over existing multimodal approaches
- That the architecture is trainable at scale
- Computational efficiency on current hardware
- That biological analogies constitute mechanistic evidence

These are open empirical questions. See Section 9.5 of the paper.

---

## Next Steps

1. **Relational binding test** — HRR-style convolution in frequency space; recovery after superposition
2. **Noise-robust allocation** — learned vs. fixed band allocation
3. **Dimensionality scaling** — production-scale embeddings (768, 4096 dims)
4. **Hardware sketch** — FFT on photonic/neuromorphic simulators vs. GPU

Pull requests, implementations, and demolitions welcome.

---

## Paper and Working Notes

> Janer, J. J. (2026). *All You Need Are Waves: Frequency as Shared Substrate for Multi-Paradigm AI Integration*. Working Paper v0.4. https://doi.org/10.5281/zenodo.XXXXXXX

---

## Citation

```bibtex
@misc{janer2026waves,
  author    = {Janer Tittarelli, Javier Ignacio},
  title     = {All You Need Are Waves: Frequency as Shared Substrate
               for Multi-Paradigm AI Integration},
  year      = {2026},
  month     = {May},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX},
  note      = {Working Paper v0.4}
}
```

---

## License

Paper and working notes: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
Code: [MIT](LICENSE)

---

*A pre-dawn conjecture, submitted to the world for demolition or development.*
*Buenos Aires, May 2026.*
