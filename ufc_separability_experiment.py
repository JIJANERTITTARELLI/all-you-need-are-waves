"""
UFC Framework - Minimal Separability Experiment
================================================
Companion to: "All You Need Are Waves: Frequency as Shared Substrate
for Multi-Paradigm AI Integration" (Janer, 2026)

Tests the core claim: paradigm representations encoded in distinct
frequency bands remain separable after superposition — under ideal
conditions, under noise, and at increasing scale.

Four experiments:
  1. Baseline         — lossless round-trip (establishes floor)
  2. Noise resilience — Gaussian noise added to combined signal
  3. Scale stress     — more items, fixed signal length
  4. Band compression — reduced spectral resolution

Note on implementation: We work with a complex-valued frequency-domain
signal throughout. This avoids the Hermitian symmetry constraint of
real signals (which would halve usable bins) and is consistent with
the framework's use of full frequency space.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Core encoding / decoding
# ─────────────────────────────────────────────

def encode(vectors, start_bin, signal_length):
    """
    Encode vectors into a complex frequency-domain array.
    Each vector occupies D consecutive bins starting at start_bin.
    Vectors are encoded as real parts; imaginary parts left zero.
    """
    n, d = vectors.shape
    freq = np.zeros(signal_length, dtype=complex)
    end_bin = start_bin + n * d
    if end_bin > signal_length:
        raise OverflowError(
            f"Band overflow: need bin {end_bin}, signal has {signal_length} bins"
        )
    for i, vec in enumerate(vectors):
        s = start_bin + i * d
        freq[s:s + d] = vec
    return freq


def decode(freq, start_bin, n_items, dim):
    """Reconstruct vectors from their assigned frequency band."""
    out = np.zeros((n_items, dim))
    for i in range(n_items):
        s = start_bin + i * dim
        out[i] = np.real(freq[s:s + dim])
    return out


def cosine_sims(original, recovered):
    """Per-item cosine similarity."""
    sims = []
    for o, r in zip(original, recovered):
        no, nr = np.linalg.norm(o), np.linalg.norm(r)
        sims.append(np.dot(o, r) / (no * nr) if no > 1e-10 and nr > 1e-10 else 0.0)
    return np.array(sims)


def run(n_sym, n_tok, d_sym, d_tok, sig_len, noise_std=0.0, label=""):
    """
    Encode both paradigms → superpose → add noise → decode → measure.
    Returns result dict.
    """
    np.random.seed(42)

    sym = np.random.randn(n_sym, d_sym)
    sym /= np.linalg.norm(sym, axis=1, keepdims=True)

    tok = np.random.randn(n_tok, d_tok)
    tok /= np.linalg.norm(tok, axis=1, keepdims=True)

    GAP = 50
    sym_start = 1
    tok_start = sym_start + n_sym * d_sym + GAP

    try:
        f_sym = encode(sym, sym_start, sig_len)
        f_tok = encode(tok, tok_start, sig_len)
    except OverflowError as e:
        return {'label': label, 'status': str(e),
                'sym_mean': None, 'tok_mean': None}

    # Superpose
    f_combined = f_sym + f_tok

    # Add noise in frequency domain (equivalent to signal noise)
    if noise_std > 0:
        f_combined += noise_std * (
            np.random.randn(sig_len) + 1j * np.random.randn(sig_len)
        )

    # Decode each paradigm from its band
    sym_rec = decode(f_combined, sym_start, n_sym, d_sym)
    tok_rec = decode(f_combined, tok_start, n_tok, d_tok)

    # Fidelity
    ss = cosine_sims(sym, sym_rec)
    ts = cosine_sims(tok, tok_rec)

    # Cross-band leakage: energy in gap between bands
    gap_s = sym_start + n_sym * d_sym
    gap_e = tok_start
    sym_energy = np.sum(np.abs(f_combined[sym_start:gap_s])**2)
    tok_energy = np.sum(np.abs(f_combined[tok_start:tok_start + n_tok * d_tok])**2)
    gap_energy = np.sum(np.abs(f_combined[gap_s:gap_e])**2)
    leakage = gap_energy / (sym_energy + tok_energy + 1e-10)

    return {
        'label':    label,
        'status':   'OK',
        'sym_mean': np.mean(ss), 'sym_std': np.std(ss), 'sym_min': np.min(ss),
        'tok_mean': np.mean(ts), 'tok_std': np.std(ts), 'tok_min': np.min(ts),
        'leakage':  leakage,
        'bins_used': tok_start + n_tok * d_tok,
        'bins_total': sig_len,
    }


def show(r):
    if r['status'] != 'OK':
        print(f"  [{r['label']}]  ✗  {r['status']}")
        return
    sym_ok = "✓" if r['sym_mean'] > 0.99 else ("~" if r['sym_mean'] > 0.90 else "✗")
    tok_ok = "✓" if r['tok_mean'] > 0.99 else ("~" if r['tok_mean'] > 0.90 else "✗")
    print(f"  [{r['label']}]")
    print(f"    Symbolic  {sym_ok}  mean={r['sym_mean']:.6f}  std={r['sym_std']:.2e}  min={r['sym_min']:.6f}")
    print(f"    Token     {tok_ok}  mean={r['tok_mean']:.6f}  std={r['tok_std']:.2e}  min={r['tok_min']:.6f}")
    print(f"    Leakage: {r['leakage']:.2e}   Bins: {r['bins_used']}/{r['bins_total']}")


# ─────────────────────────────────────────────
# Experiment suite
# ─────────────────────────────────────────────

print("=" * 64)
print("UFC SEPARABILITY EXPERIMENT  v0.1")
print("All You Need Are Waves — Janer, 2026")
print("=" * 64)

N_SYM, N_TOK = 100, 500
D_SYM, D_TOK = 20, 50
SIG = 2**15     # 32768 bins — all usable (complex signal)

print(f"\nBaseline config: {N_SYM} symbolic predicates (d={D_SYM}) + "
      f"{N_TOK} token embeddings (d={D_TOK})")
print(f"Signal length: {SIG} bins\n")

print("── 1. BASELINE (no noise) ──")
show(run(N_SYM, N_TOK, D_SYM, D_TOK, SIG, label="Baseline"))

print("\n── 2. NOISE RESILIENCE ──")
for σ in [0.001, 0.01, 0.1, 0.5, 1.0]:
    show(run(N_SYM, N_TOK, D_SYM, D_TOK, SIG, noise_std=σ, label=f"σ={σ}"))

print("\n── 3. SCALE STRESS (fixed signal length) ──")
for scale in [1, 2, 5, 10, 20]:
    show(run(N_SYM*scale, N_TOK*scale, D_SYM, D_TOK, SIG,
             label=f"{N_SYM*scale} sym + {N_TOK*scale} tok"))

print("\n── 4. SPECTRAL RESOLUTION (shrinking signal) ──")
for length in [2**15, 2**14, 2**13, 2**12]:
    show(run(N_SYM, N_TOK, D_SYM, D_TOK, length, label=f"len={length}"))

print("\n── 5. COMBINED STRESS (5x scale + noise) ──")
show(run(N_SYM*5, N_TOK*5, D_SYM, D_TOK, SIG,
         noise_std=0.1, label="5x + σ=0.1"))

print("\n" + "=" * 64)
print("INTERPRETATION")
print("=" * 64)
print("""
  ✓  cosine sim > 0.99  →  separability holds
  ~  cosine sim 0.90-0.99  →  degraded but recoverable
  ✗  cosine sim < 0.90  →  separability fails

  Leakage near 0 = clean band separation
  Leakage rising  = cross-paradigm interference

  Per paper Section 8:
  SUPPORTED if mean sim > threshold AND leakage bounded/manageable
  UNDERMINED if sim degrades below semantic-preservation threshold
              OR interference is systematic and non-recoverable
""")
