# OFDM Ambiguity Function Analysis for ISAC

This baseline implements ambiguity function analysis for OFDM-based waveforms in Integrated Sensing and Communications (ISAC) systems, comparing with traditional Linear Frequency Modulated (LFM) radar waveforms.

## 📖 Mathematical Background

### Ambiguity Function

The **ambiguity function** (AF) is the fundamental tool for analyzing radar waveform resolution properties. It measures the matched filter output as a function of delay τ and Doppler frequency ν:

```
χ(τ, ν) = ∫ s(t) s*(t-τ) e^{j2πνt} dt
```

where:
- `s(t)` is the transmitted signal
- `τ` is the time delay (related to target range: R = cτ/2)
- `ν` is the Doppler frequency (related to target velocity: v = λν/2)

### Properties

1. **Peak at origin**: `|χ(0, 0)| = E` (signal energy)
2. **Symmetry**: `|χ(τ, ν)| = |χ(-τ, -ν)|`
3. **Volume invariance**: `∫∫ |χ(τ, ν)|² dτ dν = E²`

### OFDM Signal Model

An OFDM signal with N subcarriers is:

```
s(t) = (1/√N) Σ[k=0 to N-1] X[k] e^{j2πkΔf t}
```

where:
- `X[k]` are QAM-modulated symbols (unit average power)
- `Δf = 1/T_s` is the subcarrier spacing
- `T_s` is the useful symbol duration

With cyclic prefix (CP):
```
s_cp(t) = s(t + T_CP)  for 0 ≤ t < T_CP    (CP copy)
          s(t)          for T_CP ≤ t < T_s + T_CP  (data)
```

### OFDM Ambiguity Function

For OFDM with random QAM symbols, the ambiguity function envelope is approximately:

```
|χ(τ, ν)|² ≈ |sin(πNνT_s) / sin(πνT_s)|² × sinc²(τΔf)
```

Key characteristics:
- **Doppler cut**: Dirichlet kernel (periodic sinc) → sidelobes at ±1/T_s
- **Delay cut**: sinc² envelope → 3dB width ≈ 0.886/B
- **Grid sidelobes**: Peaks at (τ = nT_s, ν = m/T) due to symbol periodicity

### LFM (Chirp) Comparison

Linear Frequency Modulated signal:
```
s(t) = exp(jπKt²)  for 0 ≤ t ≤ T
```
where K = B/T is the chirp rate.

LFM AF properties:
- **"Thumbtack" shape** near origin
- **First sidelobe**: -13.2 dB (sinc envelope)
- **Range-Doppler coupling**: Diagonal ridge in AF

## 📊 Resolution Analysis

### Range Resolution

```
ΔR = c / (2B)
```

| Bandwidth | Resolution |
|-----------|------------|
| 20 MHz    | 7.5 m      |
| 100 MHz   | 1.5 m      |
| 400 MHz   | 0.375 m    |

### Doppler Resolution

```
Δν = 1 / T_c
```

where T_c is the coherent processing interval.

| Coherent Time | Resolution |
|---------------|------------|
| 1 ms          | 1000 Hz    |
| 10 ms         | 100 Hz     |
| 100 ms        | 10 Hz      |

### PAPR Considerations

OFDM suffers from high Peak-to-Average Power Ratio (PAPR):

```
PAPR = max|s(t)|² / E[|s(t)|²]
```

- **OFDM (64 subcarriers)**: ~10-12 dB
- **LFM**: 0 dB (constant envelope)

This impacts power amplifier efficiency and dynamic range requirements.

## 🛠️ Implementation

### Module Structure

```
ofdm_ambiguity_function/
├── ofdm_ambiguity.py      # Core implementation
├── test_ofdm_ambiguity.py # Test suite
├── generate_figures.py    # Figure generation
└── README.md              # This file
```

### Core Functions

| Function | Description |
|----------|-------------|
| `generate_ofdm_signal()` | Generate OFDM waveform with QAM modulation |
| `compute_ambiguity_function()` | Compute 2D AF \|χ(τ, ν)\|² |
| `compute_ambiguity_function_ofdm()` | Optimized OFDM AF computation |
| `generate_lfm_signal()` | Generate LFM (chirp) signal |
| `plot_ambiguity_3d()` | 3D surface plot |
| `plot_ambiguity_contour()` | Contour plot |
| `compute_range_resolution()` | Theoretical range resolution |
| `compute_doppler_resolution()` | Theoretical Doppler resolution |

### Usage

```python
from ofdm_ambiguity import *

# Generate OFDM signal
ofdm_signal = generate_ofdm_signal(n_subcarriers=64, cp_len=16)

# Compute ambiguity function
af, tau_range, nu_range = compute_ambiguity_function_ofdm(n_subcarriers=64)

# Generate LFM for comparison
lfm_signal = generate_lfm_signal(bandwidth=20e6, pulse_width=10e-6)

# Plot results
plot_ambiguity_contour(af, tau_range, nu_range, title="OFDM AF")
```

## 🧪 Testing

Run the test suite:

```bash
cd code/baselines/ofdm_ambiguity_function
python -m pytest test_ofdm_ambiguity.py -v
```

Test coverage includes:
- Ambiguity function fundamental properties
- OFDM signal generation and CP handling
- LFM signal comparison
- Resolution analysis
- PAPR computation
- Edge cases

## 📈 Example Figures

Generate all figures:

```bash
python generate_figures.py --output figures
```

This produces:
1. **OFDM Ambiguity Function 3D**: Surface visualization of |χ(τ, ν)|²
2. **OFDM Ambiguity Contour**: Contour plot with dB levels
3. **LFM Ambiguity Contour**: Comparison with chirp waveform
4. **Resolution Comparison**: Range/Doppler resolution vs parameters

## 🔬 ISAC Design Implications

### Waveform Choice

| Aspect | OFDM | LFM |
|--------|------|-----|
| Communication | Native (QAM on subcarriers) | Needs modulation overlay |
| Sensing | Higher sidelobes | Better sidelobes |
| PAPR | High (10+ dB) | Low (0 dB) |
| Doppler handling | Periodic ambiguity | Continuous |
| MIMO compatibility | Excellent (flexible) | Good |

### Key Trade-offs

1. **Bandwidth allocation**: Communication vs sensing
2. **CP length**: ISI protection vs resolution loss
3. **Subcarrier spacing**: Doppler tolerance vs data rate
4. **Modulation order**: Data rate vs peak power

### Recommendations

- **High-mobility**: Use shorter OFDM symbols (wider subcarrier spacing)
- **High-resolution**: Use larger bandwidth, consider LFM overlay
- **Joint processing**: Exploit AF structure for interference mitigation

## 📚 References

1. Richards, M. A. "Fundamentals of Radar Signal Processing" (2005)
2. Levanon, N. & Mozeson, E. "Radar Signals" (2004)
3. Liu, F. et al. "Integrated Sensing and Communications: Towards Dual-functional Wireless Networks for 6G and Beyond" (2022)
4. Sturm, C. & Wiesbeck, W. "Waveform Design and Signal Processing Aspects for Fusion of Wireless Communications and Radar Sensing" (2011)

## 📝 License

MIT License - See repository LICENSE file.
