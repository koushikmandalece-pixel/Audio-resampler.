import io
from fractions import Fraction
import numpy as np
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt
from scipy import signal

st.set_page_config(page_title="SP12: Multi-Rate Audio Resampler", layout="wide")
st.title("SP12: Rational Multi-Rate Audio Resampler & Distortion Analyzer")

# --- DSP Helper Functions ---
def analyze_rational_ratio(fs_in: float, fs_out: float, max_denom: int = 2000):
    """Determines reduced L/M ratio and whether the conversion is cleanly rational."""
    exact_ratio = fs_out / fs_in
    frac = Fraction(exact_ratio).limit_denominator(max_denom)
    L, M = frac.numerator, frac.denominator
    approx_fs_out = fs_in * (L / M)
    rate_error_hz = abs(fs_out - approx_fs_out)

    # Check if the ratio is exact (error < 1e-6 Hz) and has practical coprime factors
    is_exact_rational = rate_error_hz < 1e-6

    if L == 1 and M == 1 and is_exact_rational:
        regime = "Identity Ratio (L = 1, M = 1)"
        is_practical = True
        note = "Input and target sampling rates are identical; no resampling needed."
    elif is_exact_rational and M == 1:
        regime = f"Rational — Pure Integer Upsampling (L = {L}, M = 1)"
        is_practical = True
        note = f"Exact integer interpolation by L = {L} with anti-imaging LPF at π/{L}."
    elif is_exact_rational and L == 1:
        regime = f"Rational — Pure Integer Downsampling (L = 1, M = {M})"
        is_practical = True
        note = f"Exact integer decimation by M = {M} with anti-aliasing LPF at π/{M}."
    elif is_exact_rational and L <= 500 and M <= 500:
        regime = f"Rational — Fractional Resampling (L/M = {L}/{M})"
        is_practical = True
        note = f"Exact rational conversion: upsample by {L}, filter at min(π/{L}, π/{M}), downsample by {M}."
    else:
        regime = f"Non-Rational / High-Order Ratio (Nearest L/M = {L}/{M})"
        is_practical = False
        note = (
            f"The ratio {fs_out:g}/{fs_in:g} is not a small-integer rational fraction. "
            f"Approximating with L/M = {L}/{M} yields {approx_fs_out:.4f} Hz (error: {rate_error_hz:.6f} Hz)."
        )

    return {
        "L": L,
        "M": M,
        "regime": regime,
        "is_practical": is_practical,
        "note": note,
        "approx_fs_out": approx_fs_out,
        "rate_error_hz": rate_error_hz,
    }

def design_resampler_fir(L: int, M: int, numtaps: int, beta: float):
    cutoff = min(1.0 / L, 1.0 / M)  # Normalized cutoff (1.0 = Nyquist)
    taps = signal.firwin(numtaps, cutoff=cutoff, window=("kaiser", beta)) * L
    return taps, cutoff

def rational_resample_custom(x: np.ndarray, L: int, M: int, use_filter: bool, numtaps: int, beta: float):
    if L == 1 and M == 1:
        return x.copy()
    if not use_filter:
        up = np.zeros(len(x) * L, dtype=np.float64)
        up[::L] = x * L
        return up[::M]
    taps, _ = design_resampler_fir(L, M, numtaps, beta)
    return signal.upfirdn(taps, x, up=L, down=M)

def evaluate_round_trip(x: np.ndarray, L: int, M: int, use_filter: bool, numtaps: int, beta: float):
    y_out = rational_resample_custom(x, L, M, use_filter, numtaps, beta)
    x_rec = rational_resample_custom(y_out, M, L, use_filter, numtaps, beta)
    if use_filter and not (L == 1 and M == 1):
        delay = (numtaps - 1) // 2
        shift = int(round(delay * (1.0 / L + 1.0 / M)))
        if shift < len(x_rec):
            x_rec = x_rec[shift:]
    n = min(len(x), len(x_rec))
    ref, test = x[:n], x_rec[:n]
    noise = ref - test
    mse = float(np.mean(noise ** 2))
    p_sig = float(np.sum(ref ** 2))
    p_noise = float(np.sum(noise ** 2))
    snr_db = 120.0 if p_noise < 1e-15 else 10.0 * np.log10(p_sig / p_noise)
    return y_out, snr_db, mse

def compute_thd(x: np.ndarray, fs: float, f0: float):
    N = len(x)
    if N < 16 or f0 >= fs / 2.0:
        return 0.0
    win = np.hanning(N)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    fund_idx = int(np.argmin(np.abs(freqs - f0)))
    mask = np.ones_like(spec, dtype=bool)
    mask[:3] = False
    mask[max(0, fund_idx - 2):min(len(mask), fund_idx + 3)] = False
    fund_power = np.sum(spec[max(0, fund_idx - 2):min(len(mask), fund_idx + 3)] ** 2)
    dist_power = np.sum(spec[mask] ** 2)
    if fund_power < 1e-15:
        return 0.0
    return float(100.0 * np.sqrt(dist_power / fund_power))

def array_to_wav_bytes(audio: np.ndarray, fs: float):
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1.0, 1.0), max(1, int(round(fs))), format="WAV")
    buf.seek(0)
    return buf

# --- Sidebar Controls ---
st.sidebar.header("1. Input Audio Source")
source_mode = st.sidebar.radio(
    "Select Source",
    ["Synthetic Test Signal", "Upload WAV File", "Record via Microphone"]
)

f_tone1 = 1000.0
if source_mode == "Synthetic Test Signal":
    fs_in = float(st.sidebar.number_input(
        "Enter Input Sampling Rate Fs_in (Hz)",
        min_value=1000.0, max_value=192000.0, value=48000.0, step=100.0, format="%.4f"
    ))
    f_tone1 = st.sidebar.slider("Tone 1 Frequency (Hz)", 200, 5000, 1000, step=100)
    f_tone2 = st.sidebar.slider("Tone 2 / High-Freq Component (Hz)", 1000, 20000, 9500, step=500)
    duration = 1.5
    t = np.arange(int(fs_in * duration)) / fs_in
    x_in = 0.5 * np.sin(2 * np.pi * f_tone1 * t) + 0.3 * np.sin(2 * np.pi * f_tone2 * t)
elif source_mode == "Upload WAV File":
    uploaded = st.sidebar.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded is not None:
        x_in, fs_in = sf.read(uploaded)
        fs_in = float(fs_in)
        if x_in.ndim > 1:
            x_in = np.mean(x_in, axis=1)
        x_in = x_in.astype(np.float64)
        st.sidebar.info(f"Detected File Sampling Rate: **{fs_in:g} Hz**")
    else:
        st.info("Upload a .wav file in the sidebar, or choose 'Synthetic Test Signal' / 'Record via Microphone'.")
        st.stop()
else:
    mic_audio = st.sidebar.audio_input("Record voice sample")
    if mic_audio is not None:
        x_in, fs_in = sf.read(mic_audio)
        fs_in = float(fs_in)
        if x_in.ndim > 1:
            x_in = np.mean(x_in, axis=1)
        x_in = x_in.astype(np.float64)
        st.sidebar.info(f"Detected Mic Sampling Rate: **{fs_in:g} Hz**")
    else:
        st.info("Click the microphone icon in the sidebar to record audio.")
        st.stop()

st.sidebar.header("2. Target Sampling Rate (User Input)")
fs_out = float(st.sidebar.number_input(
    "Enter Target Sampling Rate Fs_out (Hz)",
    min_value=1000.0, max_value=192000.0, value=16000.0, step=100.0, format="%.4f"
))

st.sidebar.header("3. FIR Anti-Aliasing Filter Settings")
use_filter = st.sidebar.checkbox("Enable Anti-Aliasing / Anti-Imaging FIR Filter", value=True)
numtaps = st.sidebar.slider("FIR Filter Taps (Order + 1)", 15, 255, 101, step=2)
beta = st.sidebar.slider("Kaiser Window Beta", 0.0, 12.0, 5.0, step=0.5)

# --- Rationality Determinator Analysis ---
rat_info = analyze_rational_ratio(fs_in, fs_out)
L, M = rat_info["L"], rat_info["M"]

if rat_info["is_practical"]:
    st.success(f"**Rational Resampling Determinator:** {rat_info['regime']} — {rat_info['note']}")
else:
    st.warning(f"**Rational Resampling Determinator:** {rat_info['regime']} — {rat_info['note']}")

# --- Main Resampling & Distortion Evaluation ---
y_out, snr_val, mse_val = evaluate_round_trip(x_in, L, M, use_filter, numtaps, beta)
thd_out = compute_thd(y_out, rat_info["approx_fs_out"], f_tone1)

# --- Top Metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rational Ratio (L / M)", f"{L} / {M}")
c2.metric("Sample Rate Conversion", f"{fs_in:g} → {fs_out:g} Hz")
c3.metric("Round-Trip SNR (dB)", f"{snr_val:.2f} dB")
c4.metric(
    "Output THD+N (%)" if source_mode == "Synthetic Test Signal" else "Round-Trip MSE",
    f"{thd_out:.3f} %" if source_mode == "Synthetic Test Signal" else f"{mse_val:.2e}"
)

# --- Audio Players & Download ---
col_a, col_b = st.columns(2)
with col_a:
    st.subheader(f"Original Audio ({fs_in:g} Hz)")
    st.audio(array_to_wav_bytes(x_in, fs_in), format="audio/wav")
with col_b:
    st.subheader(f"Resampled Audio ({rat_info['approx_fs_out']:g} Hz)")
    out_bytes = array_to_wav_bytes(y_out, rat_info["approx_fs_out"])
    st.audio(out_bytes, format="audio/wav")
    st.download_button(
        label=f"Download Resampled WAV ({int(round(rat_info['approx_fs_out']))} Hz)",
        data=out_bytes,
        file_name=f"resampled_{int(round(fs_in))}_to_{int(round(rat_info['approx_fs_out']))}.wav",
        mime="audio/wav"
    )

# --- Visualizations (Time, PSD, and FIR Filter Frequency Response) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 1. Time Domain
t_orig_ms = np.arange(min(len(x_in), int(0.005 * fs_in))) / fs_in * 1000
t_res_ms = np.arange(min(len(y_out), int(0.005 * rat_info["approx_fs_out"]))) / rat_info["approx_fs_out"] * 1000
axes[0].plot(t_orig_ms, x_in[:len(t_orig_ms)], label=f"Original ({fs_in:g} Hz)", alpha=0.8)
axes[0].plot(t_res_ms, y_out[:len(t_res_ms)], "o--", markersize=3, label=f"Resampled ({rat_info['approx_fs_out']:g} Hz)", alpha=0.85)
axes[0].set_title("Time Domain (First 5 ms)")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Amplitude")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Power Spectral Density
f1, Pxx1 = signal.welch(x_in, fs_in, nperseg=min(2048, len(x_in)))
f2, Pxx2 = signal.welch(y_out, rat_info["approx_fs_out"], nperseg=min(2048, len(y_out)))
axes[1].semilogy(f1 / 1000, Pxx1, label=f"Original PSD ({fs_in:g} Hz)")
axes[1].semilogy(f2 / 1000, Pxx2, label=f"Resampled PSD ({rat_info['approx_fs_out']:g} Hz)")
axes[1].set_title("Power Spectral Density (Aliasing Check)")
axes[1].set_xlabel("Frequency (kHz)")
axes[1].set_ylabel("PSD (V²/Hz)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. FIR Anti-Aliasing / Anti-Imaging Filter Response
taps, cutoff = design_resampler_fir(L, M, numtaps, beta)
w, h = signal.freqz(taps / L, worN=2048)
axes[2].plot(w / np.pi, 20 * np.log10(np.maximum(np.abs(h), 1e-8)), label=f"Kaiser FIR ({numtaps} taps)")
axes[2].axvline(cutoff, color="r", linestyle="--", label=f"Cutoff ωc/π = min(1/L, 1/M) = {cutoff:.3f}")
axes[2].set_title("FIR LPF Magnitude Response")
axes[2].set_xlabel("Normalized Frequency (×π rad/sample)")
axes[2].set_ylabel("Gain (dB)")
axes[2].set_ylim([-100, 5])
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# --- Side-by-Side Distortion Benchmark Table ---
st.subheader("Distortion Comparison Across Filter Configurations")
configs = [
    ("No Anti-Aliasing Filter (Naive)", False, numtaps, beta),
    ("Low-Order Kaiser FIR (21 taps)", True, 21, beta),
    (f"Current Kaiser FIR ({numtaps} taps)", True, numtaps, beta),
    ("High-Order Kaiser FIR (201 taps)", True, 201, 8.0),
]
rows = []
for name, filt_flag, taps_cfg, beta_cfg in configs:
    y_tmp, snr_tmp, mse_tmp = evaluate_round_trip(x_in, L, M, filt_flag, taps_cfg, beta_cfg)
    thd_tmp = compute_thd(y_tmp, rat_info["approx_fs_out"], f_tone1)
    rows.append({
        "Configuration": name,
        "Round-Trip SNR (dB)": f"{snr_tmp:.2f}",
        "Round-Trip MSE": f"{mse_tmp:.2e}",
        "Output THD+N (%)": f"{thd_tmp:.3f}" if source_mode == "Synthetic Test Signal" else "N/A (Audio)"
    })
st.table(rows)
