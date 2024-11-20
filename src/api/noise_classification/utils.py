import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import welch
import librosa.display
from pystoi import stoi
from pesq import pesq
from scipy.signal import resample
from mir_eval.separation import bss_eval_sources


def compute_si_snr(reference, enhanced):
    reference = reference - np.mean(reference)
    enhanced = enhanced - np.mean(enhanced)
    dot_product = np.dot(reference, enhanced)
    projection = dot_product * reference / np.dot(reference, reference)
    noise = enhanced - projection
    si_snr = 10 * np.log10(np.dot(projection, projection) / np.dot(noise, noise))
    return si_snr



def load_and_normalize_audio(file_path):
    """
    Load an audio file and normalize the signal.
    """
    y, sr = librosa.load(file_path, sr=None)
    y = y / np.max(np.abs(y))
    return y, sr


def compute_stft(y, n_fft=2048, hop_length=512):
    """
    Compute the Short-Time Fourier Transform (STFT).
    """
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    stft_magnitude = np.abs(stft)
    stft_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
    return stft_db


def compute_mfcc(y, sr, n_mfcc=13):
    """
    Compute Mel-frequency cepstral coefficients (MFCCs).
    """
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


def compute_psd(y, sr, nperseg=1024):
    """
    Compute Power Spectral Density (PSD).
    """
    freqs, psd = welch(y, fs=sr, nperseg=nperseg)
    return freqs, psd


def classify_noise_by_slope(freqs, psd):
    """
    Classify noise based on the slope of the log-log PSD.
    """
    log_freqs = np.log10(freqs[1:])  # Skip 0 Hz to avoid log(0)
    log_psd = np.log10(psd[1:])
    slope, _ = np.polyfit(log_freqs, log_psd, 1)

    if -0.1 <= slope <= 0.1:
        return "White Noise"
    elif -1.2 < slope <= -0.8:
        return "Pink Noise"
    elif slope < -1.2:
        return "Brownian Noise"
    elif 0.2 <= slope < 1.0:
        return "Blue Noise"
    elif slope >= 1.0:
        return "Violet Noise"
    elif 0.1 < slope < 0.2:
        return "Grey Noise"
    elif -0.2 <= slope < -0.1:
        return "Velvet Noise"
    else:
        return "Unknown"


# Metrics calculation
def metrics(reference, enhanced, rate):
    print("\nCalculating Metrics...")

    # Convert signals to float for processing
    reference_signal = reference.astype(float)
    enhanced_signal = enhanced.astype(float)

    # Ensure mono signals
    if reference_signal.ndim > 1:
        reference_signal = reference_signal.mean(axis=1)  # Convert stereo to mono
    if enhanced_signal.ndim > 1:
        enhanced_signal = enhanced_signal.mean(axis=1)

    # Resampling for PESQ compatibility
    target_rate = 16000  # Target sample rate for PESQ compatibility
    if rate != target_rate:
        print(f"Resampling from {rate} Hz to {target_rate} Hz for PESQ compatibility...")
        reference_signal_1 = resample(reference_signal, int(len(reference_signal) * target_rate / rate))
        enhanced_signal_1 = resample(enhanced_signal, int(len(enhanced_signal) * target_rate / rate))
        rate = target_rate

    # Ensure the signals are 1D
    reference_signal = np.squeeze(reference_signal)
    enhanced_signal = np.squeeze(enhanced_signal)

    if reference_signal.ndim != 1 or enhanced_signal.ndim != 1:
        raise ValueError("Signals must be 1D arrays for PESQ.")

    # PESQ Score
    pesq_score = pesq(rate, reference_signal_1, enhanced_signal_1, "wb")

    # SI-SNR
    si_snr_score = compute_si_snr(reference_signal, enhanced_signal)

    # STOI
    stoi_score = stoi(reference_signal, enhanced_signal, rate)

    # SDR
    sdr, sir, sar, perm = bss_eval_sources(reference_signal[np.newaxis, :], enhanced_signal[np.newaxis, :])

    # Display Results
    results = {
        "PESQ Score": pesq_score,
        "SI-SNR Score": f"{si_snr_score:.2f} dB",
        "STOI Score": f"{stoi_score:.2f}",
        "SDR Score": f"{sdr[0]:.2f} dB"
    }
    return results