import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import welch
import librosa.display


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
