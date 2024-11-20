import librosa
from numpy import ndarray
import matplotlib.pyplot as plt
import numpy as np


def plot_stft(stft_db, sr, hop_length=512):
    """
    Plot the STFT spectrogram.
    """
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(stft_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log", cmap="magma")
    plt.title("STFT Spectrogram (Log Scale)")
    plt.colorbar(label="Amplitude (dB)")
    plt.show()


def plot_psd(freqs, psd):
    """
    Plot the Power Spectral Density (PSD).
    """
    plt.figure(figsize=(10, 6))
    plt.semilogx(freqs, 10 * np.log10(psd), label="PSD (dB/Hz)")
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.grid(True, which="both")
    plt.show()


def plot_mfcc(mfccs, sr):
    """
    Plot the MFCCs.
    """
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mfccs, sr=sr, x_axis="time", cmap="viridis")
    plt.title("MFCCs")
    plt.colorbar(label="MFCC Coefficients")
    plt.show()


def generate_audio_graph(audio: ndarray, title: str, show=False, sr=44100):
    plt.figure(figsize=(12, 6))
    if show:
        plt.show()
        return
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr, alpha=0.5, color="b", label="Original")
    plt.title(title)


def compare_audios(audio1: ndarray, audio2: ndarray, title1: str, title2: str, sr=44100):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio1, sr=sr, alpha=0.5, color="b", label=title1)
    plt.title(title1)
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(audio2, sr=sr, alpha=0.5, color="g", label=title2)
    plt.title(title2)
    plt.tight_layout()
    plt.show()
