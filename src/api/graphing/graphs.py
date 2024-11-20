import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io

def save_plot_to_buffer(plot_func, *args, **kwargs):
    """
    Save a plot to a buffer (in-memory file).
    """
    buf = io.BytesIO()
    plot_func(*args, **kwargs)  # Generate the plot
    plt.savefig(buf, format="png", bbox_inches="tight")  # Save the plot as PNG
    buf.seek(0)
    plt.close()  # Close the figure to avoid memory issues
    return buf


def plot_stft(stft_db, sr, hop_length=512):
    """
    Generate the STFT spectrogram plot.
    """
    def plot():
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(stft_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log", cmap="magma")
        plt.title("STFT Spectrogram (Log Scale)")
        plt.colorbar(label="Amplitude (dB)")
    return save_plot_to_buffer(plot)


def plot_psd(freqs, psd):
    """
    Generate the Power Spectral Density (PSD) plot.
    """
    def plot():
        plt.figure(figsize=(10, 6))
        plt.semilogx(freqs, 10 * np.log10(psd), label="PSD (dB/Hz)")
        plt.title("Power Spectral Density")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power/Frequency (dB/Hz)")
        plt.grid(True, which="both")
    return save_plot_to_buffer(plot)


def plot_mfcc(mfccs, sr):
    """
    Generate the MFCCs plot.
    """
    def plot():
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(mfccs, sr=sr, x_axis="time", cmap="viridis")
        plt.title("MFCCs")
        plt.colorbar(label="MFCC Coefficients")
    return save_plot_to_buffer(plot)

def plot_freq(audio,sr):
    def plot():
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(audio, sr=sr, alpha=0.5, color='b', label="Original")
        plt.title("Original Audio")
        plt.tight_layout()
    return save_plot_to_buffer(plot)