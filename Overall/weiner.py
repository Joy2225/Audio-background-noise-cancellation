#!/usr/bin/env python3
from scipy.fftpack import fft, ifft
import scipy.io.wavfile as wav
import scipy.signal as sg
import numpy as np
from pystoi import stoi
from pesq import pesq
from mir_eval.separation import bss_eval_sources
from scipy.signal import resample
import os
import time
import psutil


# Function to compute SI-SNR
def compute_si_snr(reference, enhanced):
    reference = reference - np.mean(reference)
    enhanced = enhanced - np.mean(enhanced)
    dot_product = np.dot(reference, enhanced)
    projection = dot_product * reference / np.dot(reference, reference)
    noise = enhanced - projection
    si_snr = 10 * np.log10(np.dot(projection, projection) / np.dot(noise, noise))
    return si_snr


# Function to calculate metrics
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
        reference_signal = resample(reference_signal, int(len(reference_signal) * target_rate / rate))
        enhanced_signal = resample(enhanced_signal, int(len(enhanced_signal) * target_rate / rate))
        rate = target_rate

    # Ensure the signals are 1D
    reference_signal = np.squeeze(reference_signal)
    enhanced_signal = np.squeeze(enhanced_signal)

    if reference_signal.ndim != 1 or enhanced_signal.ndim != 1:
        raise ValueError("Signals must be 1D arrays for PESQ.")

    # PESQ Score
    pesq_score = pesq(rate, reference_signal, enhanced_signal, 'wb')

    # SI-SNR
    si_snr_score = compute_si_snr(reference_signal, enhanced_signal)

    # STOI
    stoi_score = stoi(reference_signal, enhanced_signal, rate)

    # SDR
    sdr, sir, sar, perm = bss_eval_sources(reference_signal[np.newaxis, :], enhanced_signal[np.newaxis, :])

    # Display Results
    print(f"PESQ Score: {pesq_score}")
    print(f"SI-SNR Score: {si_snr_score:.2f} dB")
    print(f"STOI Score: {stoi_score:.2f}")
    print(f"SDR Score: {sdr[0]:.2f} dB")


# Function for half-wave rectification
def halfwave_rectification(array):
    halfwave = np.zeros(array.size)
    halfwave[np.argwhere(array > 0)] = 1
    return halfwave


# Wiener class
class Wiener:
    def __init__(self, WAV_FILE, *T_NOISE):
        self.WAV_FILE, self.T_NOISE = WAV_FILE, T_NOISE
        self.FS, self.x = wav.read(self.WAV_FILE + '.wav')

        # Ensure mono signal
        if self.x.ndim > 1:
            self.x = self.x.mean(axis=1)

        self.NFFT, self.SHIFT, self.T_NOISE = 2**10, 0.5, T_NOISE
        self.FRAME = int(0.02 * self.FS)  # Frame of 20 ms

        # Computes the offset and number of frames for overlap-add method
        self.OFFSET = int(self.SHIFT * self.FRAME)

        # Hanning window and its energy Ew
        self.WINDOW = sg.windows.hann(self.FRAME)
        self.EW = np.sum(self.WINDOW)

        length = self.x.shape[0]
        self.frames = np.arange((length - self.FRAME) // self.OFFSET + 1)

        # Evaluate noise PSD with Welch's periodogram
        self.Sbb = self.welchs_periodogram()

    @staticmethod
    def a_priori_gain(SNR):
        G = SNR / (SNR + 1)
        return G

    def welchs_periodogram(self):
        Sbb = np.zeros(self.NFFT)
        self.N_NOISE = int(self.T_NOISE[0] * self.FS), int(self.T_NOISE[1] * self.FS)
        noise_frames = np.arange(((self.N_NOISE[1] - self.N_NOISE[0]) - self.FRAME) // self.OFFSET + 1)
        for frame in noise_frames:
            i_min, i_max = frame * self.OFFSET + self.N_NOISE[0], frame * self.OFFSET + self.FRAME + self.N_NOISE[0]
            x_framed = self.x[i_min:i_max] * self.WINDOW
            X_framed = fft(x_framed, self.NFFT)
            Sbb = frame * Sbb / (frame + 1) + np.abs(X_framed)**2 / (frame + 1)
        return Sbb

    def wiener(self):
        s_est = np.zeros(self.x.shape)  # Initialize enhanced signal
        for frame in self.frames:
            i_min, i_max = frame * self.OFFSET, frame * self.OFFSET + self.FRAME
            x_framed = self.x[i_min:i_max] * self.WINDOW
            X_framed = fft(x_framed, self.NFFT)

            # Wiener filter
            SNR_post = (np.abs(X_framed)**2 / self.EW) / self.Sbb
            G = Wiener.a_priori_gain(SNR_post)
            S = X_framed * G

            # Temporal estimated signal
            temp_s_est = np.real(ifft(S)) * self.SHIFT
            s_est[i_min:i_max] += temp_s_est[:self.FRAME]  # Overlap-add

        # Normalize enhanced signal
        s_est = s_est / np.abs(s_est).max()  # Normalize to [-1, 1]

        # Metrics evaluation
        metrics(self.x, s_est, self.FS)

        # Save enhanced audio as 16-bit PCM
        wav.write(self.WAV_FILE + '_wiener.wav', self.FS, (s_est * 32767).astype(np.int16))


# Main Script
if __name__ == "__main__":
    # Monitor process for memory and execution time
    process = psutil.Process(os.getpid())
    start = time.time()

    # Define the path to the WAV file and noise intervals
    WAV_FILE = os.path.join(os.getcwd(), 'example', 'noisefunkguitare')  # Path without .wav extension
    noise_begin, noise_end = 0, 1  # Noise interval in seconds

    # Initialize the Wiener noise reduction class
    noised_audio = Wiener(WAV_FILE, noise_begin, noise_end)

    # Apply Wiener noise reduction
    print("\nApplying Wiener noise reduction...")
    noised_audio.wiener()

    # Measure execution time
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds for Wiener filter")

    # Display memory usage
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
