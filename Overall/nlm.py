import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import resample
from pesq import pesq
from pystoi import stoi
from mir_eval.separation import bss_eval_sources
import time
import psutil


def compute_si_snr(reference, enhanced):
    reference = reference - np.mean(reference)
    enhanced = enhanced - np.mean(enhanced)
    dot_product = np.dot(reference, enhanced)
    projection = dot_product * reference / np.dot(reference, reference)
    noise = enhanced - projection
    si_snr = 10 * np.log10(np.dot(projection, projection) / np.dot(noise, noise))
    return si_snr


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
    si_snr_score = abs(compute_si_snr(reference_signal, enhanced_signal))

    # STOI
    stoi_score = stoi(reference_signal, enhanced_signal, rate)

    # SDR
    sdr, sir, sar, perm = bss_eval_sources(reference_signal[np.newaxis, :], enhanced_signal[np.newaxis, :])

    metrics_dict = {
        "pesq": pesq_score,
        "si_snr": si_snr_score,
        "stoi": stoi_score,
        "sdr": sdr[0],
    }
    return metrics_dict


def non_local_means_denoising(audio, patch_size=512, search_window=1024, h=0.8):
    """
    Perform Non-Local Means denoising on an audio signal.

    Parameters:
    - audio: numpy array, the input audio signal.
    - patch_size: int, size of the patches to compare.
    - search_window: int, size of the search window for similar patches.
    - h: float, filtering parameter (higher values result in more smoothing).

    Returns:
    - denoised_audio: numpy array, the denoised audio signal.
    """
    n = len(audio)
    denoised_audio = np.zeros(n)
    weights_sum = np.zeros(n)

    for i in range(0, n - patch_size, patch_size // 2):
        patch = audio[i : i + patch_size]

        for j in range(max(0, i - search_window), min(n - patch_size, i + search_window)):
            neighbor_patch = audio[j : j + patch_size]
            dist = np.sum((patch - neighbor_patch) ** 2)
            weight = np.exp(-dist / (h * h * patch_size))
            denoised_audio[i : i + patch_size] += weight * neighbor_patch
            weights_sum[i : i + patch_size] += weight

    # Normalize the weights
    denoised_audio /= np.maximum(weights_sum, 1e-8)
    return denoised_audio


if __name__ == "__main__":
    path = "../data/noiseSamples/noised"
    mem = 0
    time_taken = 0
    pesq_score_avg = 0
    si_snr_score_avg = 0
    stoi_score_avg = 0
    sdr_score_avg = 0

    for file in os.listdir(path):
        if not file.endswith(".wav"):
            continue
        process = psutil.Process(os.getpid())
        start = time.time()

        output_wav = "denoised_audio.wav"
        file, sr = librosa.load(f"{path}/{file}", sr=None)
        denoised_audio = non_local_means_denoising(file)
        metrics_dict = metrics(file, denoised_audio, sr)
        end = time.time()
        time_taken += end - start
        pesq_score_avg += metrics_dict["pesq"]
        si_snr_score_avg += metrics_dict["si_snr"]
        stoi_score_avg += metrics_dict["stoi"]
        sdr_score_avg += metrics_dict["sdr"]

        mem += process.memory_info().rss / 1024**2

    print(f"Average Time taken: {(end-start)/100} seconds")
    print(f"Avg Memory usage: {mem/100} MB")
    print(f"Avg PESQ Score: {pesq_score_avg/100}")
    print(f"Avg SI-SNR Score: {si_snr_score_avg/100:.2f}")
    print(f"Avg STOI Score: {stoi_score_avg/100:.2f}")
    print(f"Avg SDR Score: {sdr_score_avg/100:.2f} dB")
