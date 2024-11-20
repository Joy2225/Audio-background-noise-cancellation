import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import torch
from mir_eval.separation import bss_eval_sources
from pystoi import stoi
from pesq import pesq
from scipy.signal import resample
import scipy.io.wavfile as wav
import time
import psutil
import os

SAMPLE_RATE = 16000

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
        reference_signal_1 = resample(reference_signal, int(len(reference_signal) * target_rate / rate))
        enhanced_signal_1 = resample(enhanced_signal, int(len(enhanced_signal) * target_rate / rate))
        rate = target_rate

    # Ensure the signals are 1D
    reference_signal = np.squeeze(reference_signal)
    enhanced_signal = np.squeeze(enhanced_signal)

    if reference_signal.ndim != 1 or enhanced_signal.ndim != 1:
        raise ValueError("Signals must be 1D arrays for PESQ.")

    # PESQ Score
    pesq_score = pesq(rate, reference_signal_1, enhanced_signal_1, 'wb')

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
    process = psutil.Process(os.getpid())
    start = time.time()
    n = len(audio)
    denoised_audio = np.zeros(n)
    weights_sum = np.zeros(n)

    for i in range(0, n - patch_size, patch_size // 2):
        patch = audio[i:i + patch_size]

        for j in range(max(0, i - search_window), min(n - patch_size, i + search_window)):
            neighbor_patch = audio[j:j + patch_size]

            # Compute similarity between patches
            dist = np.sum((patch - neighbor_patch) ** 2)
            weight = np.exp(-dist / (h * h * patch_size))

            # Weighted sum
            denoised_audio[i:i + patch_size] += weight * neighbor_patch
            weights_sum[i:i + patch_size] += weight

    # Normalize the weights
    denoised_audio /= np.maximum(weights_sum, 1e-8)
    end = time.time()
    return denoised_audio, {"Execution Time(sec)": end - start, "Memory Usage(MB)": process.memory_info().rss / 1024 ** 2}

def main(input_wav, output_wav, patch_size=512, search_window=1024, h=0.8):
    """
    Main function to load an input WAV file, apply Non-Local Means denoising,
    and save the denoised audio as an output WAV file.

    Parameters:
    - input_wav: str, path to the input WAV file.
    - output_wav: str, path to save the output denoised WAV file.
    - patch_size: int, size of the patches to compare.
    - search_window: int, size of the search window for similar patches.
    - h: float, filtering parameter (higher values result in more smoothing).
    """
    # Load audio
    audio, sr = librosa.load(input_wav, sr=None)

    # Apply Non-Local Means denoising
    denoised_audio = non_local_means_denoising(audio, patch_size, search_window, h)

    # Save the denoised audio
    sf.write(output_wav, denoised_audio, sr)

    # Plot and compare
    # plt.figure(figsize=(12, 6))
    # plt.subplot(2, 1, 1)
    # librosa.display.waveshow(audio, sr=sr, alpha=0.5, color='b', label="Original")
    # plt.title("Original Audio")
    # plt.subplot(2, 1, 2)
    # librosa.display.waveshow(denoised_audio, sr=sr, alpha=0.5, color='g', label="Denoised")
    # plt.title("Denoised Audio")
    # plt.tight_layout()
    # plt.show()

    rate, data = wav.read(input_wav)
    rate_enhanced, enhanced_signal = wav.read(output_wav)
    metrics(data, enhanced_signal, rate)
    # if len(data.shape) > 1:
    #         print("Converting stereo to mono...")
    #         data = np.mean(data, axis=1)

    # # Resample if needed for PESQ compatibility
    # if rate != SAMPLE_RATE:
    #     print(f"Resampling from {rate} Hz to {SAMPLE_RATE} Hz for PESQ compatibility...")
    #     data = resample(data, int(len(data) * SAMPLE_RATE / rate))
    #     rate = SAMPLE_RATE
    # else:
    #     rate = rate
    # # Convert to mono if needed
    # if len(enhanced_signal.shape) > 1:
    #     enhanced_signal = np.mean(enhanced_signal, axis=1)

    # # Align lengths of reference and enhanced signals
    # min_length = min(len(data), len(enhanced_signal))
    # reference_signal = data[:min_length].astype(float)
    # enhanced_signal = enhanced_signal[:min_length].astype(float)

    # # Convert signals to PyTorch tensors
    # reference_tensor = torch.tensor(reference_signal).unsqueeze(0)  # Add batch dimension
    # enhanced_tensor = torch.tensor(enhanced_signal).unsqueeze(0)  # Add batch dimension

    # # Evaluate metrics
    # print("\nCalculating Metrics...")
    # si_snr_score, sdr_score, pesq_score, stoi_score = evaluate(enhanced_tensor, reference_tensor)

    # # Display Results
    # print(f"PESQ Score: {pesq_score}")
    # print(f"SI-SNR Score: {si_snr_score:.2f} dB")
    # print(f"STOI Score: {stoi_score:.2f}")
    # print(f"SDR Score: {sdr_score:.2f} dB")

if __name__ == "__main__":
    process = psutil.Process(os.getpid())
    start=time.time()
    # Input and output file paths
    input_wav = "noisefunkguitare.wav"  # Replace with your input file path
    output_wav = "denoised_audio.wav"  # Replace with your desired output file path

    # Call the main function
    main(input_wav, output_wav)
    end=time.time()
    print(f"Time taken: {end-start} seconds")
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
