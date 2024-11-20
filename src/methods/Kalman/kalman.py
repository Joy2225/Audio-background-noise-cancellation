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
import os

# Function for Kalman Filtering Denoising
def kalman_filter_denoising(audio, process_noise=1e-5, measurement_noise=1e-2):
    """
    Apply Kalman filtering for audio denoising.

    Parameters:
    - audio: numpy array, input audio signal.
    - process_noise: float, process noise covariance.
    - measurement_noise: float, measurement noise covariance.

    Returns:
    - denoised_audio: numpy array, the denoised audio signal.
    """
    n = len(audio)
    denoised_audio = np.zeros(n)

    # Initialize Kalman filter parameters
    x_hat = 0  # Estimated state
    p = 1  # Estimated error covariance
    q = process_noise  # Process noise covariance
    r = measurement_noise  # Measurement noise covariance

    for i in range(n):
        # Prediction step
        x_hat_prior = x_hat
        p_prior = p + q

        # Update step
        k = p_prior / (p_prior + r)  # Kalman gain
        x_hat = x_hat_prior + k * (audio[i] - x_hat_prior)
        p = (1 - k) * p_prior

        denoised_audio[i] = x_hat

    return denoised_audio

# SI-SNR computation
def compute_si_snr(reference, enhanced):
    reference = reference - np.mean(reference)
    enhanced = enhanced - np.mean(enhanced)
    dot_product = np.dot(reference, enhanced)
    projection = dot_product * reference / np.dot(reference, reference)
    noise = enhanced - projection
    si_snr = 10 * np.log10(np.dot(projection, projection) / np.dot(noise, noise))
    return si_snr

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
    pesq_score = pesq(rate, reference_signal_1, enhanced_signal_1, 'wb')

    # SI-SNR
    si_snr_score = abs(compute_si_snr(reference_signal, enhanced_signal))

    # STOI
    stoi_score = stoi(reference_signal, enhanced_signal, rate)

    # SDR
    sdr, sir, sar, perm = bss_eval_sources(reference_signal[np.newaxis, :], enhanced_signal[np.newaxis, :])

    # Display Results
    print(f"PESQ Score: {pesq_score}")
    print(f"SI-SNR Score: {si_snr_score:.2f} dB")
    print(f"STOI Score: {stoi_score:.2f}")
    print(f"SDR Score: {sdr[0]:.2f} dB")

def main(input_wav, output_wav):
    """
    Main function to load an input WAV file, apply Kalman filtering denoising,
    save the denoised audio, and calculate metrics.

    Parameters:
    - input_wav: str, path to the input WAV file.
    - output_wav: str, path to save the output denoised WAV file.
    - reference_wav: str, optional path to a reference clean WAV file for metrics.
    """
    # Load input audio
    audio, sr = librosa.load(input_wav, sr=None)

    # Apply Kalman filter denoising
    denoised_audio = kalman_filter_denoising(audio)

    # Save the denoised audio
    sf.write(output_wav, denoised_audio, sr)
    print(f"Denoised audio saved to: {output_wav}")

    
    metrics(audio, denoised_audio, sr)

    # Plot original and denoised signals
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr, alpha=0.5, color='b', label="Original")
    plt.title("Original Audio")
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(denoised_audio, sr=sr, alpha=0.5, color='g', label="Denoised")
    plt.title("Denoised Audio")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process = psutil.Process(os.getpid())
    start=time.time()
    # Input, output, and reference file paths
    input_wav = "noisefunkguitare.wav"  # Replace with your input file path
    output_wav = "denoised_audio.wav"  # Replace with your desired output file path

    # Call the main function
    main(input_wav, output_wav)
    end=time.time()
    print(f"Time taken: {end-start} seconds")
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
