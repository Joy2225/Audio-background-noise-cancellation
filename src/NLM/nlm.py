import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import mir_eval
from pystoi import stoi
from pesq import pesq
from scipy.signal import resample
import scipy.io.wavfile as wav

SAMPLE_RATE = 16000

def si_snr(estimate, reference, epsilon=1e-8):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR) using PyTorch.
    """
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
    mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
    scale = mix_pow / (reference_pow + epsilon)

    reference = scale * reference
    error = estimate - reference

    reference_pow = reference.pow(2)
    error_pow = error.pow(2)

    reference_pow = reference_pow.mean(axis=1)
    error_pow = error_pow.mean(axis=1)

    si_snr = 10 * torch.log10(reference_pow + epsilon) - 10 * torch.log10(error_pow + epsilon)
    return si_snr.item()

# Evaluation function integrating SI-SNR, SDR, PESQ, and STOI
def evaluate(estimate, reference):
    """
    Evaluates the enhanced signal using SI-SNR, SDR, PESQ, and STOI metrics.
    """
    si_snr_score = si_snr(estimate, reference)
    (
        sdr,
        _,
        _,
        _,
    ) = mir_eval.separation.bss_eval_sources(reference.numpy(), estimate.numpy(), False)
    pesq_score = pesq(SAMPLE_RATE, estimate[0].numpy(), reference[0].numpy(), "wb")
    stoi_score = stoi(reference[0].numpy(), estimate[0].numpy(), SAMPLE_RATE, extended=False)

    return si_snr_score, sdr[0], pesq_score, stoi_score



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
    return denoised_audio

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

    if len(data.shape) > 1:
            print("Converting stereo to mono...")
            data = np.mean(data, axis=1)

    # Resample if needed for PESQ compatibility
    if rate != SAMPLE_RATE:
        print(f"Resampling from {rate} Hz to {SAMPLE_RATE} Hz for PESQ compatibility...")
        data = resample(data, int(len(data) * SAMPLE_RATE / rate))
        rate = SAMPLE_RATE
    else:
        rate = rate
    # Convert to mono if needed
    if len(enhanced_signal.shape) > 1:
        enhanced_signal = np.mean(enhanced_signal, axis=1)

    # Align lengths of reference and enhanced signals
    min_length = min(len(data), len(enhanced_signal))
    reference_signal = data[:min_length].astype(float)
    enhanced_signal = enhanced_signal[:min_length].astype(float)

    # Convert signals to PyTorch tensors
    reference_tensor = torch.tensor(reference_signal).unsqueeze(0)  # Add batch dimension
    enhanced_tensor = torch.tensor(enhanced_signal).unsqueeze(0)  # Add batch dimension

    # Evaluate metrics
    print("\nCalculating Metrics...")
    si_snr_score, sdr_score, pesq_score, stoi_score = evaluate(enhanced_tensor, reference_tensor)

    # Display Results
    print(f"PESQ Score: {pesq_score}")
    print(f"SI-SNR Score: {si_snr_score:.2f} dB")
    print(f"STOI Score: {stoi_score:.2f}")
    print(f"SDR Score: {sdr_score:.2f} dB")

if __name__ == "__main__":
    # Input and output file paths
    input_wav = "noisefunkguitare.wav"  # Replace with your input file path
    output_wav = "denoised_audio.wav"  # Replace with your desired output file path

    # Call the main function
    main(input_wav, output_wav)
