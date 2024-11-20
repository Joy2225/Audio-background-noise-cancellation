import librosa
import numpy as np
import pywt
import soundfile as sf
from scipy.io import wavfile
from tqdm import tqdm
from tqdm import tqdm
from scipy.signal import resample
from pystoi import stoi
from pesq import pesq
from mir_eval.separation import bss_eval_sources
import torch  # Importing PyTorch for tensor operations
from scipy.io import wavfile




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
    pesq_score = pesq(rate, reference_signal_1, enhanced_signal_1, "wb")

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

def mad(coefficients):
    """Median absolute deviation (MAD) for noise estimation."""
    return np.median(np.abs(coefficients - np.median(coefficients)))


def audio_deNoise(input_data, sample_rate, wavelet="db4", level=2):
    """
    Performs noise reduction on the input audio data (ndarray) and returns the denoised audio as a ndarray.

    Parameters:
    - input_data: ndarray containing audio samples
    - sample_rate: The sample rate of the audio
    - wavelet: The wavelet type used for decomposition
    - level: Level of wavelet decomposition

    Returns:
    - Denoised audio as a ndarray
    """
    # Initialize variables
    noiseProfile = None
    output_data = []

    # Process data in chunks
    chunk_size = int(sample_rate * 0.10)  # 10% of the audio duration
    for i in tqdm(range(0, len(input_data), chunk_size)):
        # Create a block (chunk) of data
        block = input_data[i : i + chunk_size]

        # Wavelet decomposition
        coefficients = pywt.wavedec(block, wavelet, mode="per", level=level)

        # Noise estimation and threshold calculation
        sigma = mad(coefficients[-1])  # Noise standard deviation estimation
        thresh = sigma * np.sqrt(2 * np.log(len(block)))  # Universal threshold

        # Apply thresholding on wavelet coefficients
        coefficients[1:] = [pywt.threshold(i, value=thresh, mode="soft") for i in coefficients[1:]]

        # Reconstruct the clean signal
        clean_block = pywt.waverec(coefficients, wavelet, mode="per")

        # Append the denoised block to the output data
        output_data.append(clean_block)

    # Concatenate all blocks into a single array
    output_data = np.concatenate(output_data)

    # Ensure the output data is the same length as the input (trim if necessary)
    if len(output_data) > len(input_data):
        output_data = output_data[: len(input_data)]

    return output_data


input,sr = librosa.load('./noisefunkguitare.wav',sr=None)
output = audio_deNoise(input,sr)
sf.write('./denoised.wav',output,sr)
metrics(input,output,sr)