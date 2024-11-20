import numpy as np
import pywt
import soundfile
from tqdm import tqdm
from scipy.signal import resample
from pystoi import stoi
from pesq import pesq
from mir_eval.separation import bss_eval_sources
import torch  # Importing PyTorch for tensor operations
from scipy.io import wavfile
SAMPLE_RATE = 16000  # Global constant for target sample rate





import numpy as np
import pywt
import soundfile
from tqdm import tqdm

from lib.noiseProfiler import NoiseProfiler


def mad(arr):
    """Median Absolute Deviation (MAD) for noise estimation."""
    arr = np.ma.array(arr).compressed()  # Remove masked values if any
    med = np.median(arr)
    return np.median(np.abs(arr - med))


# def si_snr(estimate, reference, epsilon=1e-8):
#     """
#     Scale-Invariant Signal-to-Noise Ratio (SI-SNR) using PyTorch.
#     """
#     estimate = estimate - estimate.mean()
#     reference = reference - reference.mean()
#     reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
#     mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
#     scale = mix_pow / (reference_pow + epsilon)

#     reference = scale * reference
#     error = estimate - reference

#     reference_pow = reference.pow(2)
#     error_pow = error.pow(2)

#     reference_pow = reference_pow.mean(axis=1)
#     error_pow = error_pow.mean(axis=1)

#     si_snr = 10 * torch.log10(reference_pow + epsilon) - 10 * torch.log10(error_pow + epsilon)
#     return si_snr.item()

# # Evaluation function integrating SI-SNR, SDR, PESQ, and STOI
# def evaluate(estimate, reference):
#     """
#     Evaluates the enhanced signal using SI-SNR, SDR, PESQ, and STOI metrics.
#     """
#     si_snr_score = si_snr(estimate, reference)
#     (
#         sdr,
#         _,
#         _,
#         _,
#     ) = mir_eval.separation.bss_eval_sources(reference.numpy(), estimate.numpy(), False)
#     pesq_score = pesq(SAMPLE_RATE, estimate[0].numpy(), reference[0].numpy(), "wb")
#     stoi_score = stoi(reference[0].numpy(), estimate[0].numpy(), SAMPLE_RATE, extended=False)

#     return si_snr_score, sdr[0], pesq_score, stoi_score

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


class AudioDeNoise:

    def __init__(self, inputFile, wavelet='db4', level=2):
        self.__inputFile = inputFile
        self.target_rate = SAMPLE_RATE
        self.__noiseProfile = None
        self.wavelet = wavelet
        self.level = level

    def deNoise(self, outputFile):
        """Performs noise reduction and saves the denoised audio to outputFile."""
        info = soundfile.info(self.__inputFile)  # Audio file info
        rate = info.samplerate

        # Open output file for writing
        with soundfile.SoundFile(outputFile, "w", samplerate=rate, channels=info.channels) as of:
            # Process file in chunks
            for block in tqdm(soundfile.blocks(self.__inputFile, int(rate * info.duration * 0.10))):
                # Wavelet decomposition
                coefficients = pywt.wavedec(block, self.wavelet, mode='per', level=self.level)

                # Noise estimation and threshold calculation
                sigma = mad(coefficients[-1])  # Noise standard deviation estimation
                thresh = sigma * np.sqrt(2 * np.log(len(block)))  # Universal threshold

                # Apply thresholding on wavelet coefficients
                coefficients[1:] = [pywt.threshold(i, value=thresh, mode='soft') for i in coefficients[1:]]

                # Reconstruct the clean signal
                clean = pywt.waverec(coefficients, self.wavelet, mode='per')
                of.write(clean)  # Write to output file
        
                # Read the denoised signal
        enhanced_signal, _ = soundfile.read(outputFile)
        
        # data, _ = soundfile.read(self.__inputFile)
        rate, data = wavfile.read(self.__inputFile)
        metrics(data, enhanced_signal, rate)

        # # Convert stereo to mono if needed
        # if len(data.shape) > 1:
        #     print("Converting stereo to mono...")
        #     data = np.mean(data, axis=1)

        # # Resample if needed for PESQ compatibility
        # if rate != self.target_rate:
        #     print(f"Resampling from {rate} Hz to {self.target_rate} Hz for PESQ compatibility...")
        #     data = resample(data, int(len(data) * self.target_rate / rate))
        #     rate = self.target_rate
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



    def __del__(self):
        """Destructor to clean up resources."""
        del self.__noiseProfile
