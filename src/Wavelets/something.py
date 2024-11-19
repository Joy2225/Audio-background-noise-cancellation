import numpy as np
import pywt
import soundfile
from tqdm import tqdm
from scipy.signal import resample
from pystoi import stoi
from pesq import pesq
import mir_eval
import torch  # Importing PyTorch for tensor operations

SAMPLE_RATE = 16000  # Global constant for target sample rate


# Function to calculate Median Absolute Deviation (MAD)
def mad(arr):
    """Median Absolute Deviation (MAD) for noise estimation."""
    arr = np.ma.array(arr).compressed()
    med = np.median(arr)
    return np.median(np.abs(arr - med))


# PyTorch-based SI-SNR function
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


# Main class for audio denoising
class AudioDeNoise:
    def __init__(self, inputFile, wavelet='db4', level=1, target_rate=SAMPLE_RATE):
        self.__inputFile = inputFile
        self.wavelet = wavelet
        self.level = level
        self.target_rate = target_rate

    def deNoise(self, outputFile):
        """Performs noise reduction, resampling, and calculates metrics."""
        info = soundfile.info(self.__inputFile)
        original_rate = info.samplerate
        data, _ = soundfile.read(self.__inputFile)

        # Convert stereo to mono if needed
        if len(data.shape) > 1:
            print("Converting stereo to mono...")
            data = np.mean(data, axis=1)

        # Resample if needed for PESQ compatibility
        if original_rate != self.target_rate:
            print(f"Resampling from {original_rate} Hz to {self.target_rate} Hz for PESQ compatibility...")
            data = resample(data, int(len(data) * self.target_rate / original_rate))
            rate = self.target_rate
        else:
            rate = original_rate

        # Noise reduction
        with soundfile.SoundFile(outputFile, "w", samplerate=rate, channels=1) as of:
            for block in tqdm(soundfile.blocks(self.__inputFile, int(rate * info.duration * 0.25))):
                if len(block.shape) > 1:  # Convert stereo to mono in blocks if needed
                    block = np.mean(block, axis=1)

                coefficients = pywt.wavedec(block, self.wavelet, mode='per', level=self.level)
                sigma = mad(coefficients[-1])
                thresh = sigma * np.sqrt(2 * np.log(len(block)))
                coefficients[1:] = [pywt.threshold(i, value=thresh, mode='soft') for i in coefficients[1:]]
                clean = pywt.waverec(coefficients, self.wavelet, mode='per')

                # Ensure clean is 1D before writing
                if len(clean.shape) > 1:
                    clean = np.mean(clean, axis=1)
                of.write(clean)

        # Read the denoised signal
        enhanced_signal, _ = soundfile.read(outputFile)

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
