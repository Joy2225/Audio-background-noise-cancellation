import psutil
import os
from scipy.io import wavfile
import noisereduce as nr
import numpy as np
from scipy.signal import resample
import time


from pystoi import stoi
from pesq import pesq
import mir_eval
import torch


process = psutil.Process(os.getpid())
start=time.time()


def si_snr(estimate, reference, epsilon=1e-8):
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

    si_snr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
    return si_snr.item()









# Load data
rate, data = wavfile.read("noisefunkguitare.wav")
print(f"Sample rate: {rate}, Data shape: {data.shape}")

# Convert to mono if audio is stereo
if len(data.shape) > 1:  # Stereo
    print("Converting stereo to mono...")
    data = np.mean(data, axis=1)

# Downsample if sample rate is very high (e.g., above 44.1 kHz)
target_rate = 16000  # Target sample rate for lower memory use
if rate > target_rate:
    data = resample(data, int(len(data) * target_rate / rate))
    rate = target_rate

# Noise reduction parameters to reduce memory usage
n_fft = 2048  # FFT size
win_length = 512  # Window length

# Process each channel separately if stereo, or directly if mono
if len(data.shape) > 1:  # Stereo
    reduced_noise = np.array([nr.reduce_noise(y=data[:, i], sr=rate, n_fft=n_fft, win_length=win_length) 
                              for i in range(data.shape[1])]).T
else:  # Mono
    reduced_noise = nr.reduce_noise(y=data, sr=rate, n_fft=n_fft, win_length=win_length)

# Convert to int16 for WAV output
reduced_noise = np.int16(reduced_noise / np.max(np.abs(reduced_noise)) * 32767)

# Write the result to a file
wavfile.write("mywav_reduced_noise.wav", rate, reduced_noise)
end=time.time()
print(f"Time taken: {end-start} seconds")
print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")




reference_signal = data.astype(float)
enhanced_signal = reduced_noise.astype(float)



pesq_score = pesq(rate, reference_signal, enhanced_signal, 'wb')

# SI-SNR
si_snr_score = si_snr(enhanced_signal, reference_signal)
(sdr, _, _, _,) = mir_eval.separation.bss_eval_sources(reference_signal.numpy(), enhanced_signal.numpy(), False)


# STOI
stoi_score = stoi(reference_signal, enhanced_signal, rate)

# SDR
sdr, sir, sar, perm = mir_eval.bss_eval_sources(reference_signal[np.newaxis, :], enhanced_signal[np.newaxis, :])

# Display Results
print(f"PESQ Score: {pesq_score}")
print(f"SI-SNR Score: {si_snr_score} dB")
print(f"STOI Score: {stoi_score}")
print(f"SDR Score: {sdr[0]} dB")