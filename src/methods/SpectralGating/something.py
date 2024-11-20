from scipy.io import wavfile
import noisereduce as nr
import numpy as np
from scipy.signal import resample
from pystoi import stoi
from pesq import pesq
from mir_eval.separation import bss_eval_sources

# Function to compute SI-SNR
def compute_si_snr(reference, enhanced):
    reference = reference - np.mean(reference)
    enhanced = enhanced - np.mean(enhanced)
    dot_product = np.dot(reference, enhanced)
    projection = dot_product * reference / np.dot(reference, reference)
    noise = enhanced - projection
    si_snr = 10 * np.log10(np.dot(projection, projection) / np.dot(noise, noise))
    return si_snr

# Load data
rate, data = wavfile.read("noisefunkguitare.wav")
print(f"Original Sample Rate: {rate}, Data shape: {data.shape}")

# Convert to mono if audio is stereo
if len(data.shape) > 1:  # Stereo
    print("Converting stereo to mono...")
    data = np.mean(data, axis=1)

# Resample if needed for PESQ (8000 Hz or 16000 Hz required)
target_rate = 16000
if rate != target_rate:
    print(f"Resampling from {rate} Hz to {target_rate} Hz for PESQ compatibility...")
    data = resample(data, int(len(data) * target_rate / rate))
    rate = target_rate

# Noise reduction parameters to reduce memory usage
n_fft = 2048  # FFT size
win_length = 512  # Window length

# Apply noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate, n_fft=n_fft, win_length=win_length)

# Convert to int16 for WAV output
reduced_noise = np.int16(reduced_noise / np.max(np.abs(reduced_noise)) * 32767)

# Write the result to a file
output_file = "mywav_reduced_noise.wav"
wavfile.write(output_file, rate, reduced_noise)

# Metrics Calculation
print("\nCalculating Metrics...")
# Assuming original data is clean and noisefree for metrics calculation
reference_signal = data.astype(float)
enhanced_signal = reduced_noise.astype(float)

# PESQ (Wideband mode)
pesq_score = pesq(rate, reference_signal, enhanced_signal, 'wb')

# SI-SNR
si_snr_score = compute_si_snr(reference_signal, enhanced_signal)

# STOI
stoi_score = stoi(reference_signal, enhanced_signal, rate)

# SDR
sdr, sir, sar, perm = bss_eval_sources(reference_signal[np.newaxis, :], enhanced_signal[np.newaxis, :])

# Display Results
print(f"PESQ Score: {pesq_score}")
print(f"SI-SNR Score: {si_snr_score} dB")
print(f"STOI Score: {stoi_score}")
print(f"SDR Score: {sdr[0]} dB")
