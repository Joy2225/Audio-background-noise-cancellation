from scipy.io import wavfile
import noisereduce as nr
import numpy as np
from scipy.signal import resample

# Load data
rate, data = wavfile.read("../../data/inputs/noisefunkguitare.wav")
print(f"Sample rate: {rate}, Data shape: {data.shape}")

# Downsample if sample rate is very high (e.g., above 44.1 kHz)
# target_rate = 22050  # Target sample rate for lower memory use
# if rate > target_rate:
#     data = resample(data, int(len(data) * target_rate / rate))
#     rate = target_rate

# Noise reduction parameters to reduce memory usage
n_fft = 1024  # FFT size
win_length = 1024  # Window length

# Process each channel separately if stereo, or directly if mono
if len(data.shape) > 1:  # Stereo
    reduced_noise = np.array(
        [nr.reduce_noise(y=data[:, i], sr=rate, n_fft=n_fft, win_length=win_length) for i in range(data.shape[1])]
    ).T
else:  # Mono
    reduced_noise = nr.reduce_noise(y=data, sr=rate, n_fft=n_fft, win_length=win_length)

# Convert to int16 for WAV output
reduced_noise = np.int16(reduced_noise / np.max(np.abs(reduced_noise)) * 32767)

# Write the result to a file
wavfile.write("../../data/outputs/SpectralGrating/noisefunkguitare_denoised.wav", rate, reduced_noise)
