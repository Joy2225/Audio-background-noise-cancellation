import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf

# process = psutil.Process(os.getpid())
# start=time.time()


# Function to compute SI-SNR
def compute_si_snr(reference, enhanced):
    reference = reference - np.mean(reference)
    enhanced = enhanced - np.mean(enhanced)
    dot_product = np.dot(reference, enhanced)
    projection = dot_product * reference / np.dot(reference, reference)
    noise = enhanced - projection
    si_snr = 10 * np.log10(np.dot(projection, projection) / np.dot(noise, noise))
    return si_snr



def denoise_audio(input_file, target_rate=16000, n_fft=2048, win_length=512):
    """
    Denoises an audio file, applies resampling if needed, and returns the denoised audio as a numpy array.

    :param input_file: Path to the input audio file
    :param target_rate: Desired target sample rate for resampling
    :param n_fft: FFT size for noise reduction
    :param win_length: Window length for noise reduction
    :return: Denoised audio as a numpy array
    """
    # Load the audio file using librosa
    data, rate = librosa.load(input_file, sr=None)  # sr=None preserves the original rate
    print(f"Original sample rate: {rate}, Data length: {len(data)} samples")

    # Convert to mono if stereo
    if len(data.shape) > 1:  # Stereo
        print("Converting stereo to mono...")
        data = np.mean(data, axis=1)

    # Downsample if sample rate is very high
    if rate != target_rate:
        print(f"Resampling from {rate} Hz to {target_rate} Hz...")
        data = librosa.resample(data, orig_sr=rate, target_sr=target_rate)
        rate = target_rate

    # Noise reduction (apply to the entire audio)
    reduced_noise = nr.reduce_noise(y=data, sr=rate, n_fft=n_fft, win_length=win_length)

    # Return the denoised audio as a numpy array
    return reduced_noise

# # Metrics Calculation
# print("\nCalculating Metrics...")
# # Assuming original data is clean and noisefree for metrics calculation
# reference_signal = data.astype(float)
# enhanced_signal = reduced_noise.astype(float)

# # PESQ (Wideband mode)
# pesq_score = pesq(rate, reference_signal, enhanced_signal, 'wb')

# # SI-SNR
# si_snr_score = compute_si_snr(reference_signal, enhanced_signal)

# # STOI
# stoi_score = stoi(reference_signal, enhanced_signal, rate)

# # SDR
# sdr, sir, sar, perm = bss_eval_sources(reference_signal[np.newaxis, :], enhanced_signal[np.newaxis, :])

# # Display Results
# print(f"PESQ Score: {pesq_score}")
# print(f"SI-SNR Score: {si_snr_score} dB")
# print(f"STOI Score: {stoi_score}")
# print(f"SDR Score: {sdr[0]} dB")
# end=time.time()
# print(f"Time taken: {end-start} seconds")
# print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")