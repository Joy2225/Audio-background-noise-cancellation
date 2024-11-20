import numpy as np
import scipy.io.wavfile as wav
import os


# Function to add noise to the audio
def add_noise_to_audio(input_file, output_file, noise_type="white", noise_level=0.3):
    # Load the original audio file
    fs, audio_data = wav.read(input_file)

    # Check if the audio is stereo or mono, and handle accordingly
    if len(audio_data.shape) > 1:
        # If stereo, convert to mono by averaging channels
        audio_data = np.mean(audio_data, axis=1)

    # Normalize the audio data to [-1, 1]
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Generate the noise based on the selected type
    noise = None
    if noise_type == "white":
        noise = np.random.randn(len(audio_data))  # White noise
    elif noise_type == "pink":
        noise = generate_pink_noise(len(audio_data), fs)
    elif noise_type == "brownian":
        noise = generate_brownian_noise(len(audio_data), fs)
    elif noise_type == "blue":
        noise = generate_blue_noise(len(audio_data), fs)
    elif noise_type == "violet":
        noise = generate_violet_noise(len(audio_data), fs)

    # Normalize noise to [-1, 1]
    noise = noise / np.max(np.abs(noise))

    # Scale the noise by the desired noise level
    noisy_audio = audio_data + noise_level * noise

    # Clip the values to be between -1 and 1
    noisy_audio = np.clip(noisy_audio, -1, 1)

    # Rescale the noisy audio to the original integer range for WAV format
    noisy_audio_int = (noisy_audio * 32767).astype(np.int16)

    # Save the noisy audio to a new .wav file
    wav.write(output_file, fs, noisy_audio_int)


# Function to process all .wav files in a given directory
def process_directory(input_dir, noise_type="white", noise_level=0.05):
    # Get all .wav files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            # Define input and output file paths
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(".", f"noisy_{noise_type}_{filename}")

            # Add noise to the audio and save it
            add_noise_to_audio(input_file, output_file, noise_type, noise_level)
            print(f"Processed {filename}, saved noisy audio as {output_file}")


# Noise generators


def generate_pink_noise(length, fs):
    # A simple method to generate pink noise (1/f spectrum)
    # Using a Fourier transform to create pink noise
    freqs = np.fft.rfftfreq(length, 1 / fs)
    white_noise = np.random.randn(length)
    spectrum = np.fft.rfft(white_noise)

    # Apply 1/f scaling
    spectrum[1:] /= np.sqrt(freqs[1:])  # Apply 1/f scaling to all but the DC component
    pink_noise = np.fft.irfft(spectrum)
    return pink_noise


def generate_brownian_noise(length, fs):
    # Brownian noise is the integral of white noise (random walk)
    return np.cumsum(np.random.randn(length))


def generate_blue_noise(length, fs):
    # Blue noise has a spectral density proportional to f
    freqs = np.fft.rfftfreq(length, 1 / fs)
    white_noise = np.random.randn(length)
    spectrum = np.fft.rfft(white_noise)
    spectrum *= np.sqrt(freqs)  # Apply f scaling
    blue_noise = np.fft.irfft(spectrum)
    return blue_noise


def generate_violet_noise(length, fs):
    # Violet noise has a spectral density proportional to f^2
    freqs = np.fft.rfftfreq(length, 1 / fs)
    white_noise = np.random.randn(length)
    spectrum = np.fft.rfft(white_noise)
    spectrum *= freqs  # Apply f^2 scaling
    violet_noise = np.fft.irfft(spectrum)
    return violet_noise


# Example usage
input_dir = "/home/dhruv/Programming/CollegeProjects/Sem5/DAA/Audio-background-noise-cancellation/src/addNoiseType/noiseSamples/clean/wav"

# Process all noise types
noise_types = ["white", "pink", "brownian", "blue", "violet"]
for noise_type in noise_types:
    process_directory(input_dir, noise_type, noise_level=0.3)
