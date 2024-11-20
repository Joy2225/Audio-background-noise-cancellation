import os
import csv
import librosa
import numpy as np
from scipy.signal import welch

def classify_noise(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y = y / np.max(np.abs(y))

    # Compute Short-Time Fourier Transform (STFT)
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    stft_magnitude = np.abs(stft)
    stft_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Compute Power Spectral Density (PSD)
    freqs, psd = welch(y, fs=sr, nperseg=1024)

    # Log-log scale for linear fitting
    log_freqs = np.log10(freqs[1:])  # Skip 0 Hz to avoid log(0)
    log_psd = np.log10(psd[1:])

    # Linear regression to find the slope
    slope, _ = np.polyfit(log_freqs, log_psd, 1)

    # Classification based on slope
    if -0.1 <= slope <= 0.1:
        noise_type = "White Noise"
    elif -1.2 < slope <= -0.8:
        noise_type = "Pink Noise"
    elif slope < -1.2:
        noise_type = "Brownian Noise"
    elif 0.2 <= slope < 1.0:
        noise_type = "Blue Noise"
    elif slope >= 1.0:
        noise_type = "Violet Noise"
    elif 0.1 < slope < 0.2:
        noise_type = "Grey Noise"
    elif -0.2 <= slope < -0.1:
        noise_type = "Velvet Noise"
    else:
        noise_type = "Unknown"

    # # Visualize STFT spectrogram
    # plt.figure(figsize=(10, 6))
    # librosa.display.specshow(stft_db, sr=sr, hop_length=512, x_axis="time", y_axis="log", cmap="magma")
    # plt.title("STFT Spectrogram (Log Scale)")
    # plt.colorbar(label="Amplitude (dB)")
    # plt.show()

    # # Visualize PSD
    # plt.figure(figsize=(10, 6))
    # plt.semilogx(freqs, 10 * np.log10(psd), label="PSD (dB/Hz)")
    # plt.title("Power Spectral Density")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Power/Frequency (dB/Hz)")
    # plt.grid(True, which="both")
    # plt.show()

    # # Visualize MFCCs
    # plt.figure(figsize=(10, 6))
    # librosa.display.specshow(mfccs, sr=sr, x_axis="time", cmap="viridis")
    # plt.title("MFCCs")
    # plt.colorbar(label="MFCC Coefficients")
    # plt.show()

    return noise_type

def process_and_classify_noisy_audio(input_dir, output_csv="noise_classification_results.csv"):
    # Prepare the CSV to save the results
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["File Name", "Actual Noise Type", "Classified Noise Type"])

        # Process all .wav files in the noisy audio directory
        for filename in os.listdir(input_dir):
            if filename.endswith(".wav"):
                # Define the file path
                input_file = os.path.join(input_dir, filename)

                # Extract the actual noise type from the filename (assuming it's part of the filename)
                # For example, filenames could be like "noisy_white_filename.wav"
                actual_noise_type = filename.split("_")[1]  # Adjust this as per your naming convention

                # Classify the noise in the file
                classified_noise_type = classify_noise(input_file)

                # Save the results to the CSV file
                writer.writerow([filename, actual_noise_type, classified_noise_type])

                print(f"Processed {filename}, Actual: {actual_noise_type}, Classified: {classified_noise_type}")


# Directory where the noisy audio files are stored
noisy_audio_dir = "/home/dhruv/Programming/CollegeProjects/Sem5/DAA/Audio-background-noise-cancellation/src/noiseGeneration/noiseSamples/noised/"

# Run the process to classify and compare noise types
process_and_classify_noisy_audio(noisy_audio_dir)
