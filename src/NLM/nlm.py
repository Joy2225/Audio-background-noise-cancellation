import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

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
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr, alpha=0.5, color='b', label="Original")
    plt.title("Original Audio")
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(denoised_audio, sr=sr, alpha=0.5, color='g', label="Denoised")
    plt.title("Denoised Audio")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Input and output file paths
    input_wav = "noisefunkguitare.wav"  # Replace with your input file path
    output_wav = "denoised_audio.wav"  # Replace with your desired output file path

    # Call the main function
    main(input_wav, output_wav)
