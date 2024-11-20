import os
import soundfile as sf
from datasets import load_dataset

# Load the dataset with streaming enabled
dataset = load_dataset("ManzhenWei/MusicSet", split="train", streaming=True)

# Create the directory if it doesn't exist
os.makedirs("x", exist_ok=True)

# Iterate over the first 10 items and save each as a .flac file
for i, item in enumerate(dataset.take(20)):
    # Extract the audio array and metadata
    audio_array = item["flac"]["array"]
    sr = item['flac']['sampling_rate']
    # Define the file path where the FLAC file will be saved
    file_path = os.path.join("x", f"audio_{i + 1}.flac")

    # Save the audio array as a FLAC file
    sf.write(file_path, audio_array, 22050)  # Use the appropriate sample rate (adjust if needed)

    print(f"Saved {file_path}")
