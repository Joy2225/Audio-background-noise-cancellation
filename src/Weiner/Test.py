#!/usr/bin/env python3
import Weiner as nr
import psutil
import os
import time

# Monitor process for memory and execution time
process = psutil.Process(os.getpid())
start = time.time()

# Define the path to the WAV file and noise intervals
WAV_FILE = os.path.join(os.getcwd(), 'example', 'noisefunkguitare')  # Path without .wav extension
noise_begin, noise_end = 0, 1  # Noise interval in seconds

# Initialize the Wiener noise reduction class
noised_audio = nr.Wiener(WAV_FILE, noise_begin, noise_end)

# Apply Wiener noise reduction
print("\nApplying Wiener noise reduction...")
noised_audio.wiener()

# Measure execution time
end = time.time()
print(f"Time taken: {end - start:.2f} seconds for Wiener filter")

# Display memory usage
print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
