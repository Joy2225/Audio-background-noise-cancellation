from denoise import AudioDeNoise 
import time
import psutil
import os

process = psutil.Process(os.getpid())
start=time.time()
audioDenoiser = AudioDeNoise(inputFile="noisefunkguitare.wav")
audioDenoiser.deNoise(outputFile="test_denoised.wav")
end=time.time()
print(f"Time taken: {end-start} seconds")
print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")