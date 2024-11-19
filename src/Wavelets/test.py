from denoise import AudioDeNoise 
import time

start=time.time()
audioDenoiser = AudioDeNoise(inputFile="noisefunkguitare.wav")
audioDenoiser.deNoise(outputFile="test_denoised.wav")
end=time.time()
print(f"Time taken: {end-start} seconds")