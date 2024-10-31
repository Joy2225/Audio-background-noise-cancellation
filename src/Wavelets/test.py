from denoise import AudioDeNoise 

audioDenoiser = AudioDeNoise(inputFile="test_3.wav")
audioDenoiser.deNoise(outputFile="test_denoised.wav")
audioDenoiser.generateNoiseProfile(noiseFile="test_noise_profile.wav")