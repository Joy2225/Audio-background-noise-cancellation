from denoise import AudioDeNoise 

audioDenoiser = AudioDeNoise(inputFile="test_2.wav")
audioDenoiser.deNoise(outputFile="test_denoised.wav")
audioDenoiser.generateNoiseProfile(noiseFile="test_noise__lalaprofile.wav")