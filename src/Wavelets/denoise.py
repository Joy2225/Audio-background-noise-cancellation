
# import numpy as np
# import pywt
# import soundfile
# from tqdm import tqdm

# from lib.noiseProfiler import NoiseProfiler


# def mad(arr):

#     arr = np.ma.array(arr).compressed()
#     med = np.median(arr)
#     return np.median(np.abs(arr - med))


# class AudioDeNoise:


#     def __init__(self, inputFile):
#         self.__inputFile = inputFile
#         self.__noiseProfile = None

#     def deNoise(self, outputFile):

#         info = soundfile.info(self.__inputFile)  # getting info of the audio
#         rate = info.samplerate

#         with soundfile.SoundFile(outputFile, "w", samplerate=rate, channels=info.channels) as of:
#             for block in tqdm(soundfile.blocks(self.__inputFile, int(rate * info.duration * 0.10))):
#                 coefficients = pywt.wavedec(block, 'db4', mode='per', level=2)

#                 #  getting variance of the input signal
#                 sigma = mad(coefficients[- 1])

#                 # VISU Shrink thresholding by applying the universal threshold proposed by Donoho and Johnstone
#                 thresh = sigma * np.sqrt(2 * np.log(len(block)))

#                 # thresholding using the noise threshold generated
#                 coefficients[1:] = (pywt.threshold(i, value=thresh, mode='soft') for i in coefficients[1:])

#                 # getting the clean signal as in original form and writing to the file
#                 clean = pywt.waverec(coefficients, 'db4', mode='per')
#                 of.write(clean)

#     def generateNoiseProfile(self, noiseFile):

#         data, rate = soundfile.read(noiseFile)
#         self.__noiseProfile = NoiseProfiler(data)
#         noiseSignal = self.__noiseProfile.getNoiseDataPredicted()

#         soundfile.write(noiseFile, noiseSignal, rate)

#     def __del__(self):

#         del self.__noiseProfile


import numpy as np
import pywt
import soundfile
from tqdm import tqdm

from lib.noiseProfiler import NoiseProfiler


def mad(arr):
    """Median Absolute Deviation (MAD) for noise estimation."""
    arr = np.ma.array(arr).compressed()  # Remove masked values if any
    med = np.median(arr)
    return np.median(np.abs(arr - med))


class AudioDeNoise:

    def __init__(self, inputFile, wavelet='db4', level=2):
        self.__inputFile = inputFile
        self.__noiseProfile = None
        self.wavelet = wavelet
        self.level = level

    def deNoise(self, outputFile):
        """Performs noise reduction and saves the denoised audio to outputFile."""
        info = soundfile.info(self.__inputFile)  # Audio file info
        rate = info.samplerate

        # Open output file for writing
        with soundfile.SoundFile(outputFile, "w", samplerate=rate, channels=info.channels) as of:
            # Process file in chunks
            for block in tqdm(soundfile.blocks(self.__inputFile, int(rate * info.duration * 0.10))):
                # Wavelet decomposition
                coefficients = pywt.wavedec(block, self.wavelet, mode='per', level=self.level)

                # Noise estimation and threshold calculation
                sigma = mad(coefficients[-1])  # Noise standard deviation estimation
                thresh = sigma * np.sqrt(2 * np.log(len(block)))  # Universal threshold

                # Apply thresholding on wavelet coefficients
                coefficients[1:] = [pywt.threshold(i, value=thresh, mode='soft') for i in coefficients[1:]]

                # Reconstruct the clean signal
                clean = pywt.waverec(coefficients, self.wavelet, mode='per')
                of.write(clean)  # Write to output file

    def generateNoiseProfile(self, noiseFile):
        """Generates a noise profile from a noise-only audio sample."""
        data, rate = soundfile.read(noiseFile)
        self.__noiseProfile = NoiseProfiler(data)
        noiseSignal = self.__noiseProfile.getNoiseDataPredicted()

        soundfile.write(noiseFile, noiseSignal, rate)

    def __del__(self):
        """Destructor to clean up resources."""
        del self.__noiseProfile
