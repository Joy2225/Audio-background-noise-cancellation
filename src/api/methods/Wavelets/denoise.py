import numpy as np
import pywt
import soundfile
from tqdm import tqdm
from scipy.signal import resample
from pystoi import stoi
from pesq import pesq
from mir_eval.separation import bss_eval_sources
import torch  # Importing PyTorch for tensor operations
from scipy.io import wavfile
import psutil
import os
import time
SAMPLE_RATE = 16000  # Global constant for target sample rate


import math

import numpy
import pywt


class WindowBundle:
    def __init__(self, data: numpy, id):
        self.id = id
        self.data = data
        self.rms = None
        self.waveletPacket = None
        self.noiseWindow = None
        self.denoisedData = []

        self.dbName = None
        self.wlevels = None

    def extractWaveletPacket(self, dbName, wlevels):
        if self.waveletPacket is not None:
            return self.waveletPacket

        self.dbName = dbName
        self.wlevels = wlevels
        self.waveletPacket = pywt.WaveletPacket(self.data, dbName, "symmetric", wlevels)

        return self.waveletPacket

    def getWaveletLeafData(self):
        windowWaveletData = list()
        leafNodes = [node.path for node in self.waveletPacket.get_level(self.wlevels, "freq")]

        for node in leafNodes:
            bandData = self.waveletPacket[node].data
            windowWaveletData.extend(bandData)

        return windowWaveletData

    def setDenoisedData(self, denoisedData):
        self.denoisedData = denoisedData

    def getDenoisedData(self):
        return self.denoisedData

    def setNoiseWindow(self, window):
        self.noiseWindow = window

    def isBelowThreshold(self, threshold):
        if self.getRMS() < threshold:
            return True

        return False

    def getData(self):
        return self.data

    def getRMS(self):
        if self.rms is not None:
            return self.rms

        squaredSum = numpy.sum(numpy.power(self.data, 2))
        self.rms = math.sqrt(squaredSum / len(self.data))

        return self.rms

    # gets the Mean Absolute
    def getMA(self):
        _sum = numpy.sum(numpy.abs(self.data))
        ma = _sum / len(self.data)

        return ma

    def getRMSasArray(self):
        return self.getRMS() * numpy.ones(len(self.data))

    @staticmethod
    def joinDenoisedData(windows: list):
        result = []
        for window in windows:
            result.extend(window.denoisedData)

        return result

    @staticmethod
    def joinData(windows: list):
        result = []
        for window in windows:
            result.extend(window.data)

        return result

    @staticmethod
    def joinNoiseData(windows: list):
        result = []
        for window in windows:
            result.extend(window.noiseWindow.data)

        return result


import pywt
import matplotlib.pyplot as plt


def waveletLeafData(waveletPacket: pywt.WaveletPacket):
    leafData = list()
    leafNodes = [node.path for node in waveletPacket.get_level(waveletPacket.maxlevel, "freq")]

    for node in leafNodes:
        bandData = waveletPacket[node].data
        leafData.extend(bandData)

    return leafData


def plotWavelets(wavelets: list):
    plt.figure()
    subplotIdx = 1
    leafNodes = [node.path for node in wavelets[0].get_level(wavelets[0].maxlevel, "natural", False)]

    for wavelet in wavelets:
        plt.subplot(len(wavelets), 1, subplotIdx)
        bandIdx = 0
        for node in leafNodes:
            bandData = wavelet[node].data
            bandLength = len(bandData)
            rangeArr = range(bandIdx * bandLength, (bandIdx + 1) * bandLength)
            plt.plot(rangeArr, bandData)
            bandIdx += 1
        subplotIdx += 1
    plt.show()


import math

import matplotlib.pyplot as plt
import numpy

# from lib import windowBundle, waveletHelper
# from lib.linkedList import LinkedList


class NoiseProfiler:
    """Basic denoiser wrapper for keeping store of the settings"""

    def __init__(self, x, timeWindow=0.1, sampleRate=44100, percentileLevel=95, wlevels=4, dbName="db8"):
        self.x = x
        self.timeWindow = timeWindow
        self.windowSamples = int(timeWindow * sampleRate)
        self.wlevels = wlevels
        self.dbName = dbName

        self.windows = list()
        self.sortedWindows = list()

        self.noiseWindows = None
        self.noiseLinked = LinkedList()
        self.signalWindows = None
        self.signalLinked = LinkedList()

        self.percentileLevel = percentileLevel
        self.noiseData = None
        self.noiseWavelets = list()
        self.threshold = None

        self.extractWindows()
        print("Noise profiler finished")

    def cleanUp(self):
        self.windows = None
        self.sortedWindows = None
        self.noiseData = None
        self.noiseLinked = None
        self.signalLinked = None
        self.signalWindows = None
        self.noiseWavelets = None

    def drawOriginalVsNoiseAndSingal(self):
        self.threshold = self.extractRMSthresholdFromWindows(self.percentileLevel)
        self.extractSignalAndNoiseWindows(self.threshold)

        noiseData = self.getDataOrZeroFromPartialWindows(self.windows, self.noiseWindows)
        signalData = self.getDataOrZeroFromPartialWindows(self.windows, self.signalWindows)

        rmsEnvelope = self.getWindowsRMSasEnvelope()

        plt.figure(1)
        plt.subplot(211)
        plt.plot(self.x)
        plt.subplot(211)
        plt.plot(rmsEnvelope)
        plt.plot(-1 * rmsEnvelope)
        plt.subplot(212)
        plt.plot(signalData)
        plt.plot(noiseData)
        plt.show()

    def __getNodesWindowData(self, nodes):
        data = []
        for node in nodes:
            window = node.data
            data.extend(window.data)

        return data

    def __getNodeCircularPrediction(self, node, n):
        prevNode = node.getPrevWithValidData()
        nextNode = node.getNextWithValidData()
        if prevNode is None:
            # work with current->future period of silence
            return self.__getFutureCircularNodes(nextNode, n)
        # working with the previous period of silence
        return self.__getPastCircularNodes(prevNode, n)

    def __getFutureCircularNodes(self, initialNode, n):
        ret = []
        count = 0
        current = initialNode
        while True:
            ret.append(current)
            count += 1
            if count == n:
                return ret

            if current.next and current.next.data:
                current = current.next
            else:
                current = initialNode

    def __getPastCircularNodes(self, initialNode, n):
        ret = []
        count = 0
        current = initialNode
        while True:
            ret.append(current)
            count += 1
            if count == n:
                return ret

            if current.prev and current.prev.data:
                current = current.prev
            else:
                current = initialNode

    def getNoiseDataPredicted(self):
        self.threshold = self.extractRMSthresholdFromWindows(self.percentileLevel)
        self.extractSignalAndNoiseWindows(self.threshold)

        noiseDataPredicted = []

        consecutiveEmptyNodes = 0
        lastValidNode = None
        for node in self.noiseLinked.getAsList():
            if node.data is None:
                consecutiveEmptyNodes += 1
            else:
                lastValidNode = node

                if consecutiveEmptyNodes != 0:
                    predictedNodes = self.__getNodeCircularPrediction(node, consecutiveEmptyNodes)
                    noiseDataPredicted.extend(self.__getNodesWindowData(predictedNodes))
                    consecutiveEmptyNodes = 0

                window = node.data
                noiseDataPredicted.extend(window.data)

        # in case we had empty data on the end
        if consecutiveEmptyNodes != 0:
            predictedNodes = self.__getNodeCircularPrediction(lastValidNode, consecutiveEmptyNodes)
            noiseDataPredicted.extend(self.__getNodesWindowData(predictedNodes))

        self.cleanUp()
        return noiseDataPredicted

    def extractRMSthresholdFromWindows(self, percentileLevel):
        if self.threshold is not None:
            return self.threshold

        sortedWindows = sorted(self.windows, key=lambda x: x.getRMS(), reverse=True)
        # now the are arranged with the max DESC
        nWindows = len(sortedWindows)
        thresholdIndex = math.floor(percentileLevel / 100 * nWindows)
        self.threshold = sortedWindows[thresholdIndex].getRMS()

        return self.threshold

    def getWindowsRMSasEnvelope(self):
        envelope = numpy.array([])
        """
        :type self.windows: list[windowBundle]
        """
        for window in self.windows:
            windowEnvelope = window.getRMS() * numpy.ones(len(window.data))
            envelope = numpy.concatenate([envelope, windowEnvelope])

        return envelope

    def extractWindows(self):
        xLength = len(self.x)
        nWindows = math.ceil(xLength / self.windowSamples)
        lastWindowPaddingSamples = xLength - nWindows * self.windowSamples
        for i in range(0, nWindows):
            windowBeginning = i * self.windowSamples
            windowEnd = windowBeginning + self.windowSamples
            windowData = self.x[windowBeginning:windowEnd]
            # checking wether we need to pad the last band
            if i == nWindows - 1 and windowEnd - windowBeginning < self.windowSamples:
                paddingLength = windowEnd - windowBeginning - self.windowSamples
                paddingArray = numpy.zeros(paddingLength)
                windowData = numpy.concatenate(windowData, paddingArray)
            window = WindowBundle(windowData, i)
            self.windows.append(window)

    def extractSignalAndNoiseWindows(self, rmsThreshold):
        if self.noiseWindows is not None and self.signalWindows is not None:
            return

        self.noiseWindows = list()
        self.signalWindows = list()
        for window in self.windows:
            # giving a +5% grace on the rms threshold comparison
            if window.getRMS() < (rmsThreshold + 0.05 * rmsThreshold):
                self.noiseWindows.append(window)
                self.noiseLinked.append(window)
                self.signalLinked.append(None)
            else:
                self.signalWindows.append(window)
                self.signalLinked.append(window)
                self.noiseLinked.append(None)

    def getDataOrZeroFromPartialWindows(self, allWindows, partialWindows):
        data = []
        idx = 0
        for window in allWindows:
            if idx < len(partialWindows) and window == partialWindows[idx]:
                data.extend(window.data)
                idx += 1
            else:
                data.extend(numpy.zeros(self.windowSamples))

        return data

    def extractWavelets(self):
        for window in self.windows:
            window.extractWaveletPacket(self.dbName, self.wlevels)

    def plotWavelets(self):
        wtBandsLength = 0
        for window in self.windows:
            windowWaveletData = list()

            windowDataLength = 0
            data = window.getData()
            wt = window.extractWaveletPacket(self.dbName, self.wlevels)
            leafNodes = [node.path for node in wt.get_level(self.wlevels, "freq")]

            for node in leafNodes:
                bandData = wt[node].data
                windowWaveletData.extend(bandData)
                wtBandsLength += len(bandData)
                windowDataLength += len(bandData)

            print("window # " + str(window.id) + " of " + str(len(self.windows)))
            plt.figure(window.id)
            plt.subplot(211)
            plt.plot(window.data)
            plt.subplot(212)
            plt.plot(waveletLeafData(window.waveletPacket))
            plt.show()


class Node:
    def __init__(self, data, prev=None, next=None):
        self.data = data
        self.prev = prev
        self.next = next

    def getNextWithValidData(self):
        current = self.next
        while current is not None:
            if current.data is not None:
                return current
            current = current.next

        return None

    def getPrevWithValidData(self):
        current = self.prev
        while current is not None:
            if current.data is not None:
                return current
            current = current.prev

        return None


class LinkedList:
    def __init__(self):
        self.first = None  # head
        self.last = None  # tail
        self.__list = None

    def append(self, data):
        new_node = Node(data, None, None)
        if self.first is None:
            self.first = self.last = new_node
            self.__list = list()
        else:
            new_node.prev = self.last
            new_node.next = None
            self.last.next = new_node
            self.last = new_node

        self.__list.append(new_node)

    def getAsList(self):
        ret = list()
        current = self.first
        while current is not None:
            ret.append(current)
            current = current.next

        return ret


import numpy as np
import pywt
import soundfile
from tqdm import tqdm

# from lib.noiseProfiler import NoiseProfiler


def mad(arr):
    """Median Absolute Deviation (MAD) for noise estimation."""
    arr = np.ma.array(arr).compressed()  # Remove masked values if any
    med = np.median(arr)
    return np.median(np.abs(arr - med))


# def si_snr(estimate, reference, epsilon=1e-8):
#     """
#     Scale-Invariant Signal-to-Noise Ratio (SI-SNR) using PyTorch.
#     """
#     estimate = estimate - estimate.mean()
#     reference = reference - reference.mean()
#     reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
#     mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
#     scale = mix_pow / (reference_pow + epsilon)

#     reference = scale * reference
#     error = estimate - reference

#     reference_pow = reference.pow(2)
#     error_pow = error.pow(2)

#     reference_pow = reference_pow.mean(axis=1)
#     error_pow = error_pow.mean(axis=1)

#     si_snr = 10 * torch.log10(reference_pow + epsilon) - 10 * torch.log10(error_pow + epsilon)
#     return si_snr.item()

# # Evaluation function integrating SI-SNR, SDR, PESQ, and STOI
# def evaluate(estimate, reference):
#     """
#     Evaluates the enhanced signal using SI-SNR, SDR, PESQ, and STOI metrics.
#     """
#     si_snr_score = si_snr(estimate, reference)
#     (
#         sdr,
#         _,
#         _,
#         _,
#     ) = mir_eval.separation.bss_eval_sources(reference.numpy(), estimate.numpy(), False)
#     pesq_score = pesq(SAMPLE_RATE, estimate[0].numpy(), reference[0].numpy(), "wb")
#     stoi_score = stoi(reference[0].numpy(), estimate[0].numpy(), SAMPLE_RATE, extended=False)

#     return si_snr_score, sdr[0], pesq_score, stoi_score


def compute_si_snr(reference, enhanced):
    reference = reference - np.mean(reference)
    enhanced = enhanced - np.mean(enhanced)
    dot_product = np.dot(reference, enhanced)
    projection = dot_product * reference / np.dot(reference, reference)
    noise = enhanced - projection
    si_snr = 10 * np.log10(np.dot(projection, projection) / np.dot(noise, noise))
    return si_snr


# Metrics calculation
def metrics(reference, enhanced, rate):
    print("\nCalculating Metrics...")

    # Convert signals to float for processing
    reference_signal = reference.astype(float)
    enhanced_signal = enhanced.astype(float)

    # Ensure mono signals
    if reference_signal.ndim > 1:
        reference_signal = reference_signal.mean(axis=1)  # Convert stereo to mono
    if enhanced_signal.ndim > 1:
        enhanced_signal = enhanced_signal.mean(axis=1)

    # Resampling for PESQ compatibility
    target_rate = 16000  # Target sample rate for PESQ compatibility
    if rate != target_rate:
        print(f"Resampling from {rate} Hz to {target_rate} Hz for PESQ compatibility...")
        reference_signal_1 = resample(reference_signal, int(len(reference_signal) * target_rate / rate))
        enhanced_signal_1 = resample(enhanced_signal, int(len(enhanced_signal) * target_rate / rate))
        rate = target_rate

    # Ensure the signals are 1D
    reference_signal = np.squeeze(reference_signal)
    enhanced_signal = np.squeeze(enhanced_signal)

    if reference_signal.ndim != 1 or enhanced_signal.ndim != 1:
        raise ValueError("Signals must be 1D arrays for PESQ.")

    # PESQ Score
    pesq_score = pesq(rate, reference_signal_1, enhanced_signal_1, "wb")

    # SI-SNR
    si_snr_score = compute_si_snr(reference_signal, enhanced_signal)

    # STOI
    stoi_score = stoi(reference_signal, enhanced_signal, rate)

    # SDR
    sdr, sir, sar, perm = bss_eval_sources(reference_signal[np.newaxis, :], enhanced_signal[np.newaxis, :])

    # Display Results
    print(f"PESQ Score: {pesq_score}")
    print(f"SI-SNR Score: {si_snr_score:.2f} dB")
    print(f"STOI Score: {stoi_score:.2f}")
    print(f"SDR Score: {sdr[0]:.2f} dB")


class AudioDeNoise:
    # def __init__(self, inputFile, wavelet="db4", level=2):
    #     self.__inputFile = inputFile
    #     self.target_rate = SAMPLE_RATE
    #     self.__noiseProfile = None
    #     self.wavelet = wavelet
    #     self.level = level

    def __init__(self, audio_data, sr, wavelet="db4", level=2):
        self.audio_data = audio_data
        self.sr = sr
        self.__noiseProfile = None
        self.wavelet = wavelet
        self.level = level

    # def deNoise(self, outputFile):
    #     """Performs noise reduction and saves the denoised audio to outputFile."""
    #     info = soundfile.info(self.__inputFile)  # Audio file info
    #     rate = info.samplerate

    #     # Open output file for writing
    #     with soundfile.SoundFile(outputFile, "w", samplerate=rate, channels=info.channels) as of:
    #         # Process file in chunks
    #         for block in tqdm(soundfile.blocks(self.__inputFile, int(rate * info.duration * 0.10))):
    #             # Wavelet decomposition
    #             coefficients = pywt.wavedec(block, self.wavelet, mode="per", level=self.level)

    #             # Noise estimation and threshold calculation
    #             sigma = mad(coefficients[-1])  # Noise standard deviation estimation
    #             thresh = sigma * np.sqrt(2 * np.log(len(block)))  # Universal threshold

    #             # Apply thresholding on wavelet coefficients
    #             coefficients[1:] = [pywt.threshold(i, value=thresh, mode="soft") for i in coefficients[1:]]

    #             # Reconstruct the clean signal
    #             clean = pywt.waverec(coefficients, self.wavelet, mode="per")
    #             of.write(clean)  # Write to output file

    #             # Read the denoised signal
    #     enhanced_signal, _ = soundfile.read(outputFile)

    #     # data, _ = soundfile.read(self.__inputFile)
    #     rate, data = wavfile.read(self.__inputFile)
    #     # metrics(data, enhanced_signal, rate)

    #     return enhanced_signal

    def deNoise(self):
        process = psutil.Process(os.getpid())
        start = time.time()
        """Performs noise reduction and returns the denoised audio as a NumPy array."""
        # Wavelet decomposition
        coefficients = pywt.wavedec(self.audio_data, self.wavelet, mode="per", level=self.level)

        # Noise estimation and threshold calculation
        sigma = mad(coefficients[-1])  # Noise standard deviation estimation
        thresh = sigma * np.sqrt(2 * np.log(len(self.audio_data)))  # Universal threshold

        # Apply thresholding on wavelet coefficients
        coefficients[1:] = [pywt.threshold(i, value=thresh, mode="soft") for i in coefficients[1:]]

        # Reconstruct the clean signal
        clean_audio = pywt.waverec(coefficients, self.wavelet, mode="per")
        end = time.time()
        return clean_audio.astype(np.float32), {"Execution Time": end - start, "Memory Usage": process.memory_info().rss / 1024 ** 2}
        # # Resample if needed for PESQ compatibility
        # if rate != self.target_rate:
        #     print(f"Resampling from {rate} Hz to {self.target_rate} Hz for PESQ compatibility...")
        #     data = resample(data, int(len(data) * self.target_rate / rate))
        #     rate = self.target_rate
        # else:
        #     rate = rate
        # # Convert to mono if needed
        # if len(enhanced_signal.shape) > 1:
        #     enhanced_signal = np.mean(enhanced_signal, axis=1)

        # # Align lengths of reference and enhanced signals
        # min_length = min(len(data), len(enhanced_signal))
        # reference_signal = data[:min_length].astype(float)
        # enhanced_signal = enhanced_signal[:min_length].astype(float)

        # # Convert signals to PyTorch tensors
        # reference_tensor = torch.tensor(reference_signal).unsqueeze(0)  # Add batch dimension
        # enhanced_tensor = torch.tensor(enhanced_signal).unsqueeze(0)  # Add batch dimension

        # # Evaluate metrics
        # print("\nCalculating Metrics...")
        # si_snr_score, sdr_score, pesq_score, stoi_score = evaluate(enhanced_tensor, reference_tensor)

        # # Display Results
        # print(f"PESQ Score: {pesq_score}")
        # print(f"SI-SNR Score: {si_snr_score:.2f} dB")
        # print(f"STOI Score: {stoi_score:.2f}")
        # print(f"SDR Score: {sdr_score:.2f} dB")

    def __del__(self):
        """Destructor to clean up resources."""
        del self.__noiseProfile
