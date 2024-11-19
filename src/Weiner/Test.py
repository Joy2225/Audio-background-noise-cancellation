#!/usr/bin/env python3
import Weiner as nr
import os
import time

start = time.time()
WAV_FILE = os.getcwd() + '/example/noisefunkguitare'
noise_begin, noise_end = 0, 1

noised_audio = nr.Wiener(WAV_FILE, noise_begin, noise_end)
end_inter = time.time()
noised_audio.wiener()
end1 = time.time()
print(f"Time taken: {end1-start} seconds for weiner")
end_pre = time.time()
noised_audio.wiener_two_step()
end2 = time.time()
print(f"Time taken: {end_inter-start+end2-end_pre} seconds for weiner two step")