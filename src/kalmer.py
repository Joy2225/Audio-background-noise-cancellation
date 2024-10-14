import numpy as np
from scipy.signal import lfilter
import librosa
import soundfile as sf

def kalman_filter(data, R, Q):


    x_hat = np.zeros_like(data)  
    P = np.zeros_like(data)     

    # Kalman filter iteration
    for k in range(1, len(data)):
        # Prediction step
        x_hat_pred = x_hat[k - 1]
        P_pred = P[k - 1] + Q

        # Measurement update step
        K = P_pred / (P_pred + R)
        x_hat[k] = x_hat_pred + K * (data[k] - x_hat_pred)
        P[k] = (1 - K) * P_pred

    return x_hat


if __name__ == "__main__":
    # Load MP3 audio data
    audio_data, fs = librosa.load("./noisy_audio.mpeg")


    R = 0.01  
    Q = 0.001 


    filtered_audio = kalman_filter(audio_data, R, Q)

    sf.write("filtered_audio.wav", filtered_audio, fs)