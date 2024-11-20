from flask import request, jsonify, send_file, Flask
from graphing.graphs import plot_mfcc, plot_psd, plot_stft, plot_freq
import librosa
import numpy as np
from scipy.signal import welch
import io
import tempfile
import soundfile as sf
import zipfile
from methods.Kalman.kalman import kalman_filter_denoising
from scipy.signal import resample
from pesq import pesq
from pystoi import stoi
from mir_eval.separation import bss_eval_sources
import pandas as pd
from methods.NLM.nlm import non_local_means_denoising
from methods.Spectral_Gating.spectral_gating import spectral_denoising
from scipy.io import wavfile
from methods.Wavelets.denoise import AudioDeNoise

# from main import app
app = Flask(__name__)

# SI-SNR computation
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
    pesq_score = pesq(rate, reference_signal_1, enhanced_signal_1, 'wb')

    # SI-SNR
    si_snr_score = abs(compute_si_snr(reference_signal, enhanced_signal))

    # STOI
    stoi_score = stoi(reference_signal, enhanced_signal, rate)

    # SDR
    sdr, sir, sar, perm = bss_eval_sources(reference_signal[np.newaxis, :], enhanced_signal[np.newaxis, :])

    # Display Results
    # print(f"PESQ Score: {pesq_score}")
    # print(f"SI-SNR Score: {si_snr_score:.2f} dB")
    # print(f"STOI Score: {stoi_score:.2f}")
    # print(f"SDR Score: {sdr[0]:.2f} dB")
    return {"PESQ Score": pesq_score, "SI-SNR Score": si_snr_score, "STOI Score": stoi_score, "SDR Score": sdr[0]}




def plot_graph(audio,sr,filee):
    # audio, sr = librosa.load(audio_file, sr=None)
    audio = audio / np.max(np.abs(audio))

    # Step 3: Compute features for graph generation
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    stft_magnitude = np.abs(stft)
    stft_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
    freqs, psd = welch(audio, fs=sr, nperseg=1024)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    stft_buf = plot_stft(stft_db, sr)
    psd_buf = plot_psd(freqs, psd)
    mfcc_buf = plot_mfcc(mfccs, sr)
    graph_buf = plot_freq(audio, sr, filee)

    log_freqs = np.log10(freqs[1:])  # Skip 0 Hz to avoid log(0)
    log_psd = np.log10(psd[1:])

    # Linear regression to find the slope
    slope, _ = np.polyfit(log_freqs, log_psd, 1)
    
    # Classification based on slope
    if -0.1 <= slope <= 0.1:
        noise_type = "White Noise"
    elif -1.2 < slope <= -0.8:
        noise_type = "Pink Noise"
    elif slope < -1.2:
        noise_type = "Brownian Noise"
    elif 0.2 <= slope < 1.0:
        noise_type = "Blue Noise"
    elif slope >= 1.0:
        noise_type = "Violet Noise"
    elif 0.1 < slope < 0.2:
        noise_type = "Grey Noise"
    elif -0.2 <= slope < -0.1:
        noise_type = "Velvet Noise" 
    else:
        noise_type = "Unknown"
        
    return stft_buf, psd_buf, mfcc_buf, graph_buf, noise_type

def save_audio_to_bytes(audio, sr):
    # Create a bytes buffer
    with io.BytesIO() as audio_buffer:
        # Write audio data as WAV file to the buffer
        sf.write(audio_buffer, audio, sr, format='wav')
        # Get the value of the buffer
        audio_bytes = audio_buffer.getvalue()
    return audio_bytes

def save_metrics_to_csv(metrics_dict, noise_type, filename):
    metrics_dict["Noise Type"] = [noise_type]
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_buf = io.BytesIO()
    metrics_df.to_csv(metrics_buf, index=False)
    metrics_buf.seek(0)
    return metrics_buf  

@app.route("/classify_noise", methods=["POST"])
def classify_noise_endpoint():
    # Step 1: Validate file upload
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    audio_file = request.files["file"]
    audio, sr = librosa.load(audio_file, sr=None)
    # rate, data = wavfile.read(audio_file)
    stft_buf, psd_buf, mfcc_buf, graph_buf, noise_type = plot_graph(audio, sr,"original")
    try:
        
        denoised_audio = kalman_filter_denoising(audio)
        stft_buf_denoised_kl, psd_buf_denoised_kl, mfcc_buf_denoised_kl, freq_kl, _ = plot_graph(denoised_audio, sr, "Kalman")
        metrics_denoised = metrics(audio, denoised_audio, sr)
        metrics_denoised_buf_kl = save_metrics_to_csv(metrics_denoised, noise_type, "metrics_denoised_kl.csv")
        denoised_audio_bytes_kl = save_audio_to_bytes(denoised_audio, sr)
        

        denoised_audio = non_local_means_denoising(audio)
        stft_buf_denoised_nlm, psd_buf_denoised_nlm, mfcc_buf_denoised_nlm, freq_nlm, _ = plot_graph(denoised_audio, sr, "NLM")
        metrics_denoised = metrics(audio, denoised_audio, sr)
        metrics_denoised_buf_nlm = save_metrics_to_csv(metrics_denoised, noise_type, "metrics_denoised_nlm.csv")
        denoised_audio_bytes_nlm = save_audio_to_bytes(denoised_audio, sr)


        denoised_audio = spectral_denoising(audio, sr)
        stft_buf_denoised_sg, psd_buf_denoised_sg, mfcc_buf_denoised_sg, freq_sg, _ = plot_graph(denoised_audio, sr, "Spectral Gating")        
        metrics_denoised = metrics(audio, denoised_audio, sr)
        metrics_denoised_buf_sg = save_metrics_to_csv(metrics_denoised, noise_type, "metrics_denoised_sg.csv")
        denoised_audio_bytes_sg = save_audio_to_bytes(denoised_audio, sr)

        denoiser = AudioDeNoise(audio_data=audio, sr=sr)
        denoised_audio = denoiser.deNoise()
        stft_buf_denoised_wv, psd_buf_denoised_wv, mfcc_buf_denoised_wv, freq_wv, _ = plot_graph(denoised_audio, sr, "Wavelet")
        metrics_denoised = metrics(audio, denoised_audio, sr)
        metrics_denoised_buf_wv = save_metrics_to_csv(metrics_denoised, noise_type, "metrics_denoised_wv.csv")
        denoised_audio_bytes_wv = save_audio_to_bytes(denoised_audio, sr)




        
        

        # Step 5: Create a zip file containing all the graphs
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("metrics_denoised_kl.csv", metrics_denoised_buf_kl.getvalue())
            zf.writestr("stft.png", stft_buf.getvalue())
            zf.writestr("psd.png", psd_buf.getvalue())
            zf.writestr("mfcc.png", mfcc_buf.getvalue())
            zf.writestr("freq_graph.png", graph_buf.getvalue())

            zf.writestr("stft_denoised_kl.png", stft_buf_denoised_kl.getvalue())
            zf.writestr("psd_denoised_kl.png", psd_buf_denoised_kl.getvalue())
            zf.writestr("mfcc_denoised_kl.png", mfcc_buf_denoised_kl.getvalue())
            zf.writestr("freq_kl.png", freq_kl.getvalue())
            zf.writestr("denoised_audio_kl.wav", denoised_audio_bytes_kl)

            zf.writestr("metrics_denoised_nlm.csv", metrics_denoised_buf_nlm.getvalue())
            zf.writestr("stft_denoised_nlm.png", stft_buf_denoised_nlm.getvalue())
            zf.writestr("psd_denoised_nlm.png", psd_buf_denoised_nlm.getvalue())
            zf.writestr("mfcc_denoised_nlm.png", mfcc_buf_denoised_nlm.getvalue())
            zf.writestr("freq_nlm.png", freq_nlm.getvalue())
            zf.writestr("denoised_audio_nlm.wav", denoised_audio_bytes_nlm)

            zf.writestr("metrics_denoised_sg.csv", metrics_denoised_buf_sg.getvalue())
            zf.writestr("stft_denoised_sg.png", stft_buf_denoised_sg.getvalue())
            zf.writestr("psd_denoised_sg.png", psd_buf_denoised_sg.getvalue())
            zf.writestr("mfcc_denoised_sg.png", mfcc_buf_denoised_sg.getvalue())
            zf.writestr("freq_sg.png", freq_sg.getvalue())
            zf.writestr("denoised_audio_sg.wav", denoised_audio_bytes_sg)

            zf.writestr("metrics_denoised_wv.csv", metrics_denoised_buf_wv.getvalue())
            zf.writestr("stft_denoised_wv.png", stft_buf_denoised_wv.getvalue())
            zf.writestr("psd_denoised_wv.png", psd_buf_denoised_wv.getvalue())
            zf.writestr("mfcc_denoised_wv.png", mfcc_buf_denoised_wv.getvalue())
            zf.writestr("freq_wv.png", freq_wv.getvalue())
            zf.writestr("denoised_audio_wv.wav", denoised_audio_bytes_wv)


        zip_buffer.seek(0)

        # Return the zip file
        return send_file(zip_buffer, mimetype="application/zip", as_attachment=True, download_name="plots.zip")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True,host='127.0.0.1',port=5000)