from flask import request, jsonify, send_file, Flask
from graphing.graphs import plot_mfcc, plot_psd, plot_stft
import librosa
import numpy as np
from scipy.signal import welch
import io
import tempfile
import soundfile as sf
import zipfile

# from main import app
app = Flask(__name__)
@app.route("/classify_noise", methods=["POST"])
def classify_noise_endpoint():
    # Step 1: Validate file upload
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    audio_file = request.files["file"]

    # Step 2: Load the audio file using librosa
    audio, sr = librosa.load(audio_file, sr=None)
    audio = audio / np.max(np.abs(audio))

    # Step 3: Compute features for graph generation
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    stft_magnitude = np.abs(stft)
    stft_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
    freqs, psd = welch(audio, fs=sr, nperseg=1024)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Step 4: Generate graphs
    stft_buf = plot_stft(stft_db, sr)
    psd_buf = plot_psd(freqs, psd)
    mfcc_buf = plot_mfcc(mfccs, sr)


    # Log-log scale for linear fitting
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

    # Step 5: Create a zip file containing all the graphs
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("stft.png", stft_buf.getvalue())
        zf.writestr("psd.png", psd_buf.getvalue())
        zf.writestr("mfcc.png", mfcc_buf.getvalue())

    zip_buffer.seek(0)

    # Return the zip file
    return send_file(zip_buffer, mimetype="application/zip", as_attachment=True, download_name="plots.zip")

if __name__ == "__main__":
    app.run(debug=True,host='127.0.0.1',port=5000)