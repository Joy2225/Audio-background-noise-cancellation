from flask import request, jsonify, send_file


from graphing.graphs import plot_mfcc, plot_psd, plot_stft, generate_audio_graph, compare_audios, save_plot_to_buffer



from methods.Kalman import kalman
from methods.NLM import nlm
from methods.Wavelets import denoise
from methods.Weiner import Weiner
from methods.SpectralGating import spectral_gating
from noise_classification.utils import (
    load_and_normalize_audio,
    compute_stft,
    compute_mfcc,
    compute_psd,
    classify_noise_by_slope,
)
from ..main import app
import librosa

@app.route("/classify_noise", methods=["POST"])
def classify_noise_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    noisy_audio = request.files["audio_file"]
    if noisy_audio.content_type != "audio/wav":
        return jsonify({"error": "Invalid file type, wav files only"}), 400

    normalized_audio, sr = load_and_normalize_audio(noisy_audio)
    print(noisy_audio)
    noisy_audio_array, sr = librosa.load(noisy_audio, sr=None)
    kalman_denoised_audio = kalman.kalman_filter_denoising(noisy_audio_array)
    nlm_denoised_audio = nlm.non_local_means_denoising(noisy_audio_array)
    wavelet_denoised_audio = denoise.denoise_audio(noisy_audio_array)
    spectral_gating_denoised_audio = spectral_gating.denoise_audio(noisy_audio_array)

    wavelet_denoised_audio = denoise.AudioDeNoise(noisy_audio)
    #  may god saved my soul from the the demons that haunt my dreams. I am a moron, whose brain is being eaten by worms. Behold this and weep ye mortals who dare to gaze upon this abomination.








    
    