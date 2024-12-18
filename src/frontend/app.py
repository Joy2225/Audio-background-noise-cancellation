# import streamlit as st
# import requests
# from zipfile import ZipFile
# from io import BytesIO

# BACKEND_URL = "http://127.0.0.1:5000/classify_noise"

# st.title("Audio File Analysis")

# # File Upload
# uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
# if uploaded_file:
#     st.audio(uploaded_file, format="audio/wav")

# if st.button("Submit"):
#     if uploaded_file:
#         files = {"file": uploaded_file}
#         try:
#             # Make a POST request to the backend
#             response = requests.post(BACKEND_URL, files=files, timeout=3000)

#             if response.status_code == 200:
#                 # Extract and display the graphs from the zip file
#                 with ZipFile(BytesIO(response.content)) as zf:
#                     for filename in zf.namelist():
#                         with zf.open(filename) as file:
#                             st.image(file.read(), caption=filename.split(".")[0], use_container_width=True)

#             else:
#                 st.error(f"Error: {response.text}")
#         except requests.exceptions.RequestException as e:
#             st.error(f"Error: {str(e)}")




import streamlit as st
import requests
from zipfile import ZipFile
from io import BytesIO
import pandas as pd

BACKEND_URL = "http://127.0.0.1:5000/classify_noise"

st.title("Audio File Analysis")

# File Upload
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

if st.button("Submit"):
    if uploaded_file:
        files = {"file": uploaded_file}
        try:
            response = requests.post(BACKEND_URL, files=files, timeout=3000)
            if response.status_code == 200:
                with ZipFile(BytesIO(response.content)) as zf:
                    # Extract metrics and noise type
                    metrics_data = pd.read_csv(zf.open("metrics_denoised_kl.csv"))
                    noise_type = metrics_data.iloc[0]["Noise Type"]

                    

                    # Display original audio details
                    st.header("Original Audio")
                    st.subheader("Noise Type:")
                    st.write(noise_type)  # Display noise type
                    # st.subheader("Metrics:")
                    # st.table(metrics_data)

                    # Display original graphs
                    st.subheader("Original Graphs")
                    for filename in ["stft.png", "psd.png", "mfcc.png", "waveform_graph.png"]:
                        st.image(zf.read(filename), caption=filename.split(".")[0], use_container_width=True)

                    # Display kalman denoised audio details
                    st.header("Kalman Denoised Audio")
                    with zf.open("denoised_audio_kl.wav") as audio_file:
                        audio_data = audio_file.read()
                        st.audio(data=audio_data, format="audio/wav")
                    st.subheader("Metrics:")
                    st.table(pd.read_csv(zf.open("metrics_denoised_kl.csv")))

                    # Display denoised graphs
                    st.subheader("Kalman Denoised Graphs")
                    for filename in ["stft_denoised_kl.png", "psd_denoised_kl.png", "mfcc_denoised_kl.png", "waveform_kl.png"]:
                        st.image(zf.read(filename), caption=filename.split(".")[0], use_container_width=True)
                    
                    # Display NLM denoised audio details
                    st.header("NLM Denoised Audio")
                    with zf.open("denoised_audio_nlm.wav") as audio_file:
                        audio_data = audio_file.read()
                        st.audio(data=audio_data, format="audio/wav")
                    st.subheader("Metrics:")
                    st.table(pd.read_csv(zf.open("metrics_denoised_nlm.csv")))

                    # Display denoised graphs
                    st.subheader("NLM Denoised Graphs")
                    for filename in ["stft_denoised_nlm.png", "psd_denoised_nlm.png", "mfcc_denoised_nlm.png", "waveform_nlm.png"]:
                        st.image(zf.read(filename), caption=filename.split(".")[0], use_container_width=True)
                    
                    # Display SG denoised audio details
                    st.header("Spectral Gating Denoised Audio")
                    with zf.open("denoised_audio_sg.wav") as audio_file:
                        audio_data = audio_file.read()
                        st.audio(data=audio_data, format="audio/wav")
                    st.subheader("Metrics:")
                    st.table(pd.read_csv(zf.open("metrics_denoised_sg.csv")))

                    # Display denoised graphs
                    st.subheader("Spectral Gating Denoised Graphs")
                    for filename in ["stft_denoised_sg.png", "psd_denoised_sg.png", "mfcc_denoised_sg.png", "waveform_sg.png"]:
                        st.image(zf.read(filename), caption=filename.split(".")[0], use_container_width=True)

                    # Display Wavelet denoised audio details
                    st.header("Wavelet Denoised Audio")
                    with zf.open("denoised_audio_wv.wav") as audio_file:
                        audio_data = audio_file.read()
                        st.audio(data=audio_data, format="audio/wav")
                    st.subheader("Metrics:")
                    st.table(pd.read_csv(zf.open("metrics_denoised_wv.csv")))

                    # Display denoised graphs
                    st.subheader("Wavelet Denoised Graphs")
                    for filename in ["stft_denoised_wv.png", "psd_denoised_wv.png", "mfcc_denoised_wv.png", "waveform_wv.png"]:
                        st.image(zf.read(filename), caption=filename.split(".")[0], use_container_width=True)

                    #Display weiner denoised audio details
                    st.header("Weiner Denoised Audio")
                    with zf.open("denoised_audio_weiner.wav") as audio_file:
                        audio_data = audio_file.read()
                        st.audio(data=audio_data, format="audio/wav")
                    st.subheader("Metrics:")
                    st.table(pd.read_csv(zf.open("metrics_denoised_weiner.csv")))

                    # Display denoised graphs
                    st.subheader("weiner Denoised Graphs")
                    for filename in ["stft_denoised_weiner.png", "psd_denoised_weiner.png", "mfcc_denoised_weiner.png", "waveform_weiner.png"]:
                        st.image(zf.read(filename), caption=filename.split(".")[0], use_container_width=True)

            else:
                st.error(f"Error: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {str(e)}")
