import streamlit as st
import requests
from zipfile import ZipFile
from io import BytesIO

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
            # Make a POST request to the backend
            response = requests.post(BACKEND_URL, files=files, timeout=3000)

            if response.status_code == 200:
                # Extract and display the graphs from the zip file
                with ZipFile(BytesIO(response.content)) as zf:
                    for filename in zf.namelist():
                        with zf.open(filename) as file:
                            st.image(file.read(), caption=filename.split(".")[0], use_container_width=True)

            else:
                st.error(f"Error: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {str(e)}")
