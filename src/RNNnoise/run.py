import subprocess
import os
from pydub import AudioSegment


def process_audio(input_pcm_path, output_wav_path, executable_path):
    """
    Runs an executable to process a PCM file, converts the output PCM to WAV, and deletes the PCM files.

    Args:
        input_pcm_path (str): Path to the input PCM file.
        output_wav_path (str): Path to the output WAV file.
        executable_path (str): Path to the executable.

    Returns:
        str: Path to the generated WAV file.
    """
    try:
        # Generate the intermediate PCM output path
        output_pcm_path = f"{output_wav_path}.pcm"

        # Run the executable with the input PCM and output PCM paths
        subprocess.run([executable_path, input_pcm_path, output_pcm_path], check=True)

        # Convert PCM to WAV
        pcm_audio = AudioSegment.from_file(output_pcm_path, format="raw", frame_rate=16000, channels=1, sample_width=2)
        pcm_audio.export(output_wav_path, format="wav")

        # Remove the intermediate PCM files
        os.remove(input_pcm_path)
        os.remove(output_pcm_path)

        return output_wav_path

    except subprocess.CalledProcessError as e:
        print(f"Error running executable: {e}")
        raise
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


EXECUTABLE_PATH = "/home/dhruv/Programming/CollegeProjects/Sem5/DAA/Audio-background-noise-cancellation/src/RNNnoise/rnnoise/examples/rnnoise_demo"