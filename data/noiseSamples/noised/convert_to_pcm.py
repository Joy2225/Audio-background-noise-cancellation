import os
from pydub import AudioSegment


def convert_to_pcm(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load the audio file
            audio = AudioSegment.from_file(input_path)

            # Convert to mono if not already
            if audio.channels > 1:
                audio = audio.set_channels(1)

            # Set sample width to 2 bytes (16-bit PCM)
            audio = audio.set_sample_width(2)

            # Export the file
            audio.export(output_path, format="wav")
            print(f"Converted: {filename} -> {output_path}")


input_directory = "/home/dhruv/Programming/CollegeProjects/Sem5/DAA/Audio-background-noise-cancellation/src/noiseGeneration/noiseSamples/noised/wav"  # Replace with your input directory
output_directory = "/home/dhruv/Programming/CollegeProjects/Sem5/DAA/Audio-background-noise-cancellation/src/noiseGeneration/noiseSamples/noised/pcm"  # Replace with your output directory

convert_to_pcm(input_directory, output_directory)
