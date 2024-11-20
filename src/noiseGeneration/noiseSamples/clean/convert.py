import os
import subprocess


def convert_flac_to_wav_bulk(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through each file in the input directory
    for filename in os.listdir(input_directory):
        # Check if the file is a .flac file
        if filename.endswith(".flac"):
            input_file = os.path.join(input_directory, filename)
            output_file = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.wav")

            # Use FFmpeg to convert the FLAC file to WAV
            command = ["ffmpeg", "-i", input_file, output_file]
            subprocess.run(command, check=True)
            print(f"Converted {input_file} to {output_file}")


# Example usage
input_dir = "flac"  # Input directory containing FLAC files
output_dir = "wav_files"  # Output directory for WAV files

convert_flac_to_wav_bulk(input_dir, output_dir)
