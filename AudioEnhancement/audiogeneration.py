import soundfile as sf
import librosa
import numpy as np

def convert_spectrograms_to_audio(epoch, spectrograms, max_val, min_val, hop_length, frame_size, sr, num_files=5, file_prefix="", output_dir="GENERATEDAUDIO"):
    num_to_generate = min(num_files, spectrograms.shape[0])

    for i in range(num_to_generate):
        log_spectrogram = spectrograms[i, :, :, 0]  # Remove channel axis
        denormalised_log_spectrogram = log_spectrogram * (max_val - min_val) + min_val  # Denormalise
        # Add a nyquist bin
        denormalised_log_spectrogram = np.vstack([denormalised_log_spectrogram,
                                                  np.zeros(denormalised_log_spectrogram.shape[1])])
        spec = librosa.db_to_amplitude(denormalised_log_spectrogram)  # Convert back to linear from Db
        signal = librosa.griffinlim(spec,
                                    n_iter=32,
                                    hop_length=hop_length,  # Griffin Lim
                                    win_length=frame_size)
        audio = signal.astype(np.float32)
        sf.write(f'{output_dir}/{file_prefix}{i}_epoch_{epoch}.wav', audio, sr)