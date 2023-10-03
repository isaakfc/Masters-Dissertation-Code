import librosa
import psola
import soundfile as sf
import numpy as np
import scipy.signal as sig

# Load and ensure they have the same sample rate
# y1 is the style audio that is getting shifted
y1, sr1 = librosa.load('', mono=True)
y2, sr2 = librosa.load('', sr=sr1, mono=True)


# Limit to 15 seconds
LENGTH_IN_SAMPLES = int(sr1 * 15)
y1 = y1[:LENGTH_IN_SAMPLES]
y2 = y2[:LENGTH_IN_SAMPLES]

FRAME_LENGTH = 2048
HOP_LENGTH = FRAME_LENGTH // 4

F_MIN = librosa.note_to_hz('G1')
F_MAX = librosa.note_to_hz('A7')

# Extract pitch using pyin, lowest note will be lowest and highest notes of violin
pitch_map2, _, _ = librosa.pyin(y2, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, fmin=F_MIN, fmax=F_MAX)

# Smooth pitch
smoothed_target_pitch = sig.medfilt(pitch_map2, kernel_size=31)
# Remove the additional NaN values after median filtering.
smoothed_target_pitch[np.isnan(smoothed_target_pitch)] = pitch_map2[np.isnan(smoothed_target_pitch)]

print(smoothed_target_pitch)

y_shifted = psola.vocode(y1, sample_rate=int(sr1), target_pitch=smoothed_target_pitch, fmin=F_MIN, fmax=F_MAX)

sf.write('', y_shifted, sr1)