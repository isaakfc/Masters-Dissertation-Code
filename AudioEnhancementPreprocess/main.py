import os
import librosa
import numpy as np
import soundfile as sf


class Loader:
    
    def __init__(self, sample_rate, mono):
        self.sample_rate = sample_rate
        self.mono = mono

    def load_wav(self, file_path):
        signal, sr = librosa.load(file_path,
                                  sr=self.sample_rate,
                                  mono=self.mono)
        assert sr == self.sample_rate, f"Sample rate is not consistent to global sr: {self.sample_rate}"
        return signal

    @staticmethod
    def extract_files(file_dir_path):
        passages = sorted([f for f in os.listdir(file_dir_path) if f.endswith('.wav')])
        return passages


class Saver:
    def __init__(self,
                 passages_save_dir_wav,
                 recreations_save_dir_wav,
                 passages_save_dir_spec,
                 recreations_save_dir_spec,
                 sr):
        self.passages_save_dir_wav = passages_save_dir_wav
        self.recreations_save_dir_wav = recreations_save_dir_wav
        self.passages_save_dir_spec = passages_save_dir_spec
        self.recreations_save_dir_spec = recreations_save_dir_spec
        self.sr = sr

    def save_audio_file(self, audio, filename):
        print(f"f{filename} saved")
        audio = audio.astype(np.float32)
        sf.write(filename, audio, self.sr)

    def save_audio_segments(self, passage_segments, recreations_segments, passage_file_path, recreation_file_path):
        for idx, (passage_segment, recreations_segment) in enumerate(zip(passage_segments, recreations_segments)):
            passage_segment_filename = os.path.join(self.passages_save_dir_wav,
                                                    f"{os.path.splitext(passage_file_path)[0]}_segment_{idx}.wav")
            recreation_segment_filename = os.path.join(self.recreations_save_dir_wav,
                                                       f"{os.path.splitext(recreation_file_path)[0]}_segment_{idx}.wav")
            self.save_audio_file(passage_segment, passage_segment_filename)
            self.save_audio_file(recreations_segment, recreation_segment_filename)

    def save_feature_segments(self, passage_segments, recreations_segments, passage_file_path, recreation_file_path):
        for idx, (passage_segment, recreations_segment) in enumerate(zip(passage_segments, recreations_segments)):
            passage_segment_filename = os.path.join(self.passages_save_dir_spec,
                                                    f"{os.path.splitext(passage_file_path)[0]}_segment_{idx}.npy")
            recreation_segment_filename = os.path.join(self.recreations_save_dir_spec,
                                                       f"{os.path.splitext(recreation_file_path)[0]}_segment_{idx}.npy")
            np.save(passage_segment_filename, passage_segment)
            print(f"f{passage_segment_filename} saved")
            np.save(recreation_segment_filename, recreations_segment)
            print(f"f{recreation_segment_filename} saved")


class PadderSegmenter:

    def __init__(self, segment_sample_length, mode="constant"):
        self.mode = mode
        self.segment_sample_length = segment_sample_length

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (0, num_missing_items),
                              mode=self.mode)
        return padded_array

    def segment_and_pad(self, audio):

        num_samples = len(audio)
        num_segments = int(np.ceil(num_samples / self.segment_sample_length))

        segmented_audio = np.zeros((num_segments, self.segment_sample_length))

        for i in range(num_segments):
            start = i * self.segment_sample_length
            end = min(start + self.segment_sample_length, num_samples)
            segmented_audio[i, :end - start] = audio[start:end]
        return segmented_audio


class LogSpectrogramExtractor:

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)[:-1]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        print(log_spectrogram.shape)
        return log_spectrogram

    def extract_segments(self, signals):
        num_segments, segment_length = signals.shape
        log_spectrograms = []

        for i in range(num_segments):
            segment = signals[i, :]
            log_spectrogram = self.extract(segment)
            log_spectrograms.append(log_spectrogram)

        return np.array(log_spectrograms)

class PreprocessingPipeline:
    def __init__(self):
        self.loader = None
        self.saver = None
        self.padder_segmenter = None
        self.file_dir_passages = None
        self.file_dir_recreations = None
        self.extractor = None
        self.passages = []
        self.recreations = []

    def process(self, file_dir_passages, file_dir_recreations):
        self.passages = loader.extract_files(file_dir_passages)
        self.recreations = loader.extract_files(file_dir_recreations)
        self.file_dir_passages = file_dir_passages
        self.file_dir_recreations = file_dir_recreations
        for passage, recreation in zip(self.passages, self.recreations):
            self.process_files(passage, recreation)

    def process_files(self, passage, recreation):
        passage_data = loader.load_wav(os.path.join(self.file_dir_passages, passage))
        recreation_data = loader.load_wav(os.path.join(self.file_dir_recreations, recreation))
        truncated_passage, truncated_recreation = self.truncate(passage_data, recreation_data)
        segmented_truncated_passage = self.padder_segmenter.segment_and_pad(truncated_passage)
        segmented_truncated_recreation = self.padder_segmenter.segment_and_pad(truncated_recreation)
        segmented_truncated_normalised_passage = self.normalise_audio(segmented_truncated_passage)
        segmented_truncated_normalised_recreation = self.normalise_audio(segmented_truncated_recreation)
        spectrograms_passages = self.extractor.extract_segments(segmented_truncated_normalised_passage)
        spectrograms_recreations = self.extractor.extract_segments(segmented_truncated_normalised_recreation)
        self.saver.save_audio_segments(segmented_truncated_normalised_passage,
                                       segmented_truncated_normalised_recreation,
                                       passage,
                                       recreation)
        self.saver.save_feature_segments(spectrograms_passages,
                                         spectrograms_recreations,
                                         passage,
                                         recreation)

    @staticmethod
    def truncate(passage, recreation):

        if len(passage) > len(recreation):
            passage = passage[:len(recreation)]
        elif len(recreation) > len(passage):
            recreation = recreation[:len(passage)]

        return passage, recreation

    def normalise_audio(self, array):
        max_abs_val = np.abs(array).max()
        norm_array = array / max_abs_val
        return norm_array


if __name__ == "__main__":
    SAMPLE_RATE = 22000
    SEGMENT_LENGTH = 10
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    MONO = True
    PASSAGES_PATH = ""
    RECREATIONS_PATH = ""
    PROCESSED_PASSAGES_PATH = ""
    PROCESSED_RECREATIONS_PATH = ""
    SPECTROGRAMS_PASSAGES_PATH = ""
    SPECTROGRAMS_RECREATIONS_PATH = ""

    SEGMENT_SAMPLE_LENGTH = SAMPLE_RATE * SEGMENT_LENGTH

    loader = Loader(SAMPLE_RATE, MONO)
    saver = Saver(PROCESSED_PASSAGES_PATH,
                  PROCESSED_RECREATIONS_PATH,
                  SPECTROGRAMS_PASSAGES_PATH,
                  SPECTROGRAMS_RECREATIONS_PATH,
                  SAMPLE_RATE)
    padder_segmenter = PadderSegmenter(SEGMENT_SAMPLE_LENGTH)
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder_segmenter = padder_segmenter
    preprocessing_pipeline.saver = saver
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.process(PASSAGES_PATH, RECREATIONS_PATH)


