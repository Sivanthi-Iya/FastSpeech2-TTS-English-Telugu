import os
import torch
import torchaudio
import torchaudio.transforms as T

class Preprocessor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, win_length=1024, hop_length=256, n_mels=80, power=1.0)

    def process(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = T.Resample(sr, self.sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        mel = torch.log(torch.clamp(self.mel_transform(waveform), min=1e-5)).squeeze(0).transpose(0, 1)
        energy = torch.norm(mel, dim=-1)
        pitch = torchaudio.functional.detect_pitch_frequency(waveform, sample_rate=self.sample_rate).squeeze(0)
        return mel, energy, pitch

if __name__ == '__main__':
    print('[SUCCESS] Preprocessor ready for audio feature extraction.')
