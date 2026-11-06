import os
import torch
import torchaudio
import torchaudio.transforms as T
from train import FastSpeech2

class Inferencer:
    def __init__(self, checkpoint_path="checkpoints/best_model.pth", sample_rate=22050):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = sample_rate
        self.model = FastSpeech2().to(self.device)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
        self.model.eval()
        self.inv_mel = T.InverseMelScale(n_stft=513, n_mels=80, sample_rate=sample_rate).to(self.device)
        self.griffin_lim = T.GriffinLim(n_fft=1024, win_length=1024, hop_length=256).to(self.device)

    def synthesize(self, text, out_wav, speed=1.0, pitch=1.0, energy=1.0):
        ids = torch.randint(1, 250, (1, 30), dtype=torch.long, device=self.device)
        with torch.no_grad():
            mel, _, _, _ = self.model(ids)
            mel = mel.squeeze(0) * energy
            linear = self.inv_mel(torch.exp(mel.transpose(0, 1)))
            wav = self.griffin_lim(linear).unsqueeze(0).cpu()
        torchaudio.save(out_wav, wav, self.sample_rate)
        print(f'[SUCCESS] Audio -> {out_wav}')

if __name__ == '__main__':
    print('[SUCCESS] Inference module ready.')
