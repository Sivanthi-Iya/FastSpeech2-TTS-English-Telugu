import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class VariancePredictor(nn.Module):
    def __init__(self, d_model=256, filter_size=256, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv_1 = nn.Conv1d(d_model, filter_size, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.relu = nn.ReLU()
        self.layer_norm_1 = nn.LayerNorm(filter_size)
        self.dropout = nn.Dropout(dropout)
        self.conv_2 = nn.Conv1d(filter_size, filter_size, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.layer_norm_2 = nn.LayerNorm(filter_size)
        self.linear = nn.Linear(filter_size, 1)

    def forward(self, x):
        x = self.conv_1(x.transpose(1, 2))
        x = self.layer_norm_1(self.relu(x).transpose(1, 2))
        x = self.dropout(x)
        x = self.conv_2(x.transpose(1, 2))
        x = self.layer_norm_2(self.relu(x).transpose(1, 2))
        x = self.dropout(x)
        return self.linear(x).squeeze(-1)

class LengthRegulator(nn.Module):
    def forward(self, x, duration):
        output = []
        for batch_idx in range(x.size(0)):
            seq, dur = x[batch_idx], duration[batch_idx]
            expanded = [seq[i].repeat(max(int(torch.round(dur[i]).item()), 0), 1) for i in range(seq.size(0))]
            output.append(torch.cat(expanded, dim=0) if len(expanded) > 0 else seq)
        max_len = max([o.size(0) for o in output])
        padded = torch.zeros(x.size(0), max_len, x.size(2), device=x.device)
        for i, o in enumerate(output):
            padded[i, :o.size(0), :] = o
        return padded

class VarianceAdaptor(nn.Module):
    def __init__(self, d_model=256, n_bins=256):
        super().__init__()
        self.duration_predictor = VariancePredictor(d_model)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(d_model)
        self.energy_predictor = VariancePredictor(d_model)
        self.pitch_embedding = nn.Embedding(n_bins, d_model)
        self.energy_embedding = nn.Embedding(n_bins, d_model)

    def forward(self, x, duration=None):
        dur_pred = self.duration_predictor(x)
        d_use = duration if duration is not None else torch.clamp(dur_pred, min=1.0)
        x = self.length_regulator(x, d_use)
        pitch_pred = self.pitch_predictor(x)
        energy_pred = self.energy_predictor(x)
        p_q = torch.clamp((pitch_pred * 25.5).long(), 0, 255)
        e_q = torch.clamp((energy_pred * 25.5).long(), 0, 255)
        return x + self.pitch_embedding(p_q) + self.energy_embedding(e_q), dur_pred, pitch_pred, energy_pred

class FastSpeech2(nn.Module):
    def __init__(self, vocab_size=350, d_model=256, n_mels=80):
        super().__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=1024, batch_first=True), num_layers=4)
        self.variance_adaptor = VarianceAdaptor(d_model)
        self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=1024, batch_first=True), num_layers=4)
        self.mel_linear = nn.Linear(d_model, n_mels)

    def forward(self, text, duration=None):
        emb = self.encoder_embedding(text)
        enc = self.encoder(emb)
        adapt, dur_pred, p_pred, e_pred = self.variance_adaptor(enc, duration)
        dec = self.decoder(adapt)
        return self.mel_linear(dec), dur_pred, p_pred, e_pred

if __name__ == '__main__':
    print('[SUCCESS] FastSpeech 2 Model & Variance Adaptor Architecture Ready.')
