"""
Seq2Seq Model — Bidirectional GRU Encoder + Attention + GRU Decoder
EN → UZ Translation
"""

import re
import unicodedata

import torch
import torch.nn as nn

# ─── Special tokens ───────────────────────────────────────────────────────────
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3


# ─── Text normalizer ──────────────────────────────────────────────────────────
def normalize_string(text: str) -> str:
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─── Encoder ──────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, embed_size, padding_idx=PAD_TOKEN)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.gru(embedded)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        hidden = torch.tanh(self.fc_hidden(torch.cat((forward_hidden, backward_hidden), dim=1)))
        return outputs, hidden


# ─── Attention ────────────────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        return torch.softmax(self.v(energy).squeeze(2), dim=1)


# ─── Decoder ──────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_size, hidden_size, num_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embed_size, padding_idx=PAD_TOKEN)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embed_size + hidden_size * 2,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_size * 3 + embed_size, output_dim)

    def forward(self, input_token, hidden, encoder_outputs):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))
        attention = self.attention(hidden, encoder_outputs).unsqueeze(1)
        context = torch.bmm(attention, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        hidden = hidden.unsqueeze(0)
        output, hidden = self.gru(rnn_input, hidden)
        hidden = hidden.squeeze(0)
        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        return prediction, hidden


# ─── Seq2Seq ──────────────────────────────────────────────────────────────────
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input_token = tgt[:, 0]
        for t in range(1, tgt_len):
            output, hidden = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t, :] = output
            top1 = output.argmax(1)
            input_token = tgt[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1
        return outputs


# ─── Inference helper ─────────────────────────────────────────────────────────
def translate_sentence(
    model: Seq2Seq,
    sentence: str,
    input_word2index: dict,
    output_index2word: dict,
    device: torch.device,
    max_length: int = 30,
) -> str:
    model.eval()
    sentence = normalize_string(sentence)
    tokens = sentence.split()
    indexes = [input_word2index.get(token, UNK_TOKEN) for token in tokens]
    indexes.append(EOS_TOKEN)

    src_tensor = torch.tensor(indexes, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    input_token = torch.tensor([SOS_TOKEN]).to(device)
    translated_tokens = []

    for _ in range(max_length):
        with torch.no_grad():
            output, hidden = model.decoder(input_token, hidden, encoder_outputs)

        best_guess = output.argmax(1).item()

        if best_guess == EOS_TOKEN:
            break

        word = output_index2word.get(best_guess, "<UNK>")
        if word not in ("<PAD>", "<SOS>", "<EOS>", "<UNK>"):
            translated_tokens.append(word)

        input_token = torch.tensor([best_guess]).to(device)

    return " ".join(translated_tokens) if translated_tokens else "..."
