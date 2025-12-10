import os
import math
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import matplotlib.pyplot as plt  # vẽ loss

# ==========================
# CONFIG
# ==========================

DATA_DIR = "data"

TRAIN_EN = os.path.join(DATA_DIR, "train.en")
TRAIN_FR = os.path.join(DATA_DIR, "train.fr")
VAL_EN = os.path.join(DATA_DIR, "val.en")
VAL_FR = os.path.join(DATA_DIR, "val.fr")
TEST_EN = os.path.join(DATA_DIR, "test.en")
TEST_FR = os.path.join(DATA_DIR, "test.fr")

SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

MIN_FREQ = 1
MAX_VOCAB_SIZE = 10_000

BATCH_SIZE = 64
EMB_DIM = 256
HID_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
TEACHER_FORCING_RATIO = 0.5
MAX_LEN = 50
PATIENCE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================
# UTILITIES
# ==========================

def read_parallel(en_path: str, fr_path: str) -> Tuple[List[str], List[str]]:
    """Đọc 2 file song ngữ (đã được align)."""
    with open(en_path, encoding="utf-8") as f_en, open(fr_path, encoding="utf-8") as f_fr:
        en_lines = [l.strip() for l in f_en]
        fr_lines = [l.strip() for l in f_fr]
    assert len(en_lines) == len(fr_lines), "Số dòng EN và FR không khớp!"
    return en_lines, fr_lines


print("Loading SpaCy tokenizers...")
en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
fr_tokenizer = get_tokenizer("spacy", language="fr_core_news_sm")


def yield_tokens(sentences: List[str], tokenizer):
    for sent in sentences:
        yield tokenizer(sent.lower())


def build_vocab(sentences: List[str], tokenizer):
    """
    Xây vocab bằng build_vocab_from_iterator.
    Có thêm specials: <unk>, <pad>, <sos>, <eos>.
    Giới hạn ~MAX_VOCAB_SIZE token phổ biến nhất (bao gồm specials).
    """
    print("  > Building raw vocab...")
    vocab = build_vocab_from_iterator(
        yield_tokens(sentences, tokenizer),
        min_freq=MIN_FREQ,
        specials=SPECIAL_TOKENS,
        special_first=True,
    )
    vocab.set_default_index(UNK_IDX)

    if len(vocab) > MAX_VOCAB_SIZE:
        print(f"  > Trimming vocab from {len(vocab)} to {MAX_VOCAB_SIZE}")
        all_tokens = vocab.get_itos()
        base_tokens = all_tokens[len(SPECIAL_TOKENS):MAX_VOCAB_SIZE]

        def iterator():
            for tok in base_tokens:
                yield [tok]

        vocab = build_vocab_from_iterator(
            iterator(),
            specials=SPECIAL_TOKENS,
            special_first=True,
        )
        vocab.set_default_index(UNK_IDX)

    print("  > Vocab size:", len(vocab))
    return vocab


# ==========================
# DATASET + COLLATE FN
# ==========================

class TranslationDataset(Dataset):
    def __init__(self, src_sents, tgt_sents, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab):
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_sents)

    def encode_sentence(self, sentence, tokenizer, vocab):
        tokens = tokenizer(sentence.lower())
        ids = [vocab[token] for token in tokens]
        return torch.tensor(
            [SOS_IDX] + ids + [EOS_IDX], dtype=torch.long
        )

    def __getitem__(self, idx):
        src = self.encode_sentence(self.src_sents[idx], self.src_tokenizer, self.src_vocab)
        tgt = self.encode_sentence(self.tgt_sents[idx], self.tgt_tokenizer, self.tgt_vocab)
        return src, tgt


def collate_fn(batch):
    """
    batch: list of (src_tensor, tgt_tensor)
    - Sort theo độ dài src giảm dần (cho pack_padded_sequence)
    - Pad src & tgt
    """
    src_seqs, tgt_seqs = zip(*batch)

    src_lengths = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    sorted_idx = torch.argsort(src_lengths, descending=True).tolist()  # <-- FIX

    src_seqs = [src_seqs[i] for i in sorted_idx]
    tgt_seqs = [tgt_seqs[i] for i in sorted_idx]
    src_lengths = src_lengths[sorted_idx]

    src_padded = pad_sequence(src_seqs, padding_value=PAD_IDX)  # [src_len, batch]
    tgt_padded = pad_sequence(tgt_seqs, padding_value=PAD_IDX)  # [tgt_len, batch]

    return src_padded, src_lengths, tgt_padded


# ==========================
# MASK CHO ATTENTION
# ==========================

def make_src_mask(src, src_lengths):
    """
    src: [src_len, batch]
    src_lengths: [batch]
    return: mask [batch, src_len] (True = real token, False = pad)
    """
    src_len = src.size(0)
    mask = torch.arange(src_len, device=src.device).unsqueeze(0) < src_lengths.unsqueeze(1)
    return mask


# ==========================
# MODEL: ENCODER-DECODER LSTM + LUONG ATTENTION
# ==========================

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(self, src, src_lengths):
        """
        src: [src_len, batch]
        src_lengths: [batch]
        """
        embedded = self.embedding(src)  # [src_len, batch, emb_dim]
        packed = pack_padded_sequence(
            embedded, src_lengths.cpu(),
            enforce_sorted=True
        )
        packed_outputs, (hidden, cell) = self.lstm(packed)
        encoder_outputs, _ = pad_packed_sequence(packed_outputs)  # [src_len, batch, hid_dim]
        return encoder_outputs, hidden, cell


class LuongAttention(nn.Module):
    """
    Luong dot-product attention:
        score(h_t, h_s) = h_s^T h_t
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim

    def forward(self, decoder_hidden, encoder_outputs, src_mask):
        """
        decoder_hidden: [batch, hid_dim] (h_t)
        encoder_outputs: [src_len, batch, hid_dim] (h_1..h_n)
        src_mask: [batch, src_len] (True = real token, False = pad)
        """
        enc_out = encoder_outputs.permute(1, 0, 2)          # [B, S, H]
        dec = decoder_hidden.unsqueeze(2)                   # [B, H, 1]

        scores = torch.bmm(enc_out, dec).squeeze(2)         # [B, S]
        scores = scores.masked_fill(~src_mask, -1e9)

        attn_weights = torch.softmax(scores, dim=1)         # [B, S]
        context = torch.bmm(attn_weights.unsqueeze(1), enc_out).squeeze(1)  # [B, H]

        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=1, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attn = LuongAttention(hid_dim)
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)  # concat[hidden, context]
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs, src_mask):
        """
        input: [batch]
        hidden, cell: [n_layers, batch, hid_dim]
        encoder_outputs: [src_len, batch, hid_dim]
        src_mask: [batch, src_len]
        """
        input = input.unsqueeze(0)                      # [1, B]
        embedded = self.dropout(self.embedding(input))  # [1, B, E]

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        dec_hidden = output.squeeze(0)                  # [B, H]

        context, attn_weights = self.attn(dec_hidden, encoder_outputs, src_mask)
        concat = torch.cat([dec_hidden, context], dim=1)  # [B, 2H]
        prediction = self.fc_out(concat)                  # [B, vocab]

        return prediction, hidden, cell, attn_weights


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        """
        src: [src_len, batch]
        src_lengths: [batch]
        tgt: [tgt_len, batch]
        """
        batch_size = tgt.shape[1]
        tgt_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size, device=self.device)

        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        src_mask = make_src_mask(src, src_lengths.to(src.device))

        input_tokens = tgt[0, :]  # <sos>

        for t in range(1, tgt_len):
            output, hidden, cell, _ = self.decoder(
                input_tokens, hidden, cell, encoder_outputs, src_mask
            )
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)  # [batch]
            input_tokens = tgt[t] if teacher_force else top1

        return outputs

    def translate(self, sentence: str, src_tokenizer, src_vocab, tgt_vocab, max_len=50):
        """
        Greedy decoding cho 1 câu với attention.
        """
        self.eval()
        tokens = src_tokenizer(sentence.lower())
        ids = [src_vocab[token] for token in tokens]
        src_tensor = torch.tensor(
            [SOS_IDX] + ids + [EOS_IDX], dtype=torch.long
        ).unsqueeze(1).to(self.device)  # [src_len, 1]
        src_len = torch.tensor([src_tensor.size(0)], dtype=torch.long, device=self.device)

        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src_tensor, src_len)
            src_mask = make_src_mask(src_tensor, src_len)

            input_token = torch.tensor([SOS_IDX], dtype=torch.long, device=self.device)
            outputs = []

            for _ in range(max_len):
                output, hidden, cell, _ = self.decoder(
                    input_token, hidden, cell, encoder_outputs, src_mask
                )
                top1 = output.argmax(1)  # [1]
                if top1.item() == EOS_IDX:
                    break
                outputs.append(top1.item())
                input_token = top1

        tokens_out = []
        itos = tgt_vocab.get_itos()
        for idx in outputs:
            if idx in (PAD_IDX, SOS_IDX, EOS_IDX):
                continue
            tokens_out.append(itos[idx])
        return " ".join(tokens_out)


# ==========================
# TRAINING / EVAL FUNCTIONS
# ==========================

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0.0
    for src, src_lengths, tgt in dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src_lengths = src_lengths.to(DEVICE)

        optimizer.zero_grad()
        output = model(src, src_lengths, tgt, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)  # bỏ <sos>
        tgt_gold = tgt[1:].reshape(-1)

        loss = criterion(output, tgt_gold)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for src, src_lengths, tgt in dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            src_lengths = src_lengths.to(DEVICE)

            output = model(src, src_lengths, tgt, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            tgt_gold = tgt[1:].reshape(-1)

            loss = criterion(output, tgt_gold)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def compute_bleu(model, src_sents, tgt_sents, src_tokenizer, src_vocab, tgt_tokenizer, tgt_vocab):
    """Tính BLEU trung bình trên test set."""
    smoothie = SmoothingFunction().method4
    scores = []
    for src, tgt in zip(src_sents, tgt_sents):
        pred = model.translate(src, src_tokenizer, src_vocab, tgt_vocab, max_len=MAX_LEN)
        ref_tokens = tgt_tokenizer(tgt.lower())
        hyp_tokens = pred.split()
        if len(hyp_tokens) == 0:
            scores.append(0.0)
        else:
            scores.append(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie))
    return sum(scores) / len(scores)


# ==========================
# MAIN
# ==========================

def main():
    print("Reading datasets...")
    train_en, train_fr = read_parallel(TRAIN_EN, TRAIN_FR)
    val_en, val_fr = read_parallel(VAL_EN, VAL_FR)
    test_en, test_fr = read_parallel(TEST_EN, TEST_FR)

    print("\nBuilding vocabularies...")
    en_vocab = build_vocab(train_en, en_tokenizer)
    fr_vocab = build_vocab(train_fr, fr_tokenizer)

    print("\nCreating datasets & dataloaders...")
    train_dataset = TranslationDataset(train_en, train_fr, en_tokenizer, fr_tokenizer, en_vocab, fr_vocab)
    val_dataset = TranslationDataset(val_en, val_fr, en_tokenizer, fr_tokenizer, en_vocab, fr_vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print("\nInitializing model with attention...")
    encoder = Encoder(len(en_vocab), EMB_DIM, HID_DIM, n_layers=NUM_LAYERS, dropout=DROPOUT)
    decoder = Decoder(len(fr_vocab), EMB_DIM, HID_DIM, n_layers=NUM_LAYERS, dropout=DROPOUT)
    model = EncoderDecoder(encoder, decoder, DEVICE).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=True
    )

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    train_losses = []
    val_losses = []

    print("\nStart training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.2f} | "
            f"Val Loss: {val_loss:.3f} | Val PPL: {math.exp(val_loss):.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "en_vocab": en_vocab,   # <-- LƯU TRỰC TIẾP VOCAB
                    "fr_vocab": fr_vocab,
                    "config": {
                        "emb_dim": EMB_DIM,
                        "hid_dim": HID_DIM,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                        "use_attention": True,
                    },
                },
                "best_model_attention.pth",
            )
            print("  => Saved new best model to best_model_attention.pth")
        else:
            patience_counter += 1
            print(f"  > Early stopping patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (best at epoch {best_epoch})")
            break

    # Vẽ loss
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train loss")
    plt.plot(epochs_range, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss (with Attention)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve_attention.png")
    print("\nSaved loss curve to loss_curve_attention.png")

    print("\nTraining finished.")
    print("Loading best_model_attention.pth for evaluation...")
    checkpoint = torch.load("best_model_attention.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    print("\nTranslating a few validation examples:")
    for i in range(5):
        src = val_en[i]
        tgt = val_fr[i]
        pred = model.translate(src, en_tokenizer, en_vocab, fr_vocab, max_len=MAX_LEN)
        print("-" * 60)
        print("EN :", src)
        print("FR (gold):", tgt)
        print("FR (pred):", pred)

    print("\nComputing BLEU score on test set...")
    bleu = compute_bleu(
        model,
        test_en,
        test_fr,
        en_tokenizer,
        en_vocab,
        fr_tokenizer,
        fr_vocab,
    )
    print(f"Test BLEU score (mean sentence BLEU, attention): {bleu:.4f}")


# Hàm translate(sentence: str) dùng model có attention
def translate(sentence: str) -> str:
    """
    API đơn giản cho đề bài:
        def translate(sentence: str) -> str
    Dùng model attention, giả định đã có best_model_attention.pth.
    """
    checkpoint = torch.load("best_model_attention.pth", map_location=DEVICE)
    en_vocab = checkpoint["en_vocab"]
    fr_vocab = checkpoint["fr_vocab"]
    config = checkpoint["config"]

    encoder = Encoder(len(en_vocab), config["emb_dim"], config["hid_dim"],
                      n_layers=config["num_layers"], dropout=config["dropout"])
    decoder = Decoder(len(fr_vocab), config["emb_dim"], config["hid_dim"],
                      n_layers=config["num_layers"], dropout=config["dropout"])
    model = EncoderDecoder(encoder, decoder, DEVICE).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model.translate(sentence, en_tokenizer, en_vocab, fr_vocab, max_len=MAX_LEN)


if __name__ == "__main__":
    main()
