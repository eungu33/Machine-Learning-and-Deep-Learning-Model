# train_model_from_json.py
# JSON 기반 CNN 의미 임베딩 모델 학습 (vocab 저장 포함)

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from konlpy.tag import Okt
import numpy as np
from tqdm import tqdm
import pickle

# ---------------------------
# 설정
# ---------------------------
TRAIN_PATH = "./문화, 게임 콘텐츠 분야 용어 말뭉치/Training"
VAL_PATH = "./문화, 게임 콘텐츠 분야 용어 말뭉치/Validation"
SAVE_PATH = "cnn_embedder.pt"
VOCAB_PATH = "vocab.pkl"
BATCH_SIZE = 32
EPOCHS = 10
PATIENCE = 3
TRAIN_LIMIT = 10000
VAL_LIMIT = 2000
SIM_THRESHOLD = 0.8

# ---------------------------
# 전처리, Vocab 클래스
# ---------------------------
class Preprocessor:
    def __init__(self):
        self.okt = Okt()
    def tokenize(self, text):
        return self.okt.nouns(text)

class Vocab:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
    def build(self, token_lists):
        idx = 2
        for tokens in token_lists:
            for t in tokens:
                if t not in self.word2idx:
                    self.word2idx[t] = idx
                    self.idx2word[idx] = t
                    idx += 1
    def encode(self, tokens):
        return [self.word2idx.get(t, 1) for t in tokens]
    def pad(self, ids, max_len=30):
        return ids[:max_len] + [0] * (max_len - len(ids))

# ---------------------------
# Dataset 구성
# ---------------------------
class PairDataset(Dataset):
    def __init__(self, examples, vocab, tokenizer, max_len=30):
        self.examples = examples
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sent, definition = self.examples[idx]
        sent_tok = self.tokenizer.tokenize(sent)
        def_tok = self.tokenizer.tokenize(definition)

        sent_ids = self.vocab.pad(self.vocab.encode(sent_tok), self.max_len)
        def_ids = self.vocab.pad(self.vocab.encode(def_tok), self.max_len)

        return torch.tensor(sent_ids), torch.tensor(def_ids)

# ---------------------------
# CNN 임베딩 모델
# ---------------------------
class CNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, out_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, out_dim)
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# ---------------------------
# JSON 로딩 함수
# ---------------------------

def load_json_pairs(example_dir, limit=None):
    with open(os.path.join(example_dir, "용어.json"), encoding="utf-8") as f:
        term_dict = {str(t["id"]): t["definition"] for t in json.load(f)}

    examples = []
    for name in os.listdir(example_dir):
        if name.startswith("용례_") and name.endswith(".json"):
            with open(os.path.join(example_dir, name), encoding="utf-8") as f:
                for line in json.load(f):
                    sentence = line["sentence"]
                    for token in line["tokens"]:
                        tid = str(token["term_id"])
                        if tid in term_dict:
                            definition = term_dict[tid]
                            examples.append((sentence, definition))
                            if limit and len(examples) >= limit:
                                return examples
    return examples

# ---------------------------
# 학습 함수
# ---------------------------

def compute_accuracy(v1, v2, threshold=SIM_THRESHOLD):
    sim = F.cosine_similarity(v1, v2)
    pred = (sim > threshold).float()
    return pred.mean().item()

def train(model, train_loader, val_loader, epochs, patience, save_path):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CosineEmbeddingLoss()
    best_loss = float('inf')
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0.0
        count = 0
        for x1, x2 in tqdm(train_loader, desc=f"Epoch {epoch+1:02d} [Train]"):
            out1 = model(x1)
            out2 = model(x2)
            y = torch.ones(x1.size(0))
            loss = crit(out1, out2, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            acc = compute_accuracy(out1, out2)
            correct += acc * x1.size(0)
            count += x1.size(0)
        train_acc = correct / count

        model.eval()
        val_loss = 0.0
        correct = 0.0
        count = 0
        with torch.no_grad():
            for x1, x2 in tqdm(val_loader, desc=f"Epoch {epoch+1:02d} [Val]"):
                out1 = model(x1)
                out2 = model(x2)
                y = torch.ones(x1.size(0))
                loss = crit(out1, out2, y)
                val_loss += loss.item()
                acc = compute_accuracy(out1, out2)
                correct += acc * x1.size(0)
                count += x1.size(0)
        val_acc = correct / count
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), save_path)
            print("모델 저장 완료")
        else:
            wait += 1
            print(f"개선 없음: {wait}회 연속")
            if wait >= patience:
                print("조기 종료")
                break

# ---------------------------
# 실행
# ---------------------------
if __name__ == "__main__":
    tokenizer = Preprocessor()
    train_pairs = load_json_pairs(TRAIN_PATH, limit=TRAIN_LIMIT)
    val_pairs = load_json_pairs(VAL_PATH, limit=VAL_LIMIT)

    all_tokens = [tokenizer.tokenize(x) for pair in (train_pairs + val_pairs) for x in pair]
    vocab = Vocab()
    vocab.build(all_tokens)

    # vocab 저장
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)

    train_data = PairDataset(train_pairs, vocab, tokenizer)
    val_data = PairDataset(val_pairs, vocab, tokenizer)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = CNNEncoder(len(vocab.word2idx))
    train(model, train_loader, val_loader, EPOCHS, PATIENCE, SAVE_PATH)
    print("\n[학습 종료]")
