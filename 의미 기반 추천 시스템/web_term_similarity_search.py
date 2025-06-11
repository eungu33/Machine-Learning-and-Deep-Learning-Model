# web_term_search_by_term.py
# 용어 입력 → 정의 + 관련 문장 5개 추출 프로그램 (CNN 임베딩 기반)

import os
import json
import torch
import torch.nn.functional as F
import streamlit as st
import pickle
from konlpy.tag import Okt
from train_model_from_json import CNNEncoder, Preprocessor, Vocab

# ---------------------------
# 설정
# ---------------------------
DATA_PATH = "./문화, 게임 콘텐츠 분야 용어 말뭉치/Validation"
MODEL_PATH = "cnn_embedder.pt"
VOCAB_PATH = "vocab.pkl"
MAX_LEN = 30

# ---------------------------
# 데이터 로딩
# ---------------------------
def load_term_data(example_dir):
    with open(os.path.join(example_dir, "용어.json"), encoding="utf-8") as f:
        term_dict = {t["term"]: t["definition"] for t in json.load(f)}

    sentences = []
    terms = []
    definitions = []

    for name in os.listdir(example_dir):
        if name.startswith("용례_") and name.endswith(".json"):
            with open(os.path.join(example_dir, name), encoding="utf-8") as f:
                for line in json.load(f):
                    sentence = line["sentence"]
                    for token in line["tokens"]:
                        term = token["sub"]
                        if term in term_dict:
                            definition = term_dict[term]
                            sentences.append(sentence)
                            terms.append(term)
                            definitions.append(definition)

    return term_dict, sentences, terms, definitions

# ---------------------------
# 문장 전처리 및 임베딩
# ---------------------------
def encode_sentences(sentences, tokenizer, vocab, model):
    encoded = []
    model.eval()
    with torch.no_grad():
        for s in sentences:
            tokens = tokenizer.tokenize(s)
            ids = vocab.encode(tokens)
            padded = vocab.pad(ids, MAX_LEN)
            tensor = torch.tensor([padded])
            out = model(tensor)
            encoded.append(out.squeeze(0))
    return torch.stack(encoded)

# ---------------------------
# Streamlit 인터페이스
# ---------------------------
if __name__ == "__main__":
    st.title("용어 기반 문장 추천기 (의미 유사도 기반)")

    tokenizer = Preprocessor()
    st.write("데이터 불러오는 중...")
    term_dict, sents, terms, defs = load_term_data(DATA_PATH)

    # 학습된 vocab 로드
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    # 모델 로드 및 정의 문장 임베딩
    model = CNNEncoder(len(vocab.word2idx))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with st.spinner("문장 임베딩 중..."):
        definition_vectors = encode_sentences(defs, tokenizer, vocab, model)

    selected_term = st.text_input("궁금한 용어를 입력하세요:")

    if selected_term:
        if selected_term not in term_dict:
            st.error("해당 용어의 정의를 찾을 수 없습니다.")
        else:
            definition = term_dict[selected_term]
            st.subheader(f"[정의] {selected_term} : {definition}")

            # 해당 용어와 유사한 문장 찾기
            tokens = tokenizer.tokenize(definition)
            ids = vocab.encode(tokens)
            padded = vocab.pad(ids, MAX_LEN)
            tensor = torch.tensor([padded])

            with torch.no_grad():
                query_vec = model(tensor)
                sims = F.cosine_similarity(query_vec, definition_vectors)
                topk = torch.topk(sims, 5)

            st.subheader("[관련 문장 추천 Top 5]")
            for i in topk.indices:
                st.write(f"문장: {sents[i]}\n- 용어: {terms[i]}\n- 정의: {defs[i]}\n- 유사도: {sims[i]:.4f}")