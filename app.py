import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer

st.set_page_config(
    layout="centered", page_title="Text Classifier", page_icon="❄️"
)

# Load Tokenizer and Config
tokenizer = BertTokenizer.from_pretrained('./model/Tokenzier')

# Instantiate model
model = BertForSequenceClassification.from_pretrained(
    './model/indobert')

w2i = {'positive': 0, 'neutral': 1, 'negative': 2}
i2w = {0: 'positive', 1: 'neutral', 2: 'negative'}


c1, c2 = st.columns([0.32, 2])

with c1:

    st.image(
        "images/logo.png",
        width=85,
    )


with c2:

    st.caption("")
    st.title("Sentiment Text Classifier")


pre_defined_text = "Bahagia hatiku melihat pernikahan putri sulungku yang cantik jelita"

text = st.text_area(
    "Enter text to classify",
    pre_defined_text,
    height=200,
)

st.write(text)

subwords = tokenizer.encode(text)
subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

logits = model(subwords)[0]
label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

st.write(
    f'Text: {text} | Label : {i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)')
