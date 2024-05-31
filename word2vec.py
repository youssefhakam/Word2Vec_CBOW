import torch
import torch.nn as nn 
import pandas as pd
from datasets import load_dataset


### Define Hyperparametre 
embedding_size = 100  ## Dimensionality of word embedding
vocab_size = 10000 ## Number of words in vocabulary
window_size = 2 ## Context window size


datasets = load_dataset("atlasia/darija-translation")
df = datasets['train'].to_pandas()
df = df['darija']

def tokenizer(sentences) : 
  tokens = []
  for sentence in sentences :
    tokens.append(sentence.split())
  return tokens

def build_vocab(tokenized) :
  vocab = {}
  for sentence in tokenized : 
    for word in sentence : 
      if word not in vocab : 
        vocab[word] = len(vocab)
  return vocab

tokenized_sentences = tokenizer(df)
vocab = build_vocab(tokenized_sentences)
vocab_size = len(vocab)

word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
