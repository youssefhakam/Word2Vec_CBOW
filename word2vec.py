import torch
import torch.nn as nn 
import pandas as pd
from datasets import load_dataset


### Define Hyperparametre 
embedding_size = 100  ## Dimensionality of word embedding
window_size = 3 ## Context window size
num_neg_samples = 2 # Number of negative samples


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

def create_context_target_pairs(phrases, window_size):
    context_target_pairs = []
    for phrase in phrases:
        words = phrase.split()
        for i in range(window_size, len(words) - window_size):
            context = words[i-window_size:i] + words[i+1:i+1+window_size]
            target = words[i]
            context_target_pairs.append((context, target))
    return context_target_pairs

def get_negative_samples(target_idx, vocab_size, num_neg_samples):
    neg_samples = []
    while len(neg_samples) < num_neg_samples:
        neg_sample = np.random.randint(0, vocab_size)
        if neg_sample != target_idx:
            neg_samples.append(neg_sample)
    return neg_samples

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context_idxs):
        embeds = self.embeddings(context_idxs)
        context_vec = torch.mean(embeds, dim=0).view((1, -1))
        out = self.linear(context_vec)
        log_probs = torch.log_softmax(out, dim=1)
        return log_probs

model = CBOWModel(vocab_size, embedding_size)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
# This determines the number of context words on each side of the target word
context_target_pairs = create_context_target_pairs(df, window_size)

for epoch in range(2):
    total_loss = 0
    for context, target in context_target_pairs:
        context_idxs = torch.tensor([word2idx[w] for w in context], dtype=torch.long)
        target_idx = torch.tensor([word2idx[target]], dtype=torch.long)
        
        model.zero_grad()
        
        log_probs = model(context_idxs)
        loss = loss_function(log_probs, target_idx)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {total_loss}')
embedding_matrix = model.embeddings.weight.data
embedding_matrix_np = embedding_matrix.numpy()
for word, idx in word2idx.items():
    print(f"Embedding for {word}: {embedding_matrix_np[idx]}")
