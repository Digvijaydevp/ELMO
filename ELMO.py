import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from nltk.corpus import brown
from collections import Counter

# Determine available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Hyperparameters
VOCAB_SIZE = 20000
EMBEDDING_DIM = 256
HIDDEN_DIM = 256
BATCH_SIZE = 16
NUM_EPOCHS = 7
SEQ_LEN = 128
VALID_RATIO = 0.2  # 20% for validation

class BrownDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        forward_input = torch.tensor(seq[:-1], dtype=torch.long)
        forward_target = torch.tensor(seq[1:], dtype=torch.long)
        backward_input = torch.tensor(seq[::-1][:-1], dtype=torch.long)
        backward_target = torch.tensor(seq[::-1][1:], dtype=torch.long)
        return (forward_input, forward_target, backward_input, backward_target)

def collate_fn(batch):
    forward_inputs, forward_targets, backward_inputs, backward_targets = zip(*batch)

    forward_inputs = torch.nn.utils.rnn.pad_sequence(forward_inputs, batch_first=True, padding_value=0)
    forward_targets = torch.nn.utils.rnn.pad_sequence(forward_targets, batch_first=True, padding_value=0)
    backward_inputs = torch.nn.utils.rnn.pad_sequence(backward_inputs, batch_first=True, padding_value=0)
    backward_targets = torch.nn.utils.rnn.pad_sequence(backward_targets, batch_first=True, padding_value=0)

    return forward_inputs, forward_targets, backward_inputs, backward_targets

class ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.forward_lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.forward_lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.backward_lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.backward_lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.forward_proj = nn.Linear(hidden_dim, vocab_size)
        self.backward_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, forward_x, backward_x):
        embedded_forward = self.embedding(forward_x)
        embedded_backward = self.embedding(backward_x)

        # Forward pass
        fw_out1, _ = self.forward_lstm1(embedded_forward)
        fw_out2, _ = self.forward_lstm2(fw_out1)

        # Backward pass with sequence reversal
        bw_out1, _ = self.backward_lstm1(embedded_backward)
        bw_out1 = bw_out1.flip(1)  # Align with original sequence
        bw_out2, _ = self.backward_lstm2(bw_out1)

        # Combine bidirectional outputs
        layer1 = torch.cat([fw_out1, bw_out1], dim=-1)
        layer2 = torch.cat([fw_out2, bw_out2], dim=-1)

        return {
            'forward': [self.forward_proj(fw_out1), self.forward_proj(fw_out2)],
            'backward': [self.backward_proj(bw_out1), self.backward_proj(bw_out2)],
            'embeddings': [embedded_forward, layer1, layer2]
        }

def prepare_data():
    import nltk
    nltk.download('brown')
    sentences = brown.sents()

    vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    word_counts = Counter(word for sent in sentences for word in sent)

    # Modified: Include all words sorted by frequency
    sorted_words = [word for word, _ in word_counts.most_common(VOCAB_SIZE-4)]
    for idx, word in enumerate(sorted_words, start=4):
        vocab[word] = idx

    sequences = []
    for sent in sentences:
        tokens = ['<sos>'] + sent + ['<eos>']
        seq = [vocab.get(word, vocab['<unk>']) for word in tokens]
        seq = seq[:SEQ_LEN]
        sequences.append(seq)

    # Split into train and validation
    val_size = int(len(sequences) * VALID_RATIO)
    train_size = len(sequences) - val_size
    return random_split(sequences, [train_size, val_size]), vocab

def evaluate(model, val_loader, criterion,vocab):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for fw_in, fw_tgt, bw_in, bw_tgt in val_loader:
            fw_in = fw_in.to(device)
            fw_tgt = fw_tgt.to(device)
            bw_in = bw_in.to(device)
            bw_tgt = bw_tgt.to(device)

            outputs = model(fw_in, bw_in)

            fw_loss = sum(
                criterion(logits.view(-1, len(vocab)), fw_tgt.view(-1))
                for logits in outputs['forward']
            )
            bw_loss = sum(
                criterion(logits.view(-1, len(vocab)), bw_tgt.view(-1))
                for logits in outputs['backward']
            )

            loss = (fw_loss + bw_loss) / 4
            total_loss += loss.item()

    return total_loss / len(val_loader)

def train_elmo():
    (train_data, val_data), vocab = prepare_data()
    
    train_dataset = BrownDataset(train_data)
    val_dataset = BrownDataset(val_data)
    
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = ELMo(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-------------------------------")

        for batch_idx, (fw_in, fw_tgt, bw_in, bw_tgt) in enumerate(train_loader):
            fw_in = fw_in.to(device)
            fw_tgt = fw_tgt.to(device)
            bw_in = bw_in.to(device)
            bw_tgt = bw_tgt.to(device)

            optimizer.zero_grad()
            outputs = model(fw_in, bw_in)

            # Calculate losses
            fw_loss = sum(
                criterion(logits.view(-1, len(vocab)), fw_tgt.view(-1))
                for logits in outputs['forward']
            )
            bw_loss = sum(
                criterion(logits.view(-1, len(vocab)), bw_tgt.view(-1))
                for logits in outputs['backward']
            )

            loss = (fw_loss + bw_loss) / 4
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if (batch_idx+1) % 10 == 0:
                avg_loss = total_train_loss / (batch_idx+1)
                print(f"Train Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")

        # Validation
        val_loss = evaluate(model, val_loader, criterion, vocab)
        train_loss = total_train_loss / len(train_loader)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'vocab': vocab,
                'config': {'embed_dim': EMBEDDING_DIM, 'hidden_dim': HIDDEN_DIM}
            }, 'elmo_model_best.pt')
            print("Saved new best model!")

    # Save final model
    torch.save({
        'model_state': model.state_dict(),
        'vocab': vocab,
        'config': {'embed_dim': EMBEDDING_DIM, 'hidden_dim': HIDDEN_DIM}
    }, 'bilstm.pt')

if __name__ == '__main__':
    train_elmo()