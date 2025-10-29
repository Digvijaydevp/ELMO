import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ELMO import ELMo
import seaborn as sns
import os
import argparse
import nltk
nltk.download('punkt')

# Hyperparameters
NUM_CLASSES = 4
HIDDEN_DIM_CLS = 512
NUM_LAYERS = 1
BATCH_SIZE = 32
NUM_EPOCHS = 5
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NewsDataset(Dataset):
    def __init__(self, descriptions, labels, vocab, max_len=MAX_LEN):
        self.descriptions = descriptions
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        description = self.descriptions[idx]
        tokens = nltk.word_tokenize(description.lower())
        indices = [self.vocab.get(token, self.vocab.get('<unk>', 0)) for token in tokens]
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices += [self.vocab.get('<pad>', 1)] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long).to(DEVICE), torch.tensor(self.labels[idx], dtype=torch.long).to(DEVICE)

class NewsClassifier(nn.Module):
    def __init__(self, elmo_model=None, static_embedding=None, hidden_dim=256, 
                num_classes=4, hyperparam_setting=None):
        super().__init__()
        if elmo_model:
            self.embedding_type = 'elmo'
            self.elmo = elmo_model
            for param in self.elmo.parameters():
                param.requires_grad = False
            self.hyperparam_setting = hyperparam_setting

            # Projection layers for ELMo embeddings
            self.proj0 = nn.Linear(256, 512)  # Project embedding layer to 512
            self.proj1 = nn.Identity()        # Layer 1 already 512-dim
            self.proj2 = nn.Identity()        # Layer 2 already 512-dim
            
            if self.hyperparam_setting in ['trainable_lambdas', 'frozen_lambdas']:
                self.lambda0 = nn.Parameter(torch.rand(1))
                self.lambda1 = nn.Parameter(torch.rand(1))
                self.lambda2 = nn.Parameter(torch.rand(1))
                if self.hyperparam_setting == 'frozen_lambdas':
                    self.lambda0.requires_grad_(False)
                    self.lambda1.requires_grad_(False)
                    self.lambda2.requires_grad_(False)
            elif self.hyperparam_setting == 'learnable_function':
                self.combiner = nn.Sequential(
                    nn.Linear(1536, 512),
                    nn.LayerNorm(512),  # LayerNorm before ReLU
                    nn.ReLU(),
                    nn.Dropout(0.3),    # Add dropout for regularization
                )
                # Initialize weights with He initialization
                for layer in self.combiner:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            else:
                raise ValueError("Invalid hyperparameter setting")
            lstm_input_size = 512
        else:
            self.embedding_type = 'static'
            self.embedding_layer = static_embedding
            lstm_input_size = static_embedding.embedding_dim
            
        self.rnn = nn.LSTM(lstm_input_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if self.embedding_type == 'elmo':
            with torch.no_grad():
                elmo_out = self.elmo(x, x.flip(dims=[1]))
            e0, e1, e2 = [e.detach() for e in elmo_out['embeddings']]
            
            if self.hyperparam_setting in ['trainable_lambdas', 'frozen_lambdas']:
                # Project and combine embeddings
                e0_proj = self.proj0(e0)
                e1_proj = self.proj1(e1)
                e2_proj = self.proj2(e2)
                combined = self.lambda0 * e0_proj + self.lambda1 * e1_proj + self.lambda2 * e2_proj
            elif self.hyperparam_setting == 'learnable_function':
                e0_proj = self.proj0(e0)
                combined = torch.cat([e0_proj, e1, e2], dim=2)
                combined = self.combiner(combined)
        else:
            combined = self.embedding_layer(x)
            
        output, _ = self.rnn(combined)
        output = output[:, -1, :]
        return self.fc(output)

def evaluate_model(model, dataloader, phase='test'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{phase.capitalize()} Confusion Matrix')
    plt.savefig(f'{phase}_confusion_matrix_.png')
    plt.close()
    
    report = classification_report(all_labels, all_preds, output_dict=True)
    return report

def save_results(embedding_type, mode, train_report, test_report):
    results = {
        'Embedding': embedding_type,
        'Mode': mode,
        'Train Accuracy': train_report['accuracy'],
        'Train Precision': train_report['weighted avg']['precision'],
        'Train Recall': train_report['weighted avg']['recall'],
        'Train F1': train_report['weighted avg']['f1-score'],
        'Test Accuracy': test_report['accuracy'],
        'Test Precision': test_report['weighted avg']['precision'],
        'Test Recall': test_report['weighted avg']['recall'],
        'Test F1': test_report['weighted avg']['f1-score'],
    }
    
    df = pd.DataFrame([results])
    df.to_csv('embedding_results.csv', mode='a', header=not os.path.exists('embedding_results.csv'))

def load_static_embeddings(embedding_type):
    checkpoint = torch.load(f'{embedding_type}.pt', map_location=DEVICE)
    
    if embedding_type == 'skipgram':
        center_emb = checkpoint['center_embeddings']['weight'].float().to(DEVICE)
        context_emb = checkpoint['context_embeddings']['weight'].float().to(DEVICE)
        embeddings = (center_emb + context_emb) / 2
    else:
        embeddings = checkpoint['embeddings'].float().to(DEVICE)
    
    vocab = checkpoint.get('word_to_idx', checkpoint.get('vocab', {}))
    
    # Expand embeddings with special tokens
    original_size = embeddings.size(0)
    if '<pad>' not in vocab:
        pad_vec = torch.zeros(1, embeddings.size(1), dtype=torch.float32, device=embeddings.device)
        embeddings = torch.cat([embeddings, pad_vec])
        vocab['<pad>'] = original_size
        
    if '<unk>' not in vocab:
        unk_vec = torch.randn(1, embeddings.size(1), dtype=torch.float32, device=embeddings.device) * 0.1
        embeddings = torch.cat([embeddings, unk_vec])
        vocab['<unk>'] = original_size + 1
    
    return embeddings, vocab

def train_classifier(embedding_type, mode=None):
    # Load embeddings and vocab
    if embedding_type == 'elmo':
        checkpoint = torch.load('bilstm.pt', map_location=DEVICE)
        vocab = checkpoint['vocab']
        model = ELMo(len(vocab), 256, 256).to(DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        classifier = NewsClassifier(elmo_model=model, hidden_dim=HIDDEN_DIM_CLS, 
                                num_classes=NUM_CLASSES, hyperparam_setting=mode)
    else:
        embedding_weights, vocab = load_static_embeddings(embedding_type)
        embedding_layer = nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=embedding_weights.size(1),
            padding_idx=vocab['<pad>']
        )
        embedding_layer.weight.data = embedding_weights
        embedding_layer.weight.requires_grad = False
        classifier = NewsClassifier(static_embedding=embedding_layer, hidden_dim=HIDDEN_DIM_CLS,
                                num_classes=NUM_CLASSES)

    # Load data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Process labels
    train_labels = [ci-1 for ci in train_df['Class Index'].tolist()]
    test_labels = [ci-1 for ci in test_df['Class Index'].tolist()]
    
    # Create datasets
    train_dataset = NewsDataset(train_df['Description'].tolist(), train_labels, vocab)
    test_dataset = NewsDataset(test_df['Description'].tolist(), test_labels, vocab)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model setup
    classifier = classifier.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters())
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        classifier.train()
        epoch_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping added
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=10.0)
            
            # Gradient flow check (every 50 batches)
            if batch_idx % 50 == 0:
                total_norm = 0.0
                for p in classifier.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Batch {batch_idx} Gradient Norm: {total_norm:.4f}")
                
            optimizer.step()
            epoch_loss += loss.item()

            if (batch_idx+1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        print(f'Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}')
    
    # Evaluation
    train_report = evaluate_model(classifier, train_loader, 'train')
    test_report = evaluate_model(classifier, test_loader, 'test')
    
    # Save results
    save_results(embedding_type, mode, train_report, test_report)
    torch.save(classifier.state_dict(), f'classifier_{embedding_type}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_type', type=str, 
                       choices=['elmo', 'cbow', 'svd', 'skipgram'], required=True)
    parser.add_argument('--mode', type=str, 
                       choices=['trainable_lambdas', 'frozen_lambdas', 'learnable_function'], 
                       required=False)
    args = parser.parse_args()
    
    if args.embedding_type == 'elmo' and not args.mode:
        parser.error("--mode is required for ELMo")
    
    train_classifier(args.embedding_type, args.mode)