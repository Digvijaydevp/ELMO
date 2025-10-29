import torch
import nltk
import sys
from torch.nn import functional as F
from ELMO import ELMo
from classification import NewsClassifier

nltk.download('punkt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_models(saved_model_path):
    # Load classifier state
    classifier_state = torch.load(saved_model_path, map_location=device)
    
    # Check if ELMo is used by analyzing filename
    model_name = saved_model_path.lower()
    
    if 'elmo' in model_name:
        # Load ELMo model
        elmo_checkpoint = torch.load('bilstm.pt', map_location=device)
        vocab = elmo_checkpoint['vocab']
        elmo_model = ELMo(len(vocab), 256, 256).to(device)
        elmo_model.load_state_dict(elmo_checkpoint['model_state'])
        
        # Extract hyperparameter setting from the model name
        if 'trainable_lambdas' in model_name:
            hyperparam_setting = 'trainable_lambdas'
        elif 'frozen_lambdas' in model_name:
            hyperparam_setting = 'frozen_lambdas'    
        else:
            # Default or extract other possible hyperparameter settings
            hyperparam_setting = 'learnable_function'  # Default
        
        # Initialize classifier with ELMo
        classifier = NewsClassifier(
            elmo_model=elmo_model,
            hidden_dim=512,
            num_classes=4,
            hyperparam_setting=hyperparam_setting
        )
        classifier.load_state_dict(classifier_state)
    else:
        # For static embeddings, we'll need to extract the embeddings and vocab 
        # from the model file directly
        
        # First, load the static embeddings to get vocab for preprocessing
        embedding_type = 'svd'  # Default
        if 'svd' in model_name:
            embedding_type = 'svd'
        elif 'skipgram' in model_name:
            embedding_type = 'skipgram'
        elif 'cbow' in model_name:
            embedding_type = 'cbow'
            
        # Load vocabulary from the classifier state
        if 'static_vocab' in classifier_state:
            vocab = classifier_state['static_vocab']
        else:
            # Try to load from embedding file with different possible key names
            static_checkpoint = torch.load(f'{embedding_type}.pt', map_location=device)
            
            # Check different possible keys for vocabulary
            if 'word_to_idx' in static_checkpoint:
                vocab = static_checkpoint['word_to_idx']
            elif 'vocab' in static_checkpoint:
                vocab = static_checkpoint['vocab']
            else:
                # If none of the expected keys are found, print available keys and raise error
                print(f"Available keys in {embedding_type}.pt: {list(static_checkpoint.keys())}")
                raise KeyError(f"Could not find vocabulary in {embedding_type}.pt")
            
        # Create a classifier with the right embedding size from the state dict
        embedding_weight = None
        for key, value in classifier_state.items():
            if 'embedding_layer.weight' in key:
                embedding_weight = value
                break
        
        if embedding_weight is not None:
            vocab_size, embed_dim = embedding_weight.shape
            # Create an empty embedding layer with the correct dimensions
            embedding_layer = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0).to(device)
            # Don't initialize with embeddings, let the state_dict load do that
            classifier = NewsClassifier(static_embedding=embedding_layer, hidden_dim=512, num_classes=4)
            classifier.load_state_dict(classifier_state)
        else:
            raise ValueError(f"Could not find embedding weights in {saved_model_path}")
            
    classifier.to(device)  # Ensure classifier is on the correct device
    classifier.eval()
    return classifier, vocab

def preprocess(text, vocab, max_len=128):
    tokens = nltk.word_tokenize(text.lower())
    indices = [vocab.get(token, vocab.get('<unk>', 1)) for token in tokens]
    indices = indices[:max_len] + [0] * (max_len - len(indices))
    return torch.tensor(indices, device=device).unsqueeze(0)

def main():
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_path> <description>")
        return
    
    model_path = sys.argv[1]
    description = ' '.join(sys.argv[2:])
    
    classifier, vocab = load_models(model_path)
    input_tensor = preprocess(description, vocab)
    
    with torch.no_grad():
        logits = classifier(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().tolist()
    
    for i, prob in enumerate(probs):
        print(f"class-{i+1} {prob:.4f}")

if __name__ == "__main__":
    main()