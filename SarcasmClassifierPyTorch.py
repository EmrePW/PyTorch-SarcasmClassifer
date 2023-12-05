# Import Required Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
import json

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import Data
with open("sarcasm_v2.json", "r") as f:
    datastore = json.load(f)
data = []
for item in datastore:
    text = item['headline']
    label = item['is_sarcastic']
    element = (label, text)
    data.append(element)

# Train Test split
train_data, test_data = train_test_split(data, test_size=0.25, random_state=46)

# Build Vocab on train Data
tokenizer = get_tokenizer("basic_english")
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)
vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Functions that will convert sentences into sequences of numbers from the vocabulary
text_pipeline = lambda x: vocab(tokenizer(x)) 
class_sequencer = lambda x: int(x)

# Custom Collate Function
def collate_batch(batch):
    class_list, text_list = [], []
    for _class, _text in batch:
        class_list.append(class_sequencer(_class))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.long)
        text_list.append(processed_text)

    class_tensor = torch.tensor(class_list, dtype=torch.long)
    text_tensor = pad_sequence(text_list, batch_first=True)
    return (class_tensor.to(device), text_tensor.to(device))

# Declare hyperparameters
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 1 #1
NUM_EPOCHS = 20 
BATCH_SIZE = 5

# Define Model Architecture
class NetSarcasmClassifier(nn.Module):
    def __init__(self, vocabulary_size, embedding_dimension):
        super(NetSarcasmClassifier, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.pool = nn.AdaptiveAvgPool1d(20)
        self.fc1 = nn.Linear(20, 20)
        self.out = nn.Linear(20, 1)
        
        #torch.Size([batch_size, max_len_in_batch, pool_kernel])
    def forward(self, text):
        output = self.embedding(text) 
        output = output.mean(dim=2)
        output = self.pool(output)
        output = self.out(output)

        return F.sigmoid(output)


# Create model, errorFunction, optimizer, and scheduler
model = NetSarcasmClassifier(VOCAB_SIZE, EMBEDDING_DIM).to(device)
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2.0, gamma=0.1)

# DataLoaders
train_dataLoader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

# Train Function
def train(dataloader, epoch):
    model.train()
    correct, total = 0, 0
    print_step = 500

    for idx, (_class, _text) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(_text).squeeze()
        loss = loss_function(pred, _class.to(torch.float32))
        loss.backward()
        optimizer.step()
        correct += (pred == _class).sum().item()
        total += _class.size(0)
            
        if idx % print_step == 0 and idx > 0:
            print(f"Epoch : [{epoch}|{NUM_EPOCHS}], Batch [{idx}|{len(dataloader)}], Accuracy : {round((correct / total)*100, 3)}%, Loss = {loss.sum().item()}")
            correct, total = 0, 0

# Evaluation Function
def evaluate(dataloader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for idx, (_class, _text) in enumerate(dataloader):
            pred = model(_text).squeeze()
            loss = loss_function(pred, _class.to(torch.float32))
            correct += (pred == _class).sum().item()
            total += _class.size(0) 
        return (correct/total) * 100

# Test Function
def test(text, text_pipeline):
    with torch.no_grad():
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.long).view(1, -1).to(device)
        output = model(processed_text)
        return output.item()

overallAccuracy = None

# Train Network
def train_network():
    for epoch in range(1, NUM_EPOCHS + 1):
        train(train_dataLoader, epoch)
        accuracy = evaluate(test_dataloader)
        overallAccuracy = accuracy
        if overallAccuracy is not None and accuracy < overallAccuracy:
            scheduler.step()
    
        print(f"End of Epoch {epoch}, Validation Accuracy = [{round(overallAccuracy, 3)}%]")
        print(50 * "-")

# Test with raw data
def test_raw():
    
    sentences = [
        "granny starting to fear spiders in the garden might be real.",
        "game of thrones season finale showing this sunday night.",
        "It is okay if you do not like me. Not everyone has good taste.",
        "What should you buy?",
        "What holiday gifts should you buy?",
        "Learning is simple",
        "worst piece of art sold for astronomous amount"
    ]
    for sentence in sentences:
        prediction = test(sentence, text_pipeline)
        print(f"This sentence is {('Sarcastic' if prediction == 1.0 else 'Not Sarcastic')}")


if __name__ == "__main__":
    # train_network()
    # Save or Load a model
    # torch.save(model.state_dict(), "final7.pth")
    model.load_state_dict(torch.load("final7.pth"))

    test_raw()