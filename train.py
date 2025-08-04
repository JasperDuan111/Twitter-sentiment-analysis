import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from data import create_dataloaders, tokenizer
from model import GRU

train_loader, val_loader, test_loader = create_dataloaders()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = tokenizer.vocab_size
model = GRU(vocab_size=vocab_size, embedding_dim=128, hidden_size=128, num_layers=2, dropout=0.5, num_classes=4)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs=30

best_val_acc = 0.0

for epoch in range(num_epochs):

    model.train()
    train_loss = 0.0
    train_predictions = []
    train_labels = []
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_predictions.extend(predicted.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
    
    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_predictions.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(train_labels, train_predictions)
    val_acc = accuracy_score(val_labels, val_predictions)
    
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f'Saving best model with val accuracy: {best_val_acc:.4f}')
        print('=' * 50)
        torch.save(model.state_dict(), 'model/best_model.pth')


model.eval()
test_predictions = []
test_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs.data, 1)
        test_predictions.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(test_labels, test_predictions)
print(f'\nTest Accuracy: {test_acc:.4f}')
print('\nClassification Report:')
print(classification_report(test_labels, test_predictions))

