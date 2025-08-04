import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128, num_layers=2, dropout=0.5, num_classes=4):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, 
                         dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids) 
        mask = attention_mask.unsqueeze(-1).float()
        embedded = embedded * mask
        gru_out, hidden = self.gru(embedded)  
        lengths = attention_mask.sum(dim=1) - 1  
        lengths = lengths.clamp(min=0)
        batch_idx = torch.arange(gru_out.size(0))
        last_output = gru_out[batch_idx, lengths]
        output = self.dropout(last_output)
        output = self.fc(output)
        return output