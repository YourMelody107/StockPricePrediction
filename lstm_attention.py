import torch
import torch.nn as nn
import torch.optim as optim

class LSTM_Attention_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM_Attention_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        att_weights = torch.softmax(self.attention(out), dim=1)
        att_out = torch.sum(out * att_weights, dim=1)
        
        output = self.fc(att_out)
        return output
