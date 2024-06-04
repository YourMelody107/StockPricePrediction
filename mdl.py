import os
from log import log
import torch
import torch.nn as nn
import torch.optim as optim
from lstm_attention import LSTM_Attention_Model


class MeowModel(object):
    def __init__(self, cacheDir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTM_Attention_Model(input_dim=20, hidden_dim=128, num_layers=2, output_dim=1).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def fit(self, xdf, ydf):
        self.model.train()
        xdf = torch.tensor(xdf.values, dtype=torch.float32).to(self.device)
        ydf = torch.tensor(ydf.values, dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(xdf, ydf)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(100):
            epoch.loss = 0
            for x_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            log.inf(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}")
        log.inf("Done fitting")
        
    def predict(self, xdf):
        self.model.eval()
        xdf = torch.tensor(xdf.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(xdf).cpu().numpy()
        return predictions
