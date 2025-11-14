import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

class LSTM_collective_IDP(nn.Module):
    """
    IDP-collective LSTM:.
    Prefxes (CIBE encoded) and padded sequences: N x T x P.
    Output: N x K probabilities for each deviation label.
    """
    def __init__(self,
                 # Sizes inputs:
                 num_activities: int,
                 num_resources: int,
                 num_months:int,
                 num_trace_atts:int,
                 # Embedding dimension
                 embedding_dim: int = 16,
                 # LSTM layer hidden size
                 lstm_hidden: int = 128,
                 # Num output deviation labels
                 num_output_labels: int = None,
                 # Dropout rate
                 dropout: float = 0.1):
        
        super().__init__()
        
        if num_output_labels is None:
            raise ValueError("num_labels (L) must be provided.")

        # embeddings of all categorical event attributes
        self.embeddings_activity = nn.ModuleList([nn.Embedding(100, embedding_dim) for _ in range(num_activities)])
        self.embeddings_resources = nn.ModuleList([nn.Embedding(100, embedding_dim) for _ in range(num_resources)])
        self.embeddings_months = nn.ModuleList([nn.Embedding(100, embedding_dim) for _ in range(num_months)])

        # LSTM per categorical attribute:
        # Activities
        lstm_input_size_activity = num_activities * embedding_dim
        self.lstm_activity = nn.LSTM(input_size=lstm_input_size_activity,
                                     hidden_size=lstm_hidden)
        # Resources
        lstm_input_size_resources = num_resources * embedding_dim
        self.lstm_resource = nn.LSTM(input_size=lstm_input_size_resources,
                                     hidden_size=lstm_hidden)
        # Month
        lstm_input_size_months = num_months * embedding_dim
        self.lstm_month = nn.LSTM(input_size=lstm_input_size_months,
                                  hidden_size=lstm_hidden)
        
        # fully connected: feed forward: same for all attributes
        self.fc_lstm = nn.Linear(lstm_hidden, num_output_labels)
        
        # fully connected: for trce attributes
        self.fc_trace = nn.Linear(num_trace_atts, num_output_labels)
        
        # Final processing
        self.layer_norm = nn.LayerNorm(num_output_labels)
        
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, 
                prefixes: torch.Tensor) -> torch.Tensor:
        """
        
        add a prefix matrix:
        seq_len x batch_size x input_features? : T x N x P
        or 
        batch_size x seq_len x input_features? : N x T x P
        
        Return:
        batch x probabilties (for all deviations K): N x K
        """
        
        x_acts, x_res, x_month, x_trace_atts = prefixes
          
        # Embed categorical event attributes
        embedded_acts = self.embeddings_activity(x_acts)
        embedded_res = self.embeddings_resources(x_res)
        embedded_months = self.embeddings_months(x_month)
        
        # LSTM forward pass
        _, (h_act, _), _ = self.lstm_activity(embedded_acts)
        _, (h_res, _), _ = self.lstm_resource(embedded_res)
        _, (h_month, _), _ = self.lstm_month(embedded_months)
    
        # Forward pass through fully connected:
        h_fc_act = self.fc_lstm(h_act)
        h_fc_res = self.fc_lstm(h_res)
        h_fc_month = self.lstm_month(h_month)
        # forward pass thorugh fully connected trace atts:
        h_fc_trace = self.fc_trace(x_trace_atts)
    
        # Concatenate all four
        h_combined = torch.cat([h_fc_act, h_fc_res, h_fc_month, h_fc_trace], dim=-1)  # (N_sorted, 2*hidden)
        
        # layer norm      
        logits = self.layer_norm(h_combined)
        
        # ReLu
        logits = self.relu(logits)
        
        # Dropout
        logits = self.dropout(logits)
        
        # Sigmoid
        probs = self.sigmoid(logits)
    
        return probs
