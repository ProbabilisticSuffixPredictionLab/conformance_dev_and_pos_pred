import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

class LSTM_collective_IDP(nn.Module):
    """
    Unidirectional multi-layer LSTM for trace-level multi-label classification.
    Processes padded sequences (N, T, A) using pack_padded_sequence.
    Output: (N, L) probabilities for each deviation label.
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

        # embedding dim equal:
        lstm_input_size_activity = num_activities * embedding_dim
        lstm_input_size_resources = num_resources * embedding_dim
        lstm_input_size_months = num_months * embedding_dim

        # LSTM per categorical attribute:
        # Activities
        self.lstm_activity = nn.LSTM(input_size=lstm_input_size_activity,
                                     hidden_size=lstm_hidden)
        # Resources
        self.lstm_resource = nn.LSTM(input_size=lstm_input_size_resources,
                                     hidden_size=lstm_hidden)
        # Month
        self.lstm_month = nn.LSTM(input_size=lstm_input_size_months,
                                  hidden_size=lstm_hidden)
        
        # fully connected: feed forward: same for all attributes
        self.fc_lstm = nn.Linear(lstm_hidden, num_output_labels)
        
        self.fc_trace = nn.Linear(num_trace_atts, num_output_labels)
        
        # Final processing
        self.layer_norm = nn.LayerNorm(num_output_labels)
        
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, 
                x_acts: torch.Tensor,
                x_res: torch.Tensor,
                x_month: torch.Tensor,
                x_trace_atts: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, T, A) LongTensor - integer indices per attribute
            lengths: (N,) LongTensor - true sequence lengths

        Returns:
            probs: (N, L) FloatTensor - independent probability per label
        """
        N, T, A = x_acts.shape
        

        # Embed and concatenate attributes
        embedded = [self.embeddings[i](x_acts[:, :, i]) for i in range(A)]
        x_emb = torch.cat(embedded, dim=-1)  # (N, T, A * embedding_dim)

        # Pack padded sequences
        lengths_sorted, idx_sort = torch.sort(lengths, descending=True)
        x_sorted = x_emb.index_select(0, idx_sort)
        packed = pack_padded_sequence(x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True)

        # LSTM forward pass
        _, (hn, _) = self.lstm_activity(packed)  # hn: (num_layers, N_sorted, hidden)

        # Extract last hidden state from each layer
        h_layer = hn[-1]  # (N_sorted, hidden) - output of first layer


        
        # Concatenate all four
        h_combined = torch.cat([h_layer1, h_layer2], dim=-1)  # (N_sorted, 2*hidden)
        # Restore original order
        _, idx_unsort = torch.sort(idx_sort)
        h_combined = h_combined.index_select(0, idx_unsort)

        # Classifier
        # logits = self.fc(h_combined)
        
        logits = self.layer_norm(h_combined)
        
        logits = self.relu(logits)
        
        logits = self.dropout(logits)
        
        probs = self.sigmoid(logits)  # (N, L)
        
        

        return probs
