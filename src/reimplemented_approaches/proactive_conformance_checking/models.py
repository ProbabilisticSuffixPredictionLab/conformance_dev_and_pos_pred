import torch
import torch.nn as nn

class LSTMCollectiveIDP(nn.Module):
    def __init__(self,
                 activity_vocab_size: int,
                 resource_vocab_size: int,
                 month_vocab_size: int,
                 num_trace_features: int,
                 embedding_dim: int = 16,
                 lstm_hidden: int = 128,
                 fc_hidden: int = 128,
                 num_output_labels: int = None,
                 dropout: float = 0.1):
        super().__init__()
        if num_output_labels is None:
            raise ValueError("num_output_labels must be provided")
        
        # Embeddings
        self.embed_act = nn.Embedding(activity_vocab_size, embedding_dim)
        self.embed_res = nn.Embedding(resource_vocab_size, embedding_dim)
        self.embed_month = nn.Embedding(month_vocab_size, embedding_dim)
        
        # Separate LSTMs
        self.lstm_activity = nn.LSTM(input_size=embedding_dim,
                                     hidden_size=lstm_hidden,
                                     batch_first=True)
        self.lstm_resource = nn.LSTM(input_size=embedding_dim,
                                     hidden_size=lstm_hidden,
                                     batch_first=True)
        self.lstm_month = nn.LSTM(input_size=embedding_dim,
                                  hidden_size=lstm_hidden,
                                  batch_first=True)
        
        # FC for LSTM outputs (map hidden -> fc_hidden)
        self.fc_lstm = nn.Linear(lstm_hidden, fc_hidden)
        
        # FC for trace features
        self.fc_trace = nn.Linear(num_trace_features, fc_hidden)
        
        # Final output layer
        self.fc_output = nn.Linear(fc_hidden * 4, num_output_labels)
        
        # LayerNorm over concatenated hidden vectors
        self.layer_norm = nn.LayerNorm(fc_hidden * 4)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, prefixes):
        """
        prefixes: N x T x P
        assume feature order: [activity, resource, month, trace features...]
        """
        x_act = prefixes[..., 0].long()
        x_res = prefixes[..., 1].long()
        x_month = prefixes[..., 2].long()
        x_trace = prefixes[..., 3:]
        
        # Embeddings
        emb_act = self.embed_act(x_act)    # N x T x emb_dim
        emb_res = self.embed_res(x_res)
        emb_month = self.embed_month(x_month)
        
        # LSTM forward (take last hidden state)
        _, (h_act, _) = self.lstm_activity(emb_act)
        _, (h_res, _) = self.lstm_resource(emb_res)
        _, (h_month, _) = self.lstm_month(emb_month)
        
        h_act = h_act[-1]   # (N, hidden)
        h_res = h_res[-1]
        h_month = h_month[-1]
        
        # FC for each
        h_fc_act = self.fc_lstm(h_act)
        h_fc_res = self.fc_lstm(h_res)
        h_fc_month = self.fc_lstm(h_month)
        h_fc_trace = self.fc_trace(x_trace)
        
        # Concatenate all
        h_comb = torch.cat([h_fc_act, h_fc_res, h_fc_month, h_fc_trace], dim=-1)
        
        # Final processing
        x = self.layer_norm(h_comb)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        out = self.fc_output(x)
        out = self.sigmoid(out)
        
        return out
