
from d_transformer import TransformerBlock
from a_layerNorm import LayerNorm
import torch.nn as nn
import torch


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"]) #nn.Embedding(50257,768)
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"]) #nn.Embedding(1024,768)
        self.drop_emb = nn.Dropout(config["drop_rate"]) #nn.Dropout(0.1)
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])])
        
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(
            config["emb_dim"], config["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits