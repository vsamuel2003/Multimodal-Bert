import torch
import torch.nn as nn
import torch.nn.functional as F
from global_configs import *

class MAG(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(MAG, self).__init__()
        print(
            "Initializing MAG with beta_shift:{} hidden_prob:{}".format(
                beta_shift, dropout_prob
            )
        )

        self.W_hv = nn.Linear(VISUAL_DIM + TEXT_DIM, TEXT_DIM)

        self.W_v = nn.Linear(VISUAL_DIM, TEXT_DIM)
        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, text_embedding, visual):
        eps = 1e-6
        
        visual_dim = visual.shape[1]
        text_dim = text_embedding.shape[1]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        projection = nn.Linear(visual_dim, text_dim)
        projection = projection.to(device)
        visual_projected = projection(visual.transpose(1, 2).to(device)) 
        visual_projected = visual_projected.transpose(1, 2) 
        
        weight_v = F.relu(self.W_hv(torch.cat((visual_projected, text_embedding), dim=-1)))

        h_m = weight_v * self.W_v(visual_projected) 

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(DEVICE)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(DEVICE)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )

        return embedding_output
