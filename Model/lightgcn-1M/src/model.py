from typing import Tuple

import torch
import torch.nn as nn


class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, num_layers: int):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def propagate(self, norm_adj: torch.sparse.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [all_embeddings]
        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(norm_adj, all_embeddings)
            embs.append(all_embeddings)
        stacked = torch.stack(embs, dim=0)
        mean_emb = torch.mean(stacked, dim=0)
        users, items = torch.split(mean_emb, [self.num_users, self.num_items], dim=0)
        return users, items

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        reg_weight: float,
    ) -> torch.Tensor:
        u_e = user_emb[users]
        p_e = item_emb[pos_items]
        n_e = item_emb[neg_items]

        pos_scores = torch.sum(u_e * p_e, dim=1)
        neg_scores = torch.sum(u_e * n_e, dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

        reg = (u_e.norm(2).pow(2) + p_e.norm(2).pow(2) + n_e.norm(2).pow(2)) / users.size(0)
        return loss + reg_weight * reg

    def full_sort_scores(self, user_emb: torch.Tensor, item_emb: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        u_e = user_emb[users]
        return torch.matmul(u_e, item_emb.t())
