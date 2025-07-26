"""
Trains a single transformer layer on a single sentence. 
We verify that the implementation is good by 
    (1) checking that the trainig loss goes to 0 
    (2) ensureing that the attention computed is exactly the same as pytorch's computation

python train_one_layer.py

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


DEBUG = True

def print0(s):
    if DEBUG:
        print(s)


class GPTOneLayer(nn.Module):
    def __init__(self, n_vocab, n_pos, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(num_embeddings=n_vocab, embedding_dim=embed_dim)
        print(f"{self.in_embed.weight.shape=}")

        self.pos_embed = nn.Embedding(num_embeddings=n_pos, embedding_dim=embed_dim)
        print(f"{self.pos_embed.weight.shape=}")

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        mask = torch.tril(torch.ones(n_pos, n_pos))
        self.register_buffer("mask", mask)
        self.ln1 = nn.LayerNorm(embed_dim)

        self.mlp1 = nn.Linear(embed_dim, embed_dim * 4)
        self.mlp2 = nn.Linear(embed_dim * 4, embed_dim)

        self.lm_head = nn.Linear(embed_dim, n_vocab, bias=False)
        print(f"{self.lm_head.weight.shape=}")
        
        self.lm_head.weight = self.in_embed.weight

    def forward(self, token_ids):
        print0(f"{token_ids.shape=}")

        x = self.in_embed(token_ids)
        print0(f"{x.shape=}")

        pos = torch.arange(x.shape[0])
        print0(f"{pos.shape=}")
        pos_embed = self.pos_embed(pos)
        print0(f"{pos_embed.shape=}")
        x += pos_embed

        q = self.wq(x)
        print0(f"{q.shape=}")
        q = q.view(-1, 8, q.shape[-1] // 8).transpose(0, 1)
        print0(f"{q.shape=}")

        k = self.wk(x)
        print0(f"{k.shape=}")
        k = k.view(-1, 8, k.shape[-1] // 8).transpose(0, 1)
        print0(f"{k.shape=}")

        v = self.wv(x)
        print0(f"{v.shape=}")
        v = v.view(-1, 8, v.shape[-1] // 8).transpose(0, 1)
        print0(f"{v.shape=}")

        mha = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        print0(f"{mha.shape=}")

        mask = self.mask[:mha.shape[1], :mha.shape[1]][None, ...]
        print0(f"{mask.shape=}")

        mha = torch.where(mask == 1, mha, float("-inf"))
        print0(f"{mha.shape=}")

        mha = F.softmax(mha, dim=-1)
        print0(f"{mha[0, :3, :3]=}")

        attn = torch.matmul(mha, v)
        print0(f"{attn.shape=}")

        attn_torch = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print0(f"{attn_torch.shape=}")
        if DEBUG:
            assert torch.all(torch.isclose(attn_torch.data, attn.data))

        attn = attn.transpose(0, 1).contiguous().view(x.shape[0], -1)
        print0(f"{attn.shape=}")

        x = self.ln1(x + attn)

        x = self.mlp2(F.gelu(self.mlp1(x)))
        print0(f"{x.shape=}")

        y = self.lm_head(x)
        print0(f"{y.shape=}")

        return y
        

class BasicTokenizer:
    def __init__(self,):
        self.token2id = dict()
        self.id2token = dict()

    def add_to_vocab(self, token: str):
        if token in self.token2id:
            return

        new_id = len(self.token2id)
        self.token2id[token] = new_id
        self.id2token[new_id] = token
        assert len(self.token2id) == len(self.id2token)

    def tokenize(self, s):
        tokens = s.split()
        for token in tokens:
            self.add_to_vocab(token)
        return [self.token2id[token] for token in tokens]

    def detokenize(self, token_ids):
        return " ".join([self.id2token[token_id] for token_id in token_ids])


if __name__ == "__main__":
    tokenizer = BasicTokenizer()
    DATA = "Hello, world. I'm trying to learn how to build a transformer. transformer is a great tool world."
    assert DATA == tokenizer.detokenize(tokenizer.tokenize(DATA))

    TOKEN_IDS = tokenizer.tokenize(s=DATA)
    print(f"{TOKEN_IDS=}")

    EMBED_DIM = 16
    token_ids = torch.LongTensor(TOKEN_IDS)
    targets = token_ids[1:]
    input_seq = token_ids[:-1]
    assert input_seq.shape == targets.shape, f"{input_seq.shape=}, {targets.shape=}"

    model = GPTOneLayer(n_vocab=len(tokenizer.token2id), n_pos=50, embed_dim=EMBED_DIM)
    LR = 0.0001
    print("-" * 80)
    for param in model.named_parameters():
        print(param[0], param[1].shape)
    print("-" * 80)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    N_STEPS = 10_000 if not DEBUG else 1
    for step in range(N_STEPS):
        model.zero_grad(None)
        preds = model(input_seq)

        loss = F.cross_entropy(preds, targets)

        if (step + 1) % 100 == 0:
            print(f"step: {step} | loss: {loss.data:.4f}")

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    print(f"loss: {loss.data:.4f}")