# learn-transformers

See [train-gpt2.py](https://github.com/manikbhandari/learn-transformers/blob/main/train_gpt2.py) for simple code to train GPT from scratch based on https://github.com/karpathy/llm.c/blob/master/train_gpt2.py.


## Bare bones implementation
See [train_one_layer.py](https://github.com/manikbhandari/learn-transformers/blob/main/train_one_layer.py)

Debug log:
```
➜  python train_one_layer.py                     
TOKEN_IDS=[0, 1, 2, 3, 4, 5, 6, 4, 7, 8, 9, 10, 11, 8, 12, 13, 1]
self.in_embed.weight.shape=torch.Size([14, 16])
self.pos_embed.weight.shape=torch.Size([50, 16])
self.lm_head.weight.shape=torch.Size([14, 16])
--------------------------------------------------------------------------------
in_embed.weight torch.Size([14, 16])
pos_embed.weight torch.Size([50, 16])
wq.weight torch.Size([16, 16])
wq.bias torch.Size([16])
wk.weight torch.Size([16, 16])
wk.bias torch.Size([16])
wv.weight torch.Size([16, 16])
wv.bias torch.Size([16])
ln1.weight torch.Size([16])
ln1.bias torch.Size([16])
mlp1.weight torch.Size([64, 16])
mlp1.bias torch.Size([64])
mlp2.weight torch.Size([16, 64])
mlp2.bias torch.Size([16])
--------------------------------------------------------------------------------
token_ids.shape=torch.Size([16])
x.shape=torch.Size([16, 16])
pos.shape=torch.Size([16])
pos_embed.shape=torch.Size([16, 16])
q.shape=torch.Size([16, 16])
q.shape=torch.Size([8, 16, 2])
k.shape=torch.Size([16, 16])
k.shape=torch.Size([8, 16, 2])
v.shape=torch.Size([16, 16])
v.shape=torch.Size([8, 16, 2])
mha.shape=torch.Size([8, 16, 16])
mask.shape=torch.Size([1, 16, 16])
mha.shape=torch.Size([8, 16, 16])
mha[0, :3, :3]=tensor([[1.0000, 0.0000, 0.0000],
        [0.4811, 0.5189, 0.0000],
        [0.2927, 0.2839, 0.4234]], grad_fn=<SliceBackward0>)
attn.shape=torch.Size([8, 16, 2])
attn_torch.shape=torch.Size([8, 16, 2])
attn.shape=torch.Size([16, 16])
x.shape=torch.Size([16, 16])
y.shape=torch.Size([16, 14])
loss: 3.4841
```