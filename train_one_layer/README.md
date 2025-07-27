
## Bare bones implementation
See [train_one_layer.py](https://github.com/manikbhandari/learn-transformers/blob/main/train_one_layer.py)

Debug log:
```
python train_one_layer.py                     

TOKEN_IDS=[0, 1, 2, 3, 4, 5, 3, 6, 7, 8]
self.in_embed.weight.shape=torch.Size([9, 16])
self.pos_embed.weight.shape=torch.Size([50, 16])
self.lm_head.weight.shape=torch.Size([9, 16])
--------------------------------------------------------------------------------
in_embed.weight torch.Size([9, 16])
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
n_total_params=3920
token_ids.shape=torch.Size([9])
x.shape=torch.Size([9, 16])
pos.shape=torch.Size([9])
pos_embed.shape=torch.Size([9, 16])
q.shape=torch.Size([9, 16])
q.shape=torch.Size([8, 9, 2])
k.shape=torch.Size([9, 16])
k.shape=torch.Size([8, 9, 2])
v.shape=torch.Size([9, 16])
v.shape=torch.Size([8, 9, 2])
mha.shape=torch.Size([8, 9, 9])
mask.shape=torch.Size([1, 9, 9])
mha.shape=torch.Size([8, 9, 9])
mha[0, :3, :3]=tensor([[1.0000, 0.0000, 0.0000],
        [0.1869, 0.8131, 0.0000],
        [0.1162, 0.7096, 0.1743]], grad_fn=<SliceBackward0>)
attn.shape=torch.Size([8, 9, 2])
attn_torch.shape=torch.Size([8, 9, 2])
attn.shape=torch.Size([9, 16])
x.shape=torch.Size([9, 16])
y.shape=torch.Size([9, 9])
loss: 1.8950
```
