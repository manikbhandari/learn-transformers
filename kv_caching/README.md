# KV Caching
See [kv_caching.py](https://github.com/manikbhandari/learn-transformers/blob/main/kv_caching/kv_caching.py) for a simple kv caching implementation.
# Log
```
python kv_caching.py
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
step: 99 | loss: 1.1331
step: 199 | loss: 0.4134
step: 299 | loss: 0.1161
step: 399 | loss: 0.0464
step: 499 | loss: 0.0247
step: 599 | loss: 0.0154
step: 699 | loss: 0.0105
step: 799 | loss: 0.0077
step: 899 | loss: 0.0058
step: 999 | loss: 0.0046
loss: 0.0046
--------------------------------------------------------------------------------
output_tokens=[0, 1, 2, 1, 2, 1, 7, 8, 8, 1]
tokenizer.detokenize(output_tokens)='Hello, world. trying world. trying world. a transformer. transformer. world.'
```
