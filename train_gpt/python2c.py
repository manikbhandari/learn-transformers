# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files


def write_fp32(tensor, file):
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.write(b)


def write_bf16(tensor, file):
    t = tensor.detach().cpu().to(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16)  # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.write(b)


def write_tensors(model_tensors, L, file, dtype):
    # writes the GPT-2 model's weights to a binary file
    assert dtype in {"float32", "bfloat16"}
    write_fun = write_fp32 if dtype == "float32" else write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file)  # (V, C)
    write_fun(model_tensors["transformer.wpe.weight"], file)  # (T, C)
    for i in range(L):  # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    for i in range(L):  # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    for i in range(L):  # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    for i in range(L):  # (L, 3C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    for i in range(L):  # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    for i in range(L):  # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    for i in range(L):  # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    for i in range(L):  # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    for i in range(L):  # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    for i in range(L):  # (L, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    for i in range(L):  # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    for i in range(L):  # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file)  # (C, )
    write_fun(model_tensors["transformer.ln_f.bias"], file)  # (C, )


@torch.no_grad()
def pad_vocab(tensor, multiple=128, value=0):
    """
    The dimension of the vocab size in GPT-2 is 50,257
    which is unfortunately a very unfriendly number for a lot of
    matrix operations on the GPU. So we pad it to the nearest
    friendlier multiple, e.g. 50,304 if multiple=128 when we
    export the weights into C land. This is a NOOP algorithmically
    and is only done to make the tensor operations more efficient.
    """
    assert tensor.ndim == 2
    V, C = tensor.shape
    assert V == 50257, "just being defensive here"
    # calculate padded vocab size by rounding up to nearest multiple
    Vp = ((V + multiple - 1) // multiple) * multiple
    # pad the tensor
    pad_rows = Vp - V
    padded = (
        tensor if pad_rows == 0 else F.pad(tensor, (0, 0, 0, pad_rows), value=value)
    )
    assert padded.shape == (Vp, C)
    return padded


def write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, GPTConfig ints, padding to 1024 bytes
    assert dtype in {"float32", "bfloat16"}  # float16 todo maybe later
    version = {
        "float32": 3,  # 3: all tensors are fp32, padded vocab
        "bfloat16": 5,  # 5: all tensors are bf16, padded vocab
    }[dtype]
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240326  # magic
    header[1] = version  # checkpoint version
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_embd
    # 2) the parameters follow the header
    params = {name: param.cpu() for name, param in model.named_parameters()}
    # pad the vocab to a multiple of 128 here at export, for efficiency in C
    wte = params["transformer.wte.weight"]  # (V, C)
    wte_padded = pad_vocab(wte)  # (Vp, C)
    params["transformer.wte.weight"] = wte_padded  # (Vp, C)
    print(f"padded vocab size from {wte.size(0)} to {wte_padded.size(0)}")
    header[7] = wte_padded.size(0)  # padded vocab size store in header
    # now write to file
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes())  # header
        write_tensors(params, model.config.n_layer, file, dtype)  # params
    print(f"wrote {filename}")


def write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240327  # magic
    header[1] = 2  # run state version = 2 (1 -> 2 for padded vocab changes)
    header[2] = x.size(0)  # batch size of the batch, B
    header[3] = x.size(1)  # temporal extent of the batch, T
    grads = {name: param.grad.cpu() for name, param in model.named_parameters()}
    # pad the vocab grads here as well, to mirror write_model
    wte_grad = grads["transformer.wte.weight"]  # (V, C)
    wte_grad_padded = pad_vocab(
        wte_grad, value=0
    )  # (Vp, C) # TODO later maybe pad with nan?
    grads["transformer.wte.weight"] = wte_grad_padded  # (Vp, C)
    print(
        f"padded vocab size in reference grads from {wte_grad.size(0)} to {wte_grad_padded.size(0)}"
    )
    with open(filename, "wb") as file:
        # header
        file.write(header.numpy().tobytes())
        # input x
        file.write(x.cpu().numpy().astype("int32").tobytes())  # (B, T)
        # targets y
        file.write(y.cpu().numpy().astype("int32").tobytes())  # (B, T)
        # logits (result of the model forward pass)
        write_fp32(logits.cpu(), file)
        # loss (single float, result of the cross entropy loss)
        write_fp32(loss.cpu(), file)
        # gradients
        write_tensors(grads, model.config.n_layer, file, "float32")
    print(f"wrote {filename}")


def write_tokenizer(enc, filename):
    n = enc.max_token_value + 1
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240328  # magic
    header[1] = 2  # tokenizer version = 2 (1 -> 2: includes EOT token)
    header[2] = n  # number of tokens
    header[3] = enc.eot_token  # EOT token
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes())
        for i in range(n):
            b = enc.decode_bytes([i])
            length = len(b)
            assert length < 256, f"Token length exceeds 255: {length}"
            file.write(
                struct.pack("<B", length)
            )  # Write the length as a 1-byte unsigned integer
            file.write(b)  # Write the actual bytes
    print(f"wrote {filename}")


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    if master_process and args.write_tensors and (not args.inference_only):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in both float32 and bfloat16
        model_to_size = {
            "gpt2": "124M",
            "gpt2-medium": "355M",
            "gpt2-large": "774M",
            "gpt2-xl": "1558M",
        }
        model_to_size.update({f"d{d}": f"d{d}" for d in [12, 24, 36, 48]})
        model_size_str = model_to_size[args.model]  # e.g. "124M", or "d12"
        write_model(model, f"gpt2_{model_size_str}.bin", dtype="float32")
        write_model(model, f"gpt2_{model_size_str}_bf16.bin", dtype="bfloat16")
        # save x, y, logits, loss, and parameter gradients, for debugging C
        # always store these in fp32 to have an accurate reference (?)
        write_state(model, x, y, logits, loss, f"gpt2_{model_size_str}_debug_state.bin")
        # reset the train_loader for the optimization below
        train_loader.reset()
