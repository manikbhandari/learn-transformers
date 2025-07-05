"""
Source: https://github.com/karpathy/llm.c/blob/master/dev/data/data_common.py
Common utilities for the datasets
"""

import requests
from tqdm import tqdm
import numpy as np
import typing as T


def download_file(url: str, fname: str, chunk_size: int = 1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


HEADERS_INFO = {
    "gpt-2": {
        "magic": 20240520,
        "version": 1,
        "token_dtype": np.uint16,  # i.e. 2 bytes each
    },
}


def write_datafile(filename: str, toks: T.List[int], model_desc="gpt-2"):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s.
      Header is needed to be able to peak into the file and get some basic info about the data.
    - The tokens follow, each as uint16 (gpt-2)
    """
    assert len(toks) < 2**31, "token count too large"  # ~2.1B tokens
    assert model_desc in ["gpt-2"], f"unknown model descriptor {model_desc}"
    info = HEADERS_INFO[model_desc]
    # construct the header
    header = np.zeros(256, dtype=np.int32)  # header is always 256 int32 values
    header[0] = info["magic"]
    header[1] = info["version"]
    header[2] = len(toks)  # number of tokens after the 256*4 bytes of header
    # construct the data (numpy array of tokens)
    toks_np = np.array(toks, dtype=info["token_dtype"])
    # write to file
    num_bytes = (256 * 4) + (len(toks) * toks_np.itemsize)
    print(
        f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes) in the {model_desc} format"
    )
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())
