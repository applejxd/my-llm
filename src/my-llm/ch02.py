import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    import importlib
    import re
    from importlib.metadata import version
    from pathlib import Path

    import marimo as mo
    import requests
    import tiktoken
    import torch
    from torch.utils.data import DataLoader, Dataset


@app.cell
def _():
    print("torch version:", version("torch"))
    print("tiktoken version:", version("tiktoken"))
    return


@app.cell
def _():
    mo.md(r"""
    # Chapter 2: Working with Text Data
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 2.2 Tokenizing text
    """)
    return


@app.function
def download_verdict_data():
    project_root = Path(__file__).parent.parent.parent
    data_root = project_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    file_path = data_root / "the-verdict.txt"
    if file_path.exists():
        print(f"Data file already exists at {file_path}. Skipping download.")
    else:
        data_url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        response = requests.get(data_url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    return file_path


@app.cell
def _():
    verdict_file_path = download_verdict_data()
    with open(verdict_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Total number of character:", len(raw_text))
    print(raw_text[:99])
    return (raw_text,)


@app.cell
def _():
    text_1 = "Hello, world. This, is a test."
    _result = re.split(r"(\s)", text_1)

    print(_result)
    return (text_1,)


@app.cell
def _(text_1):
    result = re.split(r"([,.]|\s)", text_1)

    print(result)
    return (result,)


@app.cell
def _(result):
    # Strip whitespace from each item and then filter out any empty strings.
    _result = [item for item in result if item.strip()]
    print(_result)
    return


@app.cell
def _():
    text_2 = "Hello, world. Is this-- a test?"

    _result = re.split(r'([,.:;?_!"()\']|--|\s)', text_2)
    _result = [item.strip() for item in _result if item.strip()]
    print(_result)
    return


@app.cell
def _(raw_text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(preprocessed[:30])
    return (preprocessed,)


@app.cell
def _(preprocessed):
    print(len(preprocessed))
    return


@app.cell
def _():
    mo.md(r"""
    ## 2.3 Converting tokens into token IDs
    """)
    return


@app.cell
def _(preprocessed):
    all_words = sorted(set(preprocessed))
    _vocab_size = len(all_words)

    print(_vocab_size)
    return (all_words,)


@app.cell
def _(all_words):
    vocab = {token: integer for integer, token in enumerate(all_words)}
    return (vocab,)


@app.cell
def _(vocab):
    for _i, _item in enumerate(vocab.items()):
        print(_item)
        if _i >= 50:
            break
    return


@app.class_definition
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


@app.cell
def _(vocab):
    simple_tokenizer = SimpleTokenizerV1(vocab)

    text_3 = """"It's the last he painted, you know," 
               Mrs. Gisburn said with pardonable pride."""
    ids = simple_tokenizer.encode(text_3)
    print(ids)
    return ids, simple_tokenizer, text_3


@app.cell
def _(ids, simple_tokenizer):
    simple_tokenizer.decode(ids)
    return


@app.cell
def _(simple_tokenizer, text_3):
    simple_tokenizer.decode(simple_tokenizer.encode(text_3))
    return


@app.cell
def _():
    mo.md(r"""
    ## 2.4 Adding special context tokens
    """)
    return


@app.cell
def _():
    mo.md(r"""
    This will occur an error.
    """)
    return


@app.cell
def _(simple_tokenizer):
    text_4 = "Hello, do you like tea. Is this-- a test?"

    simple_tokenizer.encode(text_4)
    return


@app.cell
def _(preprocessed):
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])

    vocab_extended = {token: integer for integer, token in enumerate(all_tokens)}
    return (vocab_extended,)


@app.cell
def _(vocab_extended):
    len(vocab_extended.items())
    return


@app.cell
def _(vocab_extended):
    for _i, _item in enumerate(list(vocab_extended.items())[-5:]):
        print(_item)
    return


@app.class_definition
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
        return text


@app.cell
def _(vocab_extended):
    tokenizer_extended = SimpleTokenizerV2(vocab_extended)

    _text1 = "Hello, do you like tea?"
    _text2 = "In the sunlit terraces of the palace."

    text_extended = " <|endoftext|> ".join((_text1, _text2))

    print(text_extended)
    return text_extended, tokenizer_extended


@app.cell
def _(text_extended, tokenizer_extended):
    tokenizer_extended.encode(text_extended)
    return


@app.cell
def _(text_extended, tokenizer_extended):
    tokenizer_extended.decode(tokenizer_extended.encode(text_extended))
    return


@app.cell
def _():
    mo.md(r"""
    ## 2.5 BytePair encoding
    """)
    return


@app.cell
def _():
    print("tiktoken version:", importlib.metadata.version("tiktoken"))
    return


@app.cell
def _():
    tokenizer = tiktoken.get_encoding("gpt2")
    return (tokenizer,)


@app.cell
def _(tokenizer):
    text_5 = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
         "of someunknownPlace."
    )

    integers = tokenizer.encode(text_5, allowed_special={"<|endoftext|>"})

    print(integers)
    return (integers,)


@app.cell
def _(integers, tokenizer):
    strings = tokenizer.decode(integers)

    print(strings)
    return


@app.cell
def _():
    mo.md(r"""
    ## 2.6 Data sampling with a sliding window
    """)
    return


@app.cell
def _(raw_text, tokenizer):
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))
    return (enc_text,)


@app.cell
def _(enc_text):
    enc_sample = enc_text[50:]
    return (enc_sample,)


@app.cell
def _(enc_sample):
    context_size = 4

    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]

    print(f"x: {x}")
    print(f"y:      {y}")
    return (context_size,)


@app.cell
def _(context_size, enc_sample):
    for _i in range(1, context_size+1):
        _context = enc_sample[:_i]
        _desired = enc_sample[_i]

        print(_context, "---->", _desired)
    return


@app.cell
def _(context_size, enc_sample, tokenizer):
    for _i in range(1, context_size+1):
        _context = enc_sample[:_i]
        _desired = enc_sample[_i]

        print(tokenizer.decode(_context), "---->", tokenizer.decode([_desired]))
    return


@app.cell
def _():
    print("PyTorch version:", torch.__version__)
    return


@app.class_definition
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, (
            "Number of tokenized inputs must at least be equal to max_length+1"
        )

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


@app.function
def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


@app.cell
def _(raw_text):
    _dataloader_s1 = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )

    data_iter_s1 = iter(_dataloader_s1)
    _first_batch = next(data_iter_s1)
    print(_first_batch)
    return (data_iter_s1,)


@app.cell
def _(data_iter_s1):
    _second_batch = next(data_iter_s1)
    print(_second_batch)
    return


@app.cell
def _(raw_text):
    _dataloader_s4 = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
    )

    data_iter_s4 = iter(_dataloader_s4)
    _inputs, _targets = next(data_iter_s4)
    print("Inputs:\n", _inputs)
    print("\nTargets:\n", _targets)
    return


@app.cell
def _():
    mo.md(r"""
    ## 2.7 Creating token embeddings
    """)
    return


@app.cell
def _():
    input_ids = torch.tensor([2, 3, 5, 1])
    return (input_ids,)


@app.cell
def _():
    _vocab_size = 6
    _output_dim = 3

    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(_vocab_size, _output_dim)
    return (embedding_layer,)


@app.cell
def _(embedding_layer):
    print(embedding_layer.weight)
    return


@app.cell
def _(embedding_layer):
    print(embedding_layer(torch.tensor([3])))
    return


@app.cell
def _(embedding_layer, input_ids):
    print(embedding_layer(input_ids))
    return


@app.cell
def _():
    mo.md(r"""
    ## 2.8 Encoding word positions
    """)
    return


@app.cell
def _():
    vocab_size = 50257
    output_dim = 256

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    return output_dim, token_embedding_layer


@app.cell
def _(raw_text):
    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length,
        stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    return inputs, max_length


@app.cell
def _(inputs):
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)
    return


@app.cell
def _(inputs, token_embedding_layer):
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)

    # uncomment & execute the following line to see how the embeddings look like
    # print(token_embeddings)
    return (token_embeddings,)


@app.cell
def _(max_length, output_dim):
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    # uncomment & execute the following line to see how the embedding layer weights look like
    # print(pos_embedding_layer.weight)
    return (pos_embedding_layer,)


@app.cell
def _(max_length, pos_embedding_layer):
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    print(pos_embeddings.shape)

    # uncomment & execute the following line to see how the embeddings look like
    # print(pos_embeddings)
    return (pos_embeddings,)


@app.cell
def _(pos_embeddings, token_embeddings):
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)

    # uncomment & execute the following line to see how the embeddings look like
    # print(input_embeddings)
    return


if __name__ == "__main__":
    app.run()
