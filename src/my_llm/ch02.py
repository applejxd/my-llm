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


@app.cell
def _():
    mo.md(r"""
    Define a downloader function for the novel "The Verdict" that allows to be used for traning.
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
    mo.md(r"""
    Download and read the data.
    """)
    return


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
    mo.md(r"""
    Split words by whitespace using regex.
    """)
    return


@app.cell
def _():
    text_1 = "Hello, world. This, is a test."
    _result = re.split(r"(\s)", text_1)
    print(_result)
    return (text_1,)


@app.cell
def _():
    mo.md(r"""
    Add period (.) and comma (,) separaters for it.
    """)
    return


@app.cell
def _(text_1):
    result_1 = re.split(r"([,.]|\s)", text_1)
    print(result_1)
    return (result_1,)


@app.cell
def _():
    mo.md(r"""
    Delete whitespaces that is recognized as a separater if it is needed.
    Strip whitespace from each item and then filter out any empty strings.
    """)
    return


@app.cell
def _(result_1):
    _result = [item for item in result_1 if item.strip()]
    print(_result)
    return


@app.cell
def _():
    mo.md(r"""
    Try another example includes symbols like hyphen (-) or question mark (?). These symbols also can be extracted by treating these as separaters.
    """)
    return


@app.cell
def _():
    text_2 = "Hello, world. Is this-- a test?"

    _result = re.split(r'([,.:;?_!"()\']|--|\s)', text_2)
    _result = [item.strip() for item in _result if item.strip()]
    print(_result)
    return


@app.cell
def _():
    mo.md(r"""
    Split The Verdict data whole.
    """)
    return


@app.cell
def _(raw_text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(f"{preprocessed[:30]=}")
    print(f"{len(preprocessed)=}")
    return (preprocessed,)


@app.cell
def _():
    mo.md(r"""
    ## 2.3 Converting tokens into token IDs
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Create unique list of tokens by using `set`, then sort it.
    """)
    return


@app.cell
def _(preprocessed):
    all_words = sorted(set(preprocessed))
    _vocab_size = len(all_words)
    print(_vocab_size)
    return (all_words,)


@app.cell
def _():
    mo.md(r"""
    Create map (dictionary) from token to token ID, then show it.
    """)
    return


@app.cell
def _(all_words):
    vocab = {token: integer for integer, token in enumerate(all_words)}
    for _i, _item in enumerate(vocab.items()):
        print(_item)
        if _i >= 50:
            break
    return (vocab,)


@app.cell
def _():
    mo.md(r"""
    Define a simple tokenizer based on these, and implement not only encoder but also decoder by defining the inverse mapping.
    """)
    return


@app.class_definition
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        # inverse mapping
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        # Split by specified punctuations and whitespace
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        # Remove empty strings and whitespace-only strings
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Map to token IDs
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        # Map token IDs back to strings by joining with spaces
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


@app.cell
def _():
    mo.md(r"""
    Apply the encoder process to a sentence from The Verdict.
    """)
    return


@app.cell
def _(vocab):
    simple_tokenizer = SimpleTokenizerV1(vocab)

    text_3 = """"It's the last he painted, you know," 
               Mrs. Gisburn said with pardonable pride."""
    token_ids_3 = simple_tokenizer.encode(text_3)
    print(token_ids_3)
    return simple_tokenizer, text_3, token_ids_3


@app.cell
def _():
    mo.md(r"""
    Try decoding it to check whether it is invertive.
    """)
    return


@app.cell
def _(simple_tokenizer, token_ids_3):
    simple_tokenizer.decode(token_ids_3)
    return


@app.cell
def _():
    mo.md(r"""
    Combine these.
    """)
    return


@app.cell
def _(simple_tokenizer, text_3):
    simple_tokenizer.decode(simple_tokenizer.encode(text_3))
    return


@app.cell
def _():
    mo.md(r"""
    We cannot treat unknown words that are not included in the vocabulary. This will occur an error.
    """)
    return


@app.cell
def _(simple_tokenizer):
    text_4 = "Hello, do you like tea. Is this-- a test?"

    simple_tokenizer.encode(text_4)
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
    To fix the vocabulary issues, the special token for unknown words should be added as `unk`. The special token to combine multiple texts is also added as `endoftext`.
    """)
    return


@app.cell
def _(preprocessed):
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])

    vocab_extended = {token: integer for integer, token in enumerate(all_tokens)}
    return (vocab_extended,)


@app.cell
def _():
    mo.md(r"""
    The total number of the vocabulary is increased $1130+2=1132$.
    """)
    return


@app.cell
def _(vocab_extended):
    len(vocab_extended.items())
    return


@app.cell
def _():
    mo.md(r"""
    Check the tail.
    """)
    return


@app.cell
def _(vocab_extended):
    for _i, _item in enumerate(list(vocab_extended.items())[-5:]):
        print(_item)
    return


@app.cell
def _():
    mo.md(r"""
    This is updated simple tokenier includes the special tokens.
    """)
    return


@app.class_definition
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Handle unknown tokens
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
def _():
    mo.md(r"""
    Try to tokenize concatenated texts by the special token.
    """)
    return


@app.cell
def _(vocab_extended):
    tokenizer_extended = SimpleTokenizerV2(vocab_extended)

    _text1 = "Hello, do you like tea?"
    _text2 = "In the sunlit terraces of the palace."

    text_extended = " <|endoftext|> ".join((_text1, _text2))

    print(text_extended)
    return text_extended, tokenizer_extended


@app.cell
def _():
    mo.md(r"""
    Then, tokenize it. You will see token ID 1130 and 1131 that are asigned for the special tokens.
    """)
    return


@app.cell
def _(text_extended, tokenizer_extended):
    tokenizer_extended.encode(text_extended)
    return


@app.cell
def _():
    mo.md(r"""
    Decode it to check the tokenization.
    """)
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
    mo.md(r"""
    To avoid unknown tokens, introduce byte pair encoding (BPE) by using tiktoken.
    """)
    return


@app.cell
def _():
    print("tiktoken version:", importlib.metadata.version("tiktoken"))
    return


@app.cell
def _():
    mo.md(r"""
    Try the tiktoken tokenizer for GPT2 model. The max token ID 50256 is assigned for `endoftext`.
    """)
    return


@app.cell
def _():
    bpe_tokenizer = tiktoken.get_encoding("gpt2")

    text_5 = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
         "of someunknownPlace."
    )

    bpe_token_id_5 = bpe_tokenizer.encode(text_5, allowed_special={"<|endoftext|>"})

    print(bpe_token_id_5)
    return bpe_token_id_5, bpe_tokenizer


@app.cell
def _():
    mo.md(r"""
    Check it by using the decoding. It encodes all tokens in unique manner by using subworkds and the combined words. So, this is invertive.
    """)
    return


@app.cell
def _(bpe_token_id_5, bpe_tokenizer):
    _strings = bpe_tokenizer.decode(bpe_token_id_5)
    print(_strings)
    return


@app.cell
def _():
    mo.md(r"""
    To know BPE, see [this](../../third_party/LLMs-from-scratch/ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb).
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 2.6 Data sampling with a sliding window
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Use BPE tokenizer for tokenization of The Verdict, and show the size of the all tokens.
    """)
    return


@app.cell
def _(bpe_tokenizer, raw_text):
    enc_text = bpe_tokenizer.encode(raw_text)
    print(len(enc_text))
    return (enc_text,)


@app.cell
def _():
    mo.md(r"""
    Delete the beginning for a later demonstration.
    """)
    return


@app.cell
def _(enc_text):
    enc_sample = enc_text[50:]
    return (enc_sample,)


@app.cell
def _():
    mo.md(r"""
    Create learning pairs to predict next tokens. The task is predicting the `y` from the `x`.
    """)
    return


@app.cell
def _(enc_sample):
    context_size = 4

    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]

    print(f"x: {x}")
    print(f"y:      {y}")
    return (context_size,)


@app.cell
def _():
    mo.md(r"""
    The mappings will be learned are these.
    """)
    return


@app.cell
def _(context_size, enc_sample):
    for _i in range(1, context_size+1):
        _context = enc_sample[:_i]
        _desired = enc_sample[_i]

        print(_context, "---->", _desired)
    return


@app.cell
def _():
    mo.md(r"""
    Decode these token IDs to check the words.
    """)
    return


@app.cell
def _(bpe_tokenizer, context_size, enc_sample):
    for _i in range(1, context_size+1):
        _context = enc_sample[:_i]
        _desired = enc_sample[_i]

        print(bpe_tokenizer.decode(_context), "---->", bpe_tokenizer.decode([_desired]))
    return


@app.cell
def _():
    mo.md(r"""
    Let's define dataset for this by using PyTorch.
    """)
    return


@app.cell
def _():
    print("PyTorch version:", torch.__version__)
    return


@app.cell
def _():
    mo.md(r"""
    To define PyTorch dataset, we need to define special methods `__len__` and `__getitem__` for iterater use. We implement the indexing for tokenized tensor with chunks whose size is `max_length` with `stride` step size (sliding window).
    """)
    return


@app.class_definition
class GPTDatasetV1(Dataset):
    def __init__(self, txt, bpe_tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = bpe_tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
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


@app.cell
def _():
    mo.md(r"""
    Define a function to instantiate the dataloader class.
    """)
    return


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
    bpe_tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, bpe_tokenizer, max_length, stride)

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
def _():
    mo.md(r"""
    For debugging, define a decoder function for batches.
    """)
    return


@app.function
def decode_batch(batch, tokenizer):
    inputs, targets = batch
    batch_size, seq_length = inputs.shape

    decoded_inputs = []
    decoded_targets = []

    for i in range(batch_size):
        input_ids = inputs[i].tolist()
        target_ids = targets[i].tolist()

        decoded_input = tokenizer.decode(input_ids)
        decoded_target = tokenizer.decode(target_ids)

        decoded_inputs.append(decoded_input)
        decoded_targets.append(decoded_target)

    return decoded_inputs, decoded_targets


@app.cell
def _():
    mo.md(r"""
    Call the function to check the data batches. The output should be the pair of input IDs and target IDs.
    """)
    return


@app.cell
def _(bpe_tokenizer, raw_text):
    _dataloader_s1 = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )

    data_iter_s1 = iter(_dataloader_s1)
    _first_batch = next(data_iter_s1)
    print(f"{_first_batch=}")

    _inputs_decoded, _targets_decoded = decode_batch(_first_batch, bpe_tokenizer)
    print(f"{_inputs_decoded=}")
    print(f"{_targets_decoded=}")
    return (data_iter_s1,)


@app.cell
def _():
    mo.md(r"""
    Iterate again and check the next batch. You will see only 1 shifted tokens from the previous because of `stride=1`.
    """)
    return


@app.cell
def _(bpe_tokenizer, data_iter_s1):
    _second_batch = next(data_iter_s1)
    print(f"{_second_batch=}")

    _inputs_decoded, _targets_decoded = decode_batch(_second_batch, bpe_tokenizer)
    print(f"{_inputs_decoded=}")
    print(f"{_targets_decoded=}")
    return


@app.cell
def _():
    mo.md(r"""
    Try larger batch size (`batch_size=8`) for robust training. To avoid overfitting, the stride is also increased (`stride=4`).
    """)
    return


@app.cell
def _(bpe_tokenizer, raw_text):
    _dataloader_s4 = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
    )

    data_iter_s4 = iter(_dataloader_s4)
    _inputs, _targets = next(data_iter_s4)
    print("Inputs:\n", _inputs)
    print("\nTargets:\n", _targets)

    _inputs_decoded, _targets_decoded = decode_batch((_inputs, _targets), bpe_tokenizer)
    print(f"{_inputs_decoded=}")
    print(f"{_targets_decoded=}")
    return


@app.cell
def _():
    mo.md(r"""
    ## 2.7 Creating token embeddings
    """)
    return


@app.cell
def _():
    mo.md(r"""
    This is an example for token embeddings that accepts 6-dimensional input as the vocabulary and outputs 3-dimensional embedding vector.

    The embedding is represented by a matrix whose rows represents embedding vectors for each vocabularies.

    The representation matrix can be trained.
    """)
    return


@app.cell
def _():
    _vocab_size = 6
    _output_dim = 3

    torch.manual_seed(123)
    # randomly initizalize embedding layer
    embedding_layer = torch.nn.Embedding(_vocab_size, _output_dim)
    print(embedding_layer.weight)
    return (embedding_layer,)


@app.cell
def _():
    mo.md(r"""
    Let's try the embedding for token ID 3.
    """)
    return


@app.cell
def _(embedding_layer):
    print(embedding_layer(torch.tensor([3])))
    return


@app.cell
def _():
    mo.md(r"""
    Try more embeddings. This is just referring the matrix as LUT (Look Up Table).
    """)
    return


@app.cell
def _(embedding_layer):
    _input_ids = torch.tensor([2, 3, 5, 1])
    print(embedding_layer(_input_ids))
    return


@app.cell
def _():
    mo.md(r"""
    ## 2.8 Encoding word positions
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Try positional encoding based on this configurations.
    """)
    return


@app.cell
def _():
    _vocab_size = 50257
    output_dim = 256

    token_embedding_layer = torch.nn.Embedding(_vocab_size, output_dim)
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
def _():
    mo.md(r"""
    Then, these tokens are embedded to 256-dimensional manifold.
    """)
    return


@app.cell
def _(inputs, token_embedding_layer):
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)

    # uncomment & execute the following line to see how the embeddings look like
    # print(token_embeddings)
    return (token_embeddings,)


@app.cell
def _():
    mo.md(r"""
    Absolute positional embedding is also proceeded by using another `torch.nn.Embedding` layer.
    """)
    return


@app.cell
def _(max_length, output_dim):
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    # uncomment & execute the following line to see how the embedding layer weights look like
    print(pos_embedding_layer.weight)
    return (pos_embedding_layer,)


@app.cell
def _():
    mo.md(r"""
    Then, indexes represents absolute positions are input to it. This can be also interpreted as LUT.
    """)
    return


@app.cell
def _(max_length, pos_embedding_layer):
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    print(pos_embeddings.shape)

    # uncomment & execute the following line to see how the embeddings look like
    print(pos_embeddings) 
    return (pos_embeddings,)


@app.cell
def _():
    mo.md(r"""
    The absolute positional encoding is just adding these.
    """)
    return


@app.cell
def _(pos_embeddings, token_embeddings):
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)

    # uncomment & execute the following line to see how the embeddings look like
    print(input_embeddings)
    return


if __name__ == "__main__":
    app.run()
