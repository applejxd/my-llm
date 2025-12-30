import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import requests
    import tiktoken
    import torch
    from torch.utils.data import DataLoader, Dataset


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
    mo.md(r"""
    ## 2.6 Data sampling with a sliding window
    """)
    return


@app.class_definition
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


@app.function
def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

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
        num_workers=num_workers
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
    _dataloader_s4 = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

    data_iter_s4 = iter(_dataloader_s4)
    inputs, targets = next(data_iter_s4)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
