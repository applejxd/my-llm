import marimo

__generated_with = "0.18.4"
app = marimo.App()

with app.setup:
    from importlib.metadata import version
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import tiktoken
    import torch
    from ch02 import create_dataloader_v1, download_verdict_data
    from ch04 import GPTModel, generate_text_simple
    from gpt_download import download_and_load_gpt2
    from matplotlib.ticker import MaxNLocator

    _pkgs = [
        "matplotlib",
        "numpy",
        "tiktoken",
        "torch",
        "tensorflow",  # For OpenAI's pretrained weights
    ]

    for _p in _pkgs:
        print(f"{_p} version: {version(_p)}")


@app.cell
def _():
    mo.md(r"""
    # Chapter 5: Pretraining on Unlabeled Data
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 5.1 Evaluating generative text models
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### 5.1.1 Using GPT to generate text
    """)
    return


@app.cell
def _():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 256,  # Shortened context length (orig: 1024)
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-key-value bias
    }

    torch.manual_seed(123)
    random_model = GPTModel(GPT_CONFIG_124M)
    random_model.eval();  # Disable dropout during inference
    return GPT_CONFIG_124M, random_model


@app.cell
def _():
    mo.md(r"""
    Utility function to convert from text to token ID
    """)
    return


@app.function
def text_to_token_ids(text, tokenizer):
    # EOT is needed for padding
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


@app.cell
def _():
    mo.md(r"""
    Utility function to convert from token ID to text
    """)
    return


@app.function
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


@app.cell
def _():
    mo.md(r"""
    This is an example to use the above functions.
    The output does not make sense because it is not trained yet.
    """)
    return


@app.cell
def _(GPT_CONFIG_124M, random_model):
    _start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    _token_ids = generate_text_simple(
        model=random_model,
        idx=text_to_token_ids(_start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output text:\n", token_ids_to_text(_token_ids, tokenizer))
    return (tokenizer,)


@app.cell
def _():
    mo.md(r"""
    ### 5.1.2 Calculating the text generation loss: cross-entropy and perplexity
    """)
    return


@app.cell
def _():
    mo.md(r"""
    To show the way to calculate training loss, these inputs and targets are used.
    """)
    return


@app.cell
def _():
    inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                           [40,    1107, 588]])   #  "I really like"]

    targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                            [1107,  588, 11311]]) #  " really like chocolate"]
    return inputs, targets


@app.cell
def _():
    mo.md(r"""
    Get probabilities of next token IDs by using random_model inference
    """)
    return


@app.cell
def _(inputs, random_model):
    with torch.no_grad():
        logits = random_model(inputs)

    probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
    print(f"{probas.shape=}") # Shape: (batch_size, num_tokens, vocab_size)
    return logits, probas


@app.cell
def _():
    mo.md(r"""
    Use greedy way to get next token IDs
    """)
    return


@app.cell
def _(probas):
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids)
    return (token_ids,)


@app.cell
def _():
    mo.md(r"""
    Compare token IDs between targets and predicted
    """)
    return


@app.cell
def _(targets, token_ids, tokenizer):
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
    return


@app.cell
def _():
    mo.md(r"""
    Let's evaluate the differences quantitatively.

    Get the probabilities (likelihood) for targets in the above result.

    Remind the `probas` has (batch_size, num_tokens, vocab_size) shape.
    """)
    return


@app.cell
def _(probas, targets):
    _text_idx = 0
    target_probas_1 = probas[_text_idx, [0, 1, 2], targets[_text_idx]]
    print("Text 1:", target_probas_1)

    _text_idx = 1
    target_probas_2 = probas[_text_idx, [0, 1, 2], targets[_text_idx]]
    print("Text 2:", target_probas_2)
    return target_probas_1, target_probas_2


@app.cell
def _():
    mo.md(r"""
    Take `log` of these to get entropy (or log likelihood).
    """)
    return


@app.cell
def _(target_probas_1, target_probas_2):
    # Compute logarithm of all token probabilities
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)
    return (log_probas,)


@app.cell
def _():
    mo.md(r"""
    Get average scores of it.
    This is called as cross entropy loss.
    """)
    return


@app.cell
def _(log_probas):
    # Calculate the average probability for each token
    avg_log_probas = torch.mean(log_probas)
    print(avg_log_probas)
    return (avg_log_probas,)


@app.cell
def _():
    mo.md(r"""
    Minimization of minus log is preferred than maximization of plus log for implementations.

    This is also called as negative log likelihood (nll).
    """)
    return


@app.cell
def _(avg_log_probas):
    _neg_avg_log_probas = avg_log_probas * -1
    print(_neg_avg_log_probas)
    return


@app.cell
def _():
    mo.md(r"""
    We proceed these calculate by PyTorch.
    Before doing so, check the shapes of logits and targets to compare.
    """)
    return


@app.cell
def _(logits, targets):
    # Logits have shape (batch_size, num_tokens, vocab_size)
    print("Logits shape:", logits.shape)

    # Targets have shape (batch_size, num_tokens)
    print("Targets shape:", targets.shape)
    return


@app.cell
def _():
    mo.md(r"""
    To apply PyTorch function, `flatten` is needed.
    """)
    return


@app.cell
def _(logits, targets):
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()

    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)
    return logits_flat, targets_flat


@app.cell
def _():
    mo.md(r"""
    Then, pass these to `cross_entropy` to take softmax, log, mean and the minus.
    This results coincides the previous result.
    """)
    return


@app.cell
def _(logits_flat, targets_flat):
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(loss)
    return (loss,)


@app.cell
def _():
    mo.md(r"""
    The exponential of cross entropy is called as perplexity.

    If the probabilistic distribution is uniform (the most uncertain case), we can write the perplexity by using the size of vocabulary $N$ as

    $$
    PP
    =\exp\left(-\sum_{i=1}^N\frac{1}{N}\log\frac{1}{N}\right)
    =\exp\left(-\log\frac{1}{N}\right)=N.
    $$

    So it has meaning of uncertainty for prediction in the unit of number of vocabularies.
    """)
    return


@app.cell
def _(loss):
    _perplexity = torch.exp(loss)
    print(_perplexity)
    return


@app.cell
def _():
    mo.md(r"""
    ### 5.1.3 Calculating the training and validation set losses
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Calculate such loss on public and tiny dataset.
    At first, it should be downloaded.
    """)
    return


@app.cell
def _():
    file_path = download_verdict_data()
    with open(file_path, "r", encoding="utf-8") as _file:
        text_data = _file.read()
    return (text_data,)


@app.cell
def _():
    mo.md(r"""
    See the top.
    """)
    return


@app.cell
def _(text_data):
    # First 99 characters
    print(text_data[:99])
    return


@app.cell
def _():
    mo.md(r"""
    See the tail
    """)
    return


@app.cell
def _(text_data):
    # Last 99 characters
    print(text_data[-99:])
    return


@app.cell
def _():
    mo.md(r"""
    See the scale of this dataset. It is tiny and enough to try training trial.
    """)
    return


@app.cell
def _(text_data, tokenizer):
    _total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))

    print("Characters:", _total_characters)
    print("Tokens:", total_tokens)
    return (total_tokens,)


@app.cell
def _():
    mo.md(r"""
    Split the dataset and create dataloaders in
    $$
    \text{train}:\text{valid}=90\%:10\%.
    $$

    The function `create_dataloader_v1` is defined on [the Chapter 2](./ch02.py).
    See the definition in there.
    """)
    return


@app.cell
def _(GPT_CONFIG_124M, text_data):
    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,   # to save computation resources
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    return train_loader, train_ratio, val_loader


@app.cell
def _():
    mo.md(r"""
    Sanity check
    """)
    return


@app.cell
def _(GPT_CONFIG_124M, total_tokens, train_ratio):
    if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the training loader. "
              "Try to lower the `GPT_CONFIG_124M['context_length']` or "
              "increase the `training_ratio`")

    if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the validation loader. "
              "Try to lower the `GPT_CONFIG_124M['context_length']` or "
              "decrease the `training_ratio`")
    return


@app.cell
def _():
    mo.md(r"""
    Try the dataloader to verify the behaviors.
    These are all batches to be used.
    Surely, the number of train batch is 9 and the number of validation batch is 1 that reflects the ratio.
    """)
    return


@app.cell
def _(train_loader, val_loader):
    print("Train loader:")
    for _x, _y in train_loader:
        print(_x.shape, _y.shape)

    print("\nValidation loader:")
    for _x, _y in val_loader:
        print(_x.shape, _y.shape)
    return


@app.cell
def _():
    mo.md(r"""
    Check the total number of tokens. It is just about 5k.
    """)
    return


@app.cell
def _(train_loader, val_loader):
    _train_tokens = 0
    for _input_batch, _target_batch in train_loader:
        _train_tokens += _input_batch.numel()

    _val_tokens = 0
    for _input_batch, _target_batch in val_loader:
        _val_tokens += _input_batch.numel()

    print("Training tokens:", _train_tokens)
    print("Validation tokens:", _val_tokens)
    print("All tokens:", _train_tokens + _val_tokens)
    return


@app.cell
def _():
    mo.md(r"""
    This is a function to calculate cross entropy loss like previous.
    """)
    return


@app.function
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


@app.cell
def _():
    mo.md(r"""
    Extend the functionality by the following function to entire dataloaders.
    By setting the `num_batches` as smaller values, we can compute the losses easier.
    """)
    return


@app.function
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


@app.cell
def _():
    mo.md(r"""
    This is utility function to detect suitable device for training and inference.
    """)
    return


@app.function
def get_torch_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Use PyTorch 2.9 or newer for stable mps results
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")
    return device


@app.cell
def _():
    mo.md(r"""
    This is the first trial to compute the loss for entire dataloaders.
    We need to decrease the loss than these.
    """)
    return


@app.cell
def _(random_model, train_loader, val_loader):
    device = get_torch_device()
    print(f"Using {device} device.")

    random_model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes

    torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

    with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(train_loader, random_model, device)
        val_loss = calc_loss_loader(val_loader, random_model, device)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
    return (device,)


@app.cell
def _():
    mo.md(r"""
    ## 5.2 Training an LLM
    """)
    return


@app.cell
def _():
    mo.md(r"""
    The training process is like this. We need to define the `evaluate_model()` and `generate_and_print_sample()` later
    """)
    return


@app.function
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


@app.cell
def _():
    mo.md(r"""
    This is a function to evaluate loss quantitatively for entire dataloaders.
    """)
    return


@app.function
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()    # disable dropout
    with torch.no_grad():   # skip gradient calculations
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()    # enable dropout
    return train_loss, val_loss


@app.cell
def _():
    mo.md(r"""
    This is a function to evaluate the output qualitatively.
    """)
    return


@app.function
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


@app.cell
def _():
    mo.md(r"""
    We use `AdamW` Optimizer to surpress overfitting.
    """)
    return


@app.cell
def _(GPT_CONFIG_124M, device, tokenizer, train_loader, val_loader):
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )
    return model, num_epochs, optimizer, tokens_seen, train_losses, val_losses


@app.cell
def _():
    mo.md(r"""
    Create directory to save plots for the traning process
    """)
    return


@app.cell
def _(project_root):
    data_root = project_root / "data" / "ch05"
    data_root.mkdir(parents=True, exist_ok=True)
    return (data_root,)


@app.cell
def _():
    mo.md(r"""
    Define the function to plot training processes
    """)
    return


@app.cell
def plot_losses(data_root):
    def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
        fig, ax1 = plt.subplots(figsize=(5, 3))

        # Plot training and validation loss against epochs
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

        # Create a second x-axis for tokens seen
        ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
        ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
        ax2.set_xlabel("Tokens seen")

        fig.tight_layout()  # Adjust layout to make room
        plt.savefig(data_root / "loss-plot.pdf")
        plt.show()
    return (plot_losses,)


@app.cell
def _():
    mo.md(r"""
    The result shows overfitting because of traning with tiny dataset.
    """)
    return


@app.cell
def _(num_epochs, plot_losses, tokens_seen, train_losses, val_losses):
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    return


@app.cell
def _():
    mo.md(r"""
    ## 5.3 Decoding strategies to control randomness
    """)
    return


@app.cell
def _():
    mo.md(r"""
    The `generate_text_simple` is based on greedy decoding strategy and it is deterministic.
    """)
    return


@app.cell
def _(GPT_CONFIG_124M, model, tokenizer):
    # NEW: use CPU here as inference is cheap with 
    # this model and to ensure readers get same results in the
    # remaining sections of this book
    inference_device = torch.device("cpu")

    model.to(inference_device)
    model.eval()

    _token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output text:\n", token_ids_to_text(_token_ids, tokenizer))
    return (inference_device,)


@app.cell
def _():
    mo.md(r"""
    ### 5.3.1 Temperature scaling
    """)
    return


@app.cell
def _():
    mo.md(r"""
    This is an example to show difference between deterministic and probabilistic strategies.
    This outputs deterministic result by greedy decoding.
    """)
    return


@app.cell
def _():
    vocab = { 
        "closer": 0,
        "every": 1, 
        "effort": 2, 
        "forward": 3,
        "inches": 4,
        "moves": 5, 
        "pizza": 6,
        "toward": 7,
        "you": 8,
    } 

    inverse_vocab = {v: k for k, v in vocab.items()}

    # Suppose input is "every effort moves you", and the LLM
    # returns the following logits for the next token:
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )

    next_probas = torch.softmax(next_token_logits, dim=0)
    next_token_id = torch.argmax(next_probas).item()

    # The next generated token is then as follows:
    print(inverse_vocab[next_token_id])
    return inverse_vocab, next_probas, next_token_logits, vocab


@app.cell
def _():
    mo.md(r"""
    Try probabilistic sampling based of probabilities obtained by the logits.
    Even the `forward` occurs in high probability, it also outputs the same word.
    """)
    return


@app.cell
def _(inverse_vocab, next_probas):
    torch.manual_seed(123)
    next_token_id_multinomial = torch.multinomial(next_probas, num_samples=1).item()
    print(inverse_vocab[next_token_id_multinomial])
    return


@app.cell
def _():
    mo.md(r"""
    Try multiple times to see the probabilistic behaviors.
    """)
    return


@app.cell
def _(inverse_vocab):
    def print_sampled_tokens(probas):
        torch.manual_seed(123) # Manual seed for reproducibility
        sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
        sampled_ids = torch.bincount(torch.tensor(sample), minlength=len(probas))
        for i, freq in enumerate(sampled_ids):
            print(f"{freq} x {inverse_vocab[i]}")
    return (print_sampled_tokens,)


@app.cell
def _(next_probas, print_sampled_tokens):
    print_sampled_tokens(next_probas)
    return


@app.cell
def _():
    mo.md(r"""
    To control the probability, introduce temprature scaling like
    $$
    p(x;\beta) \propto \exp\left(\beta p\right)
    $$
    """)
    return


@app.function
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


@app.cell
def _():
    mo.md(r"""
    Try different values of the tempretures.
    """)
    return


@app.cell
def _(next_token_logits):
    # Temperature values
    temperatures = [1, 0.1, 5]  # Original, higher confidence, and lower confidence

    # Calculate scaled probabilities
    scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
    return scaled_probas, temperatures


@app.cell
def _():
    mo.md(r"""
    Plot it. It shows higher temprature make the distribution more uniform.
    """)
    return


@app.cell
def _(data_root, scaled_probas, temperatures, vocab):
    # Plotting
    x = torch.arange(len(vocab))
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(temperatures):
        rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')

    ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()

    plt.tight_layout()
    plt.savefig(data_root / "temperature-plot.pdf")
    plt.show()
    return


@app.cell
def _():
    mo.md(r"""
    `Temperature=0.1` case causes more greedy like distribution.
    """)
    return


@app.cell
def _(print_sampled_tokens, scaled_probas):
    print_sampled_tokens(scaled_probas[1])
    return


@app.cell
def _():
    mo.md(r"""
    `Temperature=5` cause more diversity but more nonsense phrases like `pizza`.
    """)
    return


@app.cell
def _(print_sampled_tokens, scaled_probas):
    print_sampled_tokens(scaled_probas[2])
    return


@app.cell
def _():
    mo.md(r"""
    ### 5.3.2 Top-k sampling
    """)
    return


@app.cell
def _():
    mo.md(r"""
    The top-k sampling strategy is another probabilistic model using cutoff.
    We only use top $k$ candidates in the probabilistic ranking and ignore others.
    """)
    return


@app.cell
def _(next_token_logits):
    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)

    print("Top logits:", top_logits)
    print("Top positions:", top_pos)
    return (top_logits,)


@app.cell
def _():
    mo.md(r"""
    Such ignored candidate logits are masked by -inf.
    """)
    return


@app.cell
def _(next_token_logits, top_logits):
    new_logits = torch.where(
        condition=next_token_logits < top_logits[-1],
        input=torch.tensor(float("-inf")), 
        other=next_token_logits
    )

    print(new_logits)
    return (new_logits,)


@app.cell
def _():
    mo.md(r"""
    The result probabilities are these.
    The predicted token will be sampled base on this result.
    """)
    return


@app.cell
def _(new_logits):
    topk_probas = torch.softmax(new_logits, dim=0)
    print(topk_probas)
    return


@app.cell
def _():
    mo.md(r"""
    ### 5.3.3 Modifying the text generation function
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Include temperature and tok-k sampling to `generate_text_simple()`.
    This generation process continues until the EOS token appears.
    """)
    return


@app.function
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1] 
            # mask other logits
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            # greedy sampling
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


@app.cell
def _():
    mo.md(r"""
    This result is different with the previous.
    """)
    return


@app.cell
def _(GPT_CONFIG_124M, inference_device, model, tokenizer):
    torch.manual_seed(123)

    _token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )

    print("Output text:\n", token_ids_to_text(_token_ids, tokenizer))
    return


@app.cell
def _():
    mo.md(r"""
    ## 5.4 Loading and saving model weights in PyTorch
    """)
    return


@app.cell
def _():
    mo.md(r"""
    We can save the trained moddel like this.
    """)
    return


@app.cell
def _(model):
    project_root = Path(__file__).parent.parent.parent
    model_dir_path = project_root / "models" / "ch05"
    train_model_path = model_dir_path / "model.pth"
    train_model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), train_model_path)
    return model_dir_path, project_root, train_model_path


@app.cell
def _():
    mo.md(r"""
    The loading is also easy.
    """)
    return


@app.cell
def _(GPT_CONFIG_124M, device, train_model_path):
    loaded_model = GPTModel(GPT_CONFIG_124M)
    print("Device:", device)

    loaded_model.load_state_dict(torch.load(train_model_path, map_location=device, weights_only=True))
    loaded_model.eval();    # disable dropout
    return (loaded_model,)


@app.cell
def _():
    mo.md(r"""
    We can also save the optimizer status, and continue the training later.
    """)
    return


@app.cell
def _(loaded_model, model_dir_path, optimizer):
    train_model_optimizer_path = model_dir_path / "model_and_optimizer.pth"
    train_model_optimizer_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": loaded_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        }, 
        train_model_optimizer_path
    )
    return (train_model_optimizer_path,)


@app.cell
def _():
    mo.md(r"""
    Such optimizer status can be loaded as follows.
    """)
    return


@app.cell
def _(GPT_CONFIG_124M, train_model_optimizer_path):
    _checkpoint = torch.load(train_model_optimizer_path, weights_only=True)

    _model = GPTModel(GPT_CONFIG_124M)
    _model.load_state_dict(_checkpoint["model_state_dict"])

    _optimizer = torch.optim.AdamW(_model.parameters(), lr=0.0005, weight_decay=0.1)
    _optimizer.load_state_dict(_checkpoint["optimizer_state_dict"])
    _model.train();
    return


@app.cell
def _():
    mo.md(r"""
    ## 5.5 Loading pretrained weights from OpenAI
    """)
    return


@app.cell
def _():
    mo.md(r"""
    We will use trained GPT2 model from OpenAI.
    It is defined as TensorFlow model, so we need to convert it to PyTorch model.
    """)
    return


@app.cell
def _():
    print("TensorFlow version:", version("tensorflow"))
    print("tqdm version:", version("tqdm"))
    return


@app.cell
def _():
    mo.md(r"""
    The `download_and_load_gpt2` function is defined at [gpt_download.py](./gpt_download.py).
    The code is just downloading files and read the data only.
    """)
    return


@app.cell
def _(project_root):
    models_dir = project_root / "models" / "gpt2"
    models_dir.mkdir(parents=True, exist_ok=True)
    settings, params = download_and_load_gpt2(model_size="124M", models_dir=models_dir)
    return params, settings


@app.cell
def _():
    mo.md(r"""
    These are hyperparameters.
    """)
    return


@app.cell
def _(settings):
    print("Settings:", settings)
    return


@app.cell
def _():
    mo.md(r"""
    These are keys of parameters.
    """)
    return


@app.cell
def _(params):
    print("Parameter dictionary keys:", params.keys())
    return


@app.cell
def _():
    mo.md(r"""
    We can access each parameters like this.
    """)
    return


@app.cell
def _(params):
    print(params["wte"])
    print("Token embedding weight tensor dimensions:", params["wte"].shape)
    return


@app.cell
def _():
    mo.md(r"""
    The files supports several architecture of GPT2. We will use the smallest model.
    """)
    return


@app.cell
def _(GPT_CONFIG_124M):
    # Define model configurations in a dictionary for compactness
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # Copy the base configuration and update with specific model settings
    model_name = "gpt2-small (124M)"  # Example model name
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

    gpt = GPTModel(NEW_CONFIG)
    gpt.eval();
    return NEW_CONFIG, gpt


@app.cell
def _():
    mo.md(r"""
    This is a function to load numerics as model weights with sanity checks.
    """)
    return


@app.function
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


@app.cell
def _():
    mo.md(r"""
    This is a loader function for GPT2 weights by using the above.
    """)
    return


@app.function
def load_weights_into_gpt(gpt, params):
    # positional and token embeddings
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    # transformer blocks
    for b in range(len(params["blocks"])):
        # multi-head attention -> linear projection (matrix)
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        # multi-head attention -> linear projection (bias)
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        # multi-head attention -> output projection
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        # feed-forward network (Linear -> GELU -> Linear)
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        # layer normalization
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


@app.cell
def _():
    mo.md(r"""
    Then, load it.
    """)
    return


@app.cell
def _(device, gpt, params):
    load_weights_into_gpt(gpt, params)
    gpt.to(device);
    return


@app.cell
def _():
    mo.md(r"""
    It outputs rational sentence.
    """)
    return


@app.cell
def _(NEW_CONFIG, device, gpt, tokenizer):
    torch.manual_seed(123)

    _token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )

    print("Output text:\n", token_ids_to_text(_token_ids, tokenizer))
    return


if __name__ == "__main__":
    app.run()
