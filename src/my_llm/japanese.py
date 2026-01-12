import marimo

__generated_with = "0.19.2"
app = marimo.App()

with app.setup:
    import numpy as np
    import torch
    from pathlib import Path
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
    from bs4 import BeautifulSoup

    import my_llm
    from ch04 import GPTModel, generate_text_simple
    from ch05 import train_model_simple, plot_losses, generate, assign


@app.cell
def _(mo):
    mo.md(r"""
    ## Chapter 2 相当: トークナイザーの用意
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    [rinna/japanese-gpt2-medium](https://huggingface.co/rinna/japanese-gpt2-medium) のトークナイザは [sentencepiece](https://github.com/google/sentencepiece) ベース。
    """)
    return


@app.cell
def _():
    tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium", use_fast=False)
    tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

    print(f"{type(tokenizer)=}")
    print(f"{tokenizer=}")
    return (tokenizer,)


@app.cell
def _(mo):
    mo.md(r"""
    重要なものを抜き出し
    """)
    return


@app.cell
def _(tokenizer):
    print(f"{tokenizer.vocab_size=}")   # Size of Vocabulary
    print(f"{tokenizer.pad_token=}")    # PAD: Padding Token
    print(f"{tokenizer.padding_side=}")  # left or right
    print(f"{tokenizer.unk_token=}")    # UNK: Unknown Token
    print(f"{tokenizer.eos_token=}")    # EOS: End Of Sentence
    print(f"{tokenizer.bos_token=}")    # BOS: Beginning Of Sentence
    return


@app.cell
def _(tokenizer):
    sample_text = "私は、その男の写真を三葉、見たことがある。"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"{encoded=}")
    print(f"{decoded=}")
    return (encoded_tensor,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Chapter 4 相当: 未学習 GPT-2 モデルで推論
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ランダム初期化されたモデルを用意（後できちんと上書きのコードを追加）
    """)
    return


@app.cell
def _():
    torch.manual_seed(42)

    # rinna/japanese-gpt2-medium model と合わせて変更
    GPT_CONFIG_124M = {
        "vocab_size": 32000,    # Vocabulary size
        "context_length": 1024,  # Shortened context length
        "emb_dim": 1024,         # Embedding dimension
        "n_heads": 16,          # Number of attention heads
        "n_layers": 24,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": True       # Query-Key-Value bias
    }
    random_model = GPTModel(GPT_CONFIG_124M)
    random_model.eval();
    return GPT_CONFIG_124M, random_model


@app.cell
def _(mo):
    mo.md(r"""
    事前学習されたモデルを用意
    """)
    return


@app.cell
def _():
    rinna_model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
    rinna_model.eval();
    return (rinna_model,)


@app.cell
def _(mo):
    mo.md(r"""
    学習なしでは意味をなさない
    """)
    return


@app.cell
def _(GPT_CONFIG_124M, encoded_tensor, random_model, tokenizer):
    torch.manual_seed(123)

    _out_gen = generate_text_simple(
        model=random_model,
        idx=encoded_tensor, 
        max_new_tokens=6, 
        context_size=GPT_CONFIG_124M["context_length"]
    )
    _decoded_text = tokenizer.decode(_out_gen.squeeze(0).tolist())

    print("Output:", _out_gen)
    print("Output length:", len(_out_gen[0]))
    print(_decoded_text)
    return


@app.cell
def _(mo):
    mo.md(r"""
    事前学習済みのモデルを試してみる。それっぽい文章が続く。
    """)
    return


@app.cell
def _(encoded_tensor, rinna_model, tokenizer):
    torch.manual_seed(123)

    with torch.no_grad():
        rinna_out = rinna_model.generate(
            encoded_tensor,
            min_length=50,
            max_length=100,
            do_sample=True,
            top_k=100,
            top_p=0.95,
            num_return_sequences=3,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )[0]

    print(f"{rinna_out[:50]=}")
    _decoded_text = tokenizer.decode(rinna_out.squeeze(0).tolist())
    print(_decoded_text)
    return


@app.cell
def _(mo):
    mo.md(r"""
    このモデルを GPT-2 from scratch へ写し取る。
    """)
    return


@app.cell
def _(rinna_model):
    print(f"{type(rinna_model)=}")
    print(f"{rinna_model=}")
    print(f"{rinna_model.__dict__.keys()=}")
    return


@app.cell
def _(rinna_model):
    rinna_model.config
    return


@app.cell
def _(rinna_model):
    sd_dict = dict(rinna_model.state_dict())  # dict[str, Tensor]
    print(f"{sd_dict.keys()=}")
    return (sd_dict,)


@app.cell
def _(mo):
    mo.md(r"""
    重み共有があるかチェック。重み共有があれば 0 となるはず。
    """)
    return


@app.cell
def _(sd_dict):
    _weight_diff_sum = torch.sum(
        sd_dict["transformer.wte.weight"] - sd_dict["lm_head.weight"])
    print(f"{_weight_diff_sum=}")
    return


@app.function
def load_gpt2_params_from_rinna(gpt, trained_model: GPT2LMHeadModel):
    sd_dict = trained_model.state_dict()
    # positional and token embeddings
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, sd_dict["transformer.wte.weight"])
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, sd_dict["transformer.wpe.weight"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, sd_dict["transformer.ln_f.weight"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, sd_dict["transformer.ln_f.bias"])
    gpt.out_head.weight = assign(gpt.out_head.weight, sd_dict["lm_head.weight"])

    # transformer blocks
    for b in range(trained_model.config.n_layer):
        # multi-head attention -> linear projection (matrix)
        q_w, k_w, v_w = np.split(
            sd_dict[f"transformer.h.{b}.attn.c_attn.weight"], 3, axis=-1
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
            sd_dict[f"transformer.h.{b}.attn.c_attn.bias"], 3, axis=-1
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
            sd_dict[f"transformer.h.{b}.attn.c_proj.weight"].T,
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            sd_dict[f"transformer.h.{b}.attn.c_proj.bias"],
        )

        # feed-forward network (Linear -> GELU -> Linear)
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            sd_dict[f"transformer.h.{b}.mlp.c_fc.weight"].T,
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, sd_dict[f"transformer.h.{b}.mlp.c_fc.bias"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            sd_dict[f"transformer.h.{b}.mlp.c_proj.weight"].T,
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            sd_dict[f"transformer.h.{b}.mlp.c_proj.bias"],
        )

        # layer normalization
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, sd_dict[f"transformer.h.{b}.ln_1.weight"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, sd_dict[f"transformer.h.{b}.ln_1.bias"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, sd_dict[f"transformer.h.{b}.ln_2.weight"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, sd_dict[f"transformer.h.{b}.ln_2.bias"]
        )


@app.cell
def _(GPT_CONFIG_124M, rinna_model):
    my_model = GPTModel(GPT_CONFIG_124M)
    load_gpt2_params_from_rinna(my_model, rinna_model)
    return (my_model,)


@app.cell
def _(GPT_CONFIG_124M, encoded_tensor, my_model, tokenizer):
    torch.manual_seed(123)
    _out_gen = generate_text_simple(
        model=my_model,
        idx=encoded_tensor, 
        max_new_tokens=20, 
        context_size=GPT_CONFIG_124M["context_length"]
    )
    _decoded_text = tokenizer.decode(_out_gen.squeeze(0).tolist())

    print("Output:", _out_gen)
    print("Output length:", len(_out_gen[0]))
    print(_decoded_text)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Chapter 5 相当: 青空文庫で事前学習
    """)
    return


@app.cell
def _():
    Path(my_llm.__path__).parent.parent / "third_party" / "aozorabunko" / "cards"
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Chapter 6 相当: 作家分類問題のファインチューニング
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    [参考](https://qiita.com/tsukemono/items/88f6040001e16e4bbe7f?utm_source=chatgpt.com)
    """)
    return


if __name__ == "__main__":
    app.run()
