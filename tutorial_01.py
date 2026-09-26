r"""Code-along 1: run GPT-2 and take a first look inside.

Read TUTORIAL_01.md alongside this file. Run from PowerShell:
    .\.venv\Scripts\python.exe .\tutorial_01.py

The first run downloads model weights. Importing this file does nothing.
"""

# These are the only settings to change during this code-along.
PROMPT = "The capital of France is"
TOP_K = 5
LAYER = 0  # Layers and heads are numbered starting at zero.
HEAD = 0


def main():
    import torch
    from transformer_lens import HookedTransformer

    # 1. Load the trained model. CPU works without a graphics card.
    # Keep the original weight conventions explicit; no training happens here.
    print("Loading GPT-2 Small on CPU (first run downloads weights)...", flush=True)
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        device="cpu",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
    )
    model.eval()
    print(f"Model: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads per layer")

    if not PROMPT.strip():
        raise ValueError("Use a nonempty prompt for this code-along.")
    if not 0 <= LAYER < model.cfg.n_layers or not 0 <= HEAD < model.cfg.n_heads:
        raise ValueError("LAYER or HEAD is outside this model's range.")
    if not 1 <= TOP_K <= model.cfg.d_vocab:
        raise ValueError("TOP_K must be between 1 and the vocabulary size.")

    # 2. Turn the prompt into token IDs. BOS means beginning of sequence.
    tokens = model.to_tokens(PROMPT, prepend_bos=True)
    labels = model.to_str_tokens(tokens[0])
    if tokens.shape[1] > model.cfg.n_ctx:
        raise ValueError(f"Prompt exceeds the {model.cfg.n_ctx}-token context window.")
    print(f"\nPrompt: {PROMPT!r}")
    print("Position | Token ID | Readable token")
    for position, (token_id, label) in enumerate(zip(tokens[0].tolist(), labels)):
        print(f"{position:8d} | {token_id:8d} | {label!r}")
    print("tokens shape [batch, position]:", tuple(tokens.shape))

    # 3. Run once and save two kinds of intermediate result from one layer.
    # A cache is a dictionary-like collection of values from this forward pass.
    residual_name = f"blocks.{LAYER}.hook_resid_pre"
    pattern_name = f"blocks.{LAYER}.attn.hook_pattern"
    with torch.inference_mode():
        logits, cache = model.run_with_cache(
            tokens, names_filter=[residual_name, pattern_name]
        )

        # 4. At each position, the output predicts the NEXT token.
        # [0, -1] means: first prompt, last input position.
        next_token_logits = logits[0, -1]
        probabilities = next_token_logits.softmax(dim=-1)
        top_probabilities, top_ids = probabilities.topk(TOP_K)
        print("\nlogits shape [batch, position, vocabulary]:", tuple(logits.shape))
        print(f"Top {TOP_K} candidates for the next token:")
        for probability, token_id in zip(top_probabilities.tolist(), top_ids.tolist()):
            text = model.to_string(token_id)
            print(f"  {text!r:24} probability={probability:.2%}")

        # 5. Inspect a representation and one head's attention routing.
        residual = cache[residual_name]
        pattern = cache[pattern_name]
        print("\nResidual shape [batch, position, features]:", tuple(residual.shape))
        print("Attention shape [batch, head, query, source]:", tuple(pattern.shape))

        # This row says where the last position reads from in this head.
        attention_row = pattern[0, HEAD, -1, :]
        weights, positions = attention_row.topk(min(TOP_K, len(labels)))
        print(f"\nLayer {LAYER}, head {HEAD}: attention FROM position {len(labels) - 1}")
        for weight, position in zip(weights.tolist(), positions.tolist()):
            print(f"  TO position {position:2d} {labels[position]!r:24} weight={weight:.2%}")

        # These checks run on your machine. They check the interpretation of axes.
        torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
        row_sums = pattern.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums))
        assert pattern[0, HEAD].triu(diagonal=1).abs().max().item() < 1e-6
        print("\nChecks passed: probabilities sum to 1; attention is normalized and causal.")
        print("Attention weights describe routing, not proof of a token's causal importance.")


if __name__ == "__main__":
    main()