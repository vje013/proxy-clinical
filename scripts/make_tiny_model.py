"""Build a tiny, randomly initialised Qwen2-architecture model plus a BPE
tokenizer trained on the pilot corpus, for CPU smoke runs of the Block 2
pipeline. It exercises exactly the same code paths as the real model (local
directory instead of a Hub id) and nothing else.

    python scripts/make_tiny_model.py --corpus data/pilot/corpus.jsonl --out models/tiny-qwen2-smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=6000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- tokenizer: byte-level BPE on the corpus text plus JSON targets
    texts: list[str] = []
    with open(args.corpus, "r", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            texts.append(r["text"])
            texts.append(json.dumps({"mentions": [{"span": [m["start"], m["end"]], "type": m["type"], "id": m["entity_id"]}
                                                  for m in r["mentions"]]}, separators=(",", ":")))
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
        model_max_length=16384,
        padding_side="left",
    )
    fast.chat_template = CHATML_TEMPLATE
    fast.save_pretrained(out)

    # --- model
    torch.manual_seed(args.seed)
    cfg = Qwen2Config(
        vocab_size=len(fast),
        hidden_size=args.hidden,
        intermediate_size=args.hidden * 2,
        num_hidden_layers=args.layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16384,
        tie_word_embeddings=True,
        pad_token_id=fast.pad_token_id,
        eos_token_id=fast.eos_token_id,
        bos_token_id=None,
    )
    model = Qwen2ForCausalLM(cfg)
    model.generation_config.eos_token_id = fast.eos_token_id
    model.generation_config.pad_token_id = fast.pad_token_id
    model.save_pretrained(out, safe_serialization=True)
    n = sum(p.numel() for p in model.parameters())
    (out / "README.md").write_text(
        f"Tiny random Qwen2 stand-in ({n:,} params, vocab {len(fast)}) for CPU smoke runs. Not a trained model.\n",
        encoding="utf-8",
    )
    print(f"wrote {out} ({n:,} params, vocab {len(fast)})")


if __name__ == "__main__":
    main()
