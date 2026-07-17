import torch
from f_token_generator import generate_next_tokens


def text_to_token_ids(text, tokenizer, device=None):
    token_ids = tokenizer.encode(
        text,
        allowed_special={"<|endoftext|>"}
    )
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def generate_text(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=50,
    temperature=0.0,
    top_k=None,
    eos_id=None,
):
    was_training = model.training
    model.eval()

    with torch.no_grad():
        token_ids = generate_next_tokens(
            model=model,
            idx=text_to_token_ids(prompt, tokenizer, device),
            max_new_tokens=max_new_tokens,
            context_size=model.pos_emb.weight.shape[0],
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id,
        )

    if was_training:
        model.train()

    return token_ids_to_text(token_ids, tokenizer)


def print_generated_text(*args, **kwargs):
    text = generate_text(*args, **kwargs)
    print(text.replace("\n", " "))
    return text