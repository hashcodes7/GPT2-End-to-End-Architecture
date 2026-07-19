import torch

from g_text_generator import generate_text
from k_loss_calculator import calc_loss_batch, calc_loss_loader
from z_helpers import model_mode


def train_model_simple(model,train_loader,val_loader,optimizer,device,num_epochs,
eval_freq,eval_iter,start_context,tokenizer,
):
    """
    Simple training loop for GPT model.
    
    Args:
        model: GPT model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer instance
        device: torch device
        num_epochs: Number of training epochs
        eval_freq: Evaluation frequency (in steps)
        eval_iter: Number of iterations for evaluation
        start_context: Starting prompt for generation
        tokenizer: Tokenizer instance
        
    Returns:
        Tuple of (train_losses, val_losses, track_tokens_seen)
    """
    train_losses = []
    val_losses = []
    track_tokens_seen = []

    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            loss = calc_loss_batch(
                input_batch,
                target_batch,
                model,
                device,
            )

            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter,
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Ep {epoch + 1} "
                    f"(Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        sample = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=start_context,
            device=device,
        )

        print(sample.replace("\n", " "))

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(
    model,
    train_loader,
    val_loader,
    device,
    eval_iter,
):
    """
    Evaluate model on training and validation sets.
    
    Args:
        model: GPT model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: torch device
        eval_iter: Number of iterations for evaluation
        
    Returns:
        Tuple of (train_loss, val_loss)
    """
    with model_mode(model, training=False):
        train_loss = calc_loss_loader(
            train_loader,
            model,
            device,
            eval_iter,
        )

        val_loss = calc_loss_loader(
            val_loader,
            model,
            device,
            eval_iter,
        )

    return train_loss, val_loss
