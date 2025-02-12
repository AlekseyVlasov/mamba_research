import argparse
import torch
import pytorch_warmup as warmup
import wandb
from tqdm import tqdm
import yaml
import sys
import os

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from mamba_ssm.models.config_mamba import MambaConfig

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

from utils import print_model_size, fix_seed
from models.MambaWithEmbeddings import MambaLMHeadModelWithEmbeddings

from dotenv import load_dotenv

load_dotenv()


def add_special_token(embedded_inputs, special_token, period, tokens_num):
    # Expand special_token to match batch size
    batch_special_token = special_token.expand(embedded_inputs.size(0), -1, -1)

    # Calculate number of period-token chunks
    num_chunks = embedded_inputs.shape[1] // period + (1 if embedded_inputs.shape[1] % period != 0 else 0)

    # Insert special token after every period tokens
    embedded_with_special = torch.cat(
        [
            torch.cat((batch_special_token, embedded_inputs[:, i * period : (i + 1) * period]), dim=1)
            for i in range(num_chunks)
        ],
        dim=1
    )

    # Ensure sequence length matches the original plus the added special tokens
    assert embedded_with_special.size()[1] == embedded_inputs.size(1) + tokens_num * num_chunks
#     embedded_with_special = embedded_with_special[:, :embedded_inputs.size(1) + num_chunks]


    return embedded_with_special

def inference(model, data, device, criterion=None, num_last_tokens=1, special_token=None, period=None):
    """
    Performs inference with the model. Can handle a single batch or an entire DataLoader.
    
    Parameters:
    - model: torch.nn.Module - the model for inference.
    - data: torch.Tensor or torch.utils.data.DataLoader - input data or DataLoader.
    - device: torch.device - device (CPU or GPU).
    - criterion: torch.nn.Module (optional) - loss function for calculating loss.
    - num_last_tokens: int - number of last tokens to predict (default is 1).
    - special_token: torch.Tensor (optional) - special token to insert after every 1000 tokens.

    Returns:
    - list of batch losses and accuracy if DataLoader passed.
    - torch.Tensor if single batch passed - predicted classes.
    """
    model.eval()  # Set model to evaluation mode
    
    batch_losses = []
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(data, desc="Inference")
        for batch in pbar:
            inputs, labels = batch['input_ids'], batch['labels']
            inputs, labels = inputs.to(device), labels.to(device)
            
            if special_token is not None:
                (b, tokens_num, embed_size) = special_token.size()
                # Convert inputs to embeddings
                embedded_inputs = model.backbone.embedding(inputs)

                embedded_with_special = add_special_token(embedded_inputs, special_token, period, tokens_num)
                
                # Forward pass with embeddings as input, using is_embeds=True
                outputs = model(embedded_with_special, is_embeds=True, num_last_tokens=num_last_tokens)
            else:
                # If no special_token is given, proceed with the standard input
                outputs = model(inputs, num_last_tokens=num_last_tokens)

            # Extract logits and calculate loss if criterion is provided
            logits = outputs.logits[:, 0, :]
            if criterion is not None:
                loss = criterion(logits, labels)
                batch_losses.append(loss.item())
                pbar.set_postfix({"Loss (batch)": loss.item()})

            # Calculate accuracy
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Calculate step accuracy
            step_accuracy = 100 * predicted.eq(labels).sum().item() / labels.size(0)

            # Log metrics for each step
            wandb.log({
                "inference_loss_step": loss.item() if criterion is not None else None,
                "inference_accuracy_step": step_accuracy
            })

    # Calculate overall accuracy
    accuracy = 100 * correct / total
    return batch_losses, accuracy

def train_model(seed, device, train_loader, val_loader, model, wandb_config, train_config, model_config, save_model=False):
    fix_seed(seed)

    print_model_size(model)

    num_epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    warmup_percent = train_config['warmup_percent']  # Now represents a percentage of total steps

    # Move the model to the specified device (GPU or CPU)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(warmup_percent * total_steps)  # Calculate warmup steps as a percentage of total steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=5e-6)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_steps)

    # Lists to store batch-wise losses
    train_batch_losses = []
    val_batch_losses = []

    fix_seed(seed)

    for epoch in range(num_epochs):
        # ---- Training phase ----
        model.train()  # Set model to training mode
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for batch in pbar:
            inputs, labels = batch['input_ids'], batch['labels']
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, num_last_tokens=1)
            logits = outputs.logits[:, 0, :]  # Extract logits field for loss calculation
            loss = criterion(logits, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Apply warmup and scheduler updates
            with warmup_scheduler.dampening():
                scheduler.step()

            # Accumulate training loss and accuracy
            train_loss += loss.item()
            train_batch_losses.append(loss.item())  # Store the loss for each batch
            _, predicted = logits.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            # Log training metrics for each step
            wandb.log({
                "train_loss_step": loss.item(),
                "train_accuracy_step": predicted.eq(labels).sum().item() / labels.size(0),
                "lr_step": optimizer.param_groups[0]['lr']
            })

            # Update progress bar with current batch loss
            pbar.set_postfix({"Train Loss (batch)": loss.item()})

        # ---- Epoch-Based Logging ----
        train_accuracy_epoch = 100 * correct_train / total_train
        wandb.log({
            "train_loss_epoch": train_loss / len(train_loader),
            "train_accuracy_epoch": train_accuracy_epoch
        })

        # ---- Validation phase after each epoch ----
        val_batch_losses_local, val_accuracy = inference(model, val_loader, device, criterion=criterion, num_last_tokens=1)
        val_loss = sum(val_batch_losses_local) / len(val_batch_losses_local)

        # Log validation metrics for the epoch
        wandb.log({
            "val_loss_epoch": val_loss,
            "val_accuracy_epoch": val_accuracy
        })

    if save_model:
        torch.save(model, f'models/{model_config['save_name']}.pth')
    

def train_embeddings(seed, device, train_loader, val_loader, model, wandb_config, train_config, model_config, save_model=False):
    num_epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    tokens_num = model_config['tokens_num']
    period = model_config['period']
    warmup_percent = train_config['warmup_percent']  # Now represents a percentage of total steps

    special_token = torch.randn(1, tokens_num, model.config.d_model, requires_grad=True, device=device)
    model.to(device)
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW([special_token, model.parameters()], lr=learning_rate)

    optimizer = optim.AdamW(
        [{'params': [special_token], 'lr': learning_rate},
        {'params': model.parameters(), 'lr': learning_rate}],
    )

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(warmup_percent * total_steps)  # Calculate warmup steps as a percentage of total steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=5e-6)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_steps)

    # Log results before training
    test_batch_losses, test_accuracy = inference(model, val_loader, device, criterion=criterion, num_last_tokens=1)
    test_loss = sum(test_batch_losses) / len(test_batch_losses)
    wandb.log({"Test Loss": test_loss, "Test Accuracy": test_accuracy})

    fix_seed(seed)

    for epoch in range(num_epochs):
        model.train()  # Set model to evaluation mode
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for batch in pbar:
            inputs, labels = batch['input_ids'], batch['labels']
            # Move data to the specified device
            inputs, labels = inputs.to(device), labels.to(device)

            # Convert inputs to embeddings without tracking gradients
            with torch.no_grad():
                embedded_inputs = model.backbone.embedding(inputs)

            embedded_with_special = add_special_token(embedded_inputs, special_token, period, tokens_num)
            
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass with embeddings as input, using is_embeds=True
            outputs = model(embedded_with_special, is_embeds=True, num_last_tokens=1)
            logits = outputs.logits[:, 0, :]
            loss = criterion(logits, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Apply warmup and scheduler updates
            with warmup_scheduler.dampening():
                scheduler.step()

            # Accumulate training loss and accuracy
            train_loss += loss.item()
            _, predicted = logits.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            # Calculate local train accuracy
            train_accuracy_local = 100 * correct_train / total_train

            # Log training metrics for each step
            wandb.log({
                "train_loss_step": loss.item(),
                "train_accuracy_step": predicted.eq(labels).sum().item() / labels.size(0),
                "lr_step": optimizer.param_groups[0]['lr']
            })

            # Display the current loss for each training batch
            pbar.set_postfix({"Train Loss (batch)": loss.item()})

        # ---- Epoch-Based Logging ----
        train_accuracy_epoch = 100 * correct_train / total_train
        wandb.log({
            "train_loss_epoch": train_loss / len(train_loader),
            "train_accuracy_epoch": train_accuracy_epoch
        })

        # ---- Validation phase after each epoch ----
        val_batch_losses_local, val_accuracy = inference(
            model, val_loader, device, criterion=criterion, num_last_tokens=1, special_token=special_token, period=period
        )
        val_loss = sum(val_batch_losses_local) / len(val_batch_losses_local)

        # Log validation metrics for the epoch
        wandb.log({
            "val_loss_epoch": val_loss,
            "val_accuracy_epoch": val_accuracy
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="experiment config file")
    parser.add_argument("--tokens_num", required=False, default=None, help="Value for model['tokens_num'] if given")
    parser.add_argument("--period", required=False, default=None, help="Value for model['period'] if given")

    config = parser.parse_args().config
    tokens_num = parser.parse_args().tokens_num
    period = parser.parse_args().period

    print("\nUsing config {0}".format(config))


    # get args
    with open("configs/"+config) as stream:
        try:
            args = yaml.safe_load(stream)            
        except yaml.YAMLError as exc:
            raise RuntimeError(exc)
    
    if tokens_num is not None:
        args['model']['tokens_num'] = int(tokens_num)
        args['wandb']['name'] += ' ' + tokens_num
    
    if period is not None:
        args['model']['period'] = int(period)
        args['wandb']['name'] += ' ' + period

    # get GPU info
    if not torch.cuda.is_available():
        raise NotImplementedError("No GPU available!")
    gpu_number = args['gpu_number']
    device = torch.device(f'cuda:{gpu_number}' if torch.cuda.is_available() else 'cpu')
    seed = args["seed"]

    print(f"Running on device {device}")
    gpu_type = torch.cuda.get_device_name(0)
    print("Running on {0}".format(gpu_type))

    # get wandb config
    if "wandb" in args:
        wandb_config = args.pop("wandb")
    else:
        wandb_config = None
    
    print("\nCONFIG:")
    print(yaml.dump(args))

    # split configs
    model_config = args["model"]
    train_config = args["training"]
    training_type = args["training_type"]
    if training_type == "fine-tune":
        freeze = args["freeze"]

    # start wandb logging
    if wandb_config is not None:
        wandb_api_key = os.getenv("WANDB_API_KEY")

        wandb.login(key=wandb_api_key)
        wandb.init(
                project=wandb_config["project"],
                group=wandb_config["group"],
                name=wandb_config["name"],
                config=args,
                job_type="train",
                settings=wandb.Settings(_disable_stats=True)
        )
    
    dataset = load_dataset("yelp_polarity")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    if tokenizer.pad_token is None:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False, collate_fn=data_collator)


    # extract model class [mamba | transformer | etc.]
    model_name = model_config.pop("name")
    
    fix_seed(seed)
    model = MambaLMHeadModelWithEmbeddings.from_pretrained(model_name, num_labels=2)

    train_fn = None
    if training_type == "embeddings":
        model.freeze_layers()
        train_fn = train_embeddings
    elif training_type == "fine-tune":
        if freeze:
            model.freeze_layers()
        train_fn = train_model
    
    train_fn(
        seed,
        device,
        train_dataloader,
        test_dataloader,
        model,
        wandb_config,
        train_config,
        model_config,
        save_model=args['save_model']
    )
    
    try:
        if wandb_config is not None:
            wandb.finish()
    except:
        sys.exit(0)
