import torch
import pytorch_warmup as warmup
import wandb
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn

from utils import print_model_size, fix_seed, print_trainable_params_num


def add_special_token(embedded_inputs, special_token, period, tokens_num):
    # Expand special_token to match batch size
    batch_special_token = special_token.expand(embedded_inputs.size(0), -1, -1)
    
    # Special case: add token only at the beginning when period is -1
    if period == -1:
        embedded_with_special = torch.cat((batch_special_token, embedded_inputs), dim=1)
        
        # Ensure sequence length matches the original plus the added special tokens
        assert embedded_with_special.size()[1] == embedded_inputs.size(1) + tokens_num
        
        return embedded_with_special
    
    # Regular case: Insert special token after every period tokens
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
    
    return embedded_with_special

def inference(model, data, device, criterion=None, num_last_tokens=1, special_token=None, period=None, log=True):
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
            inputs, labels = unpack_batch(batch)
            if inputs is None:
                continue
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
            if log:
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
            inputs, labels = unpack_batch(batch)
            if inputs is None:
                continue
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

def unpack_batch(batch):
    if type(batch) == list:
        inputs, labels = batch
    else:
        inputs, labels = batch['input_ids'], batch['labels']
    return inputs, labels
    

def train_embeddings(seed, device, train_loader, val_loader, model, wandb_config, train_config, model_config, save_model=False):
    num_epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    tokens_num = model_config['tokens_num']
    period = model_config['period']
    warmup_percent = train_config['warmup_percent']
    accumulation_steps = train_config.get('accumulation_steps', 1)  # Gradient accumulation steps

    # Initialize the special token as trainable parameters
    special_token = torch.randn(1, tokens_num, model.config.d_model, requires_grad=True, device=device)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Define the optimizer
    optimizer = optim.AdamW(
        [{'params': [special_token], 'lr': learning_rate},
         {'params': model.parameters(), 'lr': learning_rate}],
    )

    # Define learning rate scheduler and warmup
    total_steps = num_epochs * len(train_loader) / accumulation_steps
    warmup_steps = int(warmup_percent * total_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=5e-6)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_steps)

    # Log results before training
    test_batch_losses, test_accuracy = inference(model, val_loader, device, criterion=criterion, num_last_tokens=1)
    test_loss = sum(test_batch_losses) / len(test_batch_losses)
    wandb.log({"Test Loss": test_loss, "Test Accuracy": test_accuracy})

    fix_seed(seed)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        optimizer.zero_grad()  # Clear gradients at the start of the epoch

        # Track accumulated loss and accuracy over accumulation steps
        accumulated_loss = 0.0
        accumulated_correct = 0
        accumulated_total = 0

        for step, batch in enumerate(pbar):
            inputs, labels = unpack_batch(batch)
            if inputs is None:
                continue
            inputs, labels = inputs.to(device), labels.to(device)


            # Convert inputs to embeddings without tracking gradients
            with torch.no_grad():
                embedded_inputs = model.backbone.embedding(inputs)

            # Add special token embeddings
            embedded_with_special = add_special_token(embedded_inputs, special_token, period, tokens_num)

            # Forward pass
            outputs = model(embedded_with_special, is_embeds=True, num_last_tokens=1)
            logits = outputs.logits[:, 0, :]
            loss = criterion(logits, labels)

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps  
            loss.backward()  # Accumulate gradients

            # Update accumulated metrics
            accumulated_loss += loss.item()
            _, predicted = logits.max(1)
            accumulated_total += labels.size(0)
            accumulated_correct += predicted.eq(labels).sum().item()

            # Update weights only at accumulation steps
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

                # Update learning rate scheduler
                with warmup_scheduler.dampening():
                    scheduler.step()

                # Accumulate total epoch loss
                total_train_loss += accumulated_loss
                total_train += accumulated_total
                correct_train += accumulated_correct

                # Compute accuracy across all accumulation steps
                accumulated_accuracy = accumulated_correct / accumulated_total

                # Log metrics after processing `accumulation_steps`
                wandb.log({
                    "train_loss_step": accumulated_loss,
                    "train_accuracy_step": accumulated_accuracy,
                    "lr_step": optimizer.param_groups[0]['lr']
                })

                # Display batch loss in tqdm progress bar
                pbar.set_postfix({"Train Loss (batch)": accumulated_loss})

                # Reset accumulated values
                accumulated_loss = 0.0
                accumulated_correct = 0
                accumulated_total = 0

        # Epoch-based logging
        train_accuracy_epoch = 100 * correct_train / total_train
        wandb.log({
            "train_loss_epoch": total_train_loss / len(train_loader),
            "train_accuracy_epoch": train_accuracy_epoch
        })

        # Validation phase after each epoch
        val_batch_losses_local, val_accuracy = inference(
            model, val_loader, device, criterion=criterion, num_last_tokens=1, special_token=special_token, period=period
        )
        val_loss = sum(val_batch_losses_local) / len(val_batch_losses_local)

        # Log validation metrics for the epoch
        wandb.log({
            "val_loss_epoch": val_loss,
            "val_accuracy_epoch": val_accuracy
        })
    if save_model:
        torch.save(special_token, f'models/{model_config['save_name']}.pth')

def train_lora(seed, device, train_loader, val_loader, model, wandb_config, train_config, model_config, save_model=False):
    num_epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    warmup_percent = train_config['warmup_percent']
    accumulation_steps = train_config.get('accumulation_steps', 1)
    
    model.to(device)
    
    for name, param in model.named_parameters():
        if "lora" not in name and "classification_head" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    print_trainable_params_num(model)
    
    total_steps = num_epochs * len(train_loader) / accumulation_steps
    warmup_steps = int(warmup_percent * total_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=5e-6)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_steps)
    
    fix_seed(seed)
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        optimizer.zero_grad()
        accumulated_loss = 0.0
        accumulated_correct = 0
        accumulated_total = 0
        
        for step, batch in enumerate(pbar):
            inputs, labels = unpack_batch(batch)
            if inputs is None:
                continue
            inputs, labels = inputs.to(device), labels.to(device)


            outputs = model(inputs, num_last_tokens=1)
            logits = outputs.logits[:, 0, :]
            loss = criterion(logits, labels)
            
            loss = loss / accumulation_steps
            loss.backward()
            
            accumulated_loss += loss.item()
            _, predicted = logits.max(1)
            accumulated_total += labels.size(0)
            accumulated_correct += predicted.eq(labels).sum().item()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                
                with warmup_scheduler.dampening():
                    scheduler.step()
                
                total_train_loss += accumulated_loss
                total_train += accumulated_total
                correct_train += accumulated_correct
                
                wandb.log({
                    "train_loss_step": accumulated_loss,
                    "train_accuracy_step": accumulated_correct / accumulated_total,
                    "lr_step": optimizer.param_groups[0]['lr']
                })
                
                pbar.set_postfix({"Train Loss (batch)": accumulated_loss})
                
                accumulated_loss = 0.0
                accumulated_correct = 0
                accumulated_total = 0
        
        train_accuracy_epoch = 100 * correct_train / total_train
        wandb.log({
            "train_loss_epoch": total_train_loss / len(train_loader),
            "train_accuracy_epoch": train_accuracy_epoch
        })
        
        val_batch_losses, val_accuracy = inference(model, val_loader, device, criterion=criterion, num_last_tokens=1)
        val_loss = sum(val_batch_losses) / len(val_batch_losses)
        
        wandb.log({
            "val_loss_epoch": val_loss,
            "val_accuracy_epoch": val_accuracy
        })