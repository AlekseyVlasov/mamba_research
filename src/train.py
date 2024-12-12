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

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from mamba_ssm.models.config_mamba import MambaConfig

from utils import print_model_size
from models.MambaWithEmbeddings import MambaLMHeadModelWithEmbeddings
from data.InductionHeads import ICLDataModule

def log_loss_vs_complexity(model, test_loader, device, criterion, special_token=None, period=None, wandb_run=None, suffix=""):
    # Initialize lists for positions and tokens
    pos = []
    tokens = []

    # Iterate over the test_loader
    for sample, label in test_loader:
        # Find positions where the value equals 15
        positions = torch.nonzero(sample[0] == 15, as_tuple=True)[0]

        pos.append(positions[0])  # Store position (first occurrence)
        tokens.append(label)

    # Inference
    test_batch_losses, test_accuracy = inference(
        model, test_loader, device, criterion=criterion, num_last_tokens=1, 
        special_token=special_token, period=period
        )

    test_loss = sum(test_batch_losses) / len(test_batch_losses)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Ensure all lists have the same length
    assert len(test_batch_losses) == len(pos) == len(tokens), "All lists must have the same length"

    # Create a color map with 14 distinct colors and set boundaries for each category
    cmap = ListedColormap(plt.cm.tab20.colors[:14])
    norm = BoundaryNorm(np.arange(-0.5, 14, 1), cmap.N)  # Boundaries set to center color intervals

    # Plot scatter plot
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(pos, test_batch_losses, c=tokens, cmap=cmap, norm=norm, s=60, alpha=0.7, edgecolor='k')

    # Plot formatting
    plt.title(f"Loss vs. Task Complexity by Token Category {suffix}", fontsize=16)
    plt.xlabel("Task Complexity", fontsize=14)
    plt.ylabel("Loss", fontsize=14)

    # Color bar to indicate token categories with centered ticks
    cbar = plt.colorbar(sc, ticks=range(14))
    cbar.set_label("Token Category", fontsize=12)
    cbar.set_ticks(range(14))  # Ensure color bar has ticks for each category
    cbar.set_ticklabels(range(1, 15))

    # Show grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Log plot to wandb
    if wandb_run:
        wandb.log({f"Loss vs Task Complexity {suffix}": wandb.Image(plt)})
    
    # Close plot to free memory after logging
    plt.close()


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
        for inputs, labels in pbar:
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

    accuracy = 100 * correct / total
    return batch_losses, accuracy

def train_model(seed, device, train_loader, val_loader, model, wandb_config, train_config, model_config, save_model=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print_model_size(model)

    num_epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']

    # Move the model to the specified device (GPU or CPU)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min = 5e-6)
    warmup_scheduler = warmup.LinearWarmup(optimizer, train_config["warmup"])

    # Lists to store batch-wise losses
    train_batch_losses = []
    val_batch_losses = []

    torch.manual_seed(seed)
    np.random.seed(seed)

    for epoch in range(num_epochs):
        # ---- Training phase ----
        model.train()  # Set model to training mode
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for inputs, labels in pbar:
            # Move data to the specified device
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

            # Accumulate training loss and accuracy
            train_loss += loss.item()
            train_batch_losses.append(loss.item())  # Store the loss for each batch
            _, predicted = logits.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            # Update progress bar with current batch loss
            pbar.set_postfix({"Train Loss (batch)": loss.item()})

        # Calculate average training accuracy
        train_accuracy = 100 * correct_train / total_train

        with warmup_scheduler.dampening():
            scheduler.step()

        # ---- Validation phase ----
        val_batch_losses_local, val_accuracy = inference(model, val_loader, device, criterion=criterion, num_last_tokens=1)
        val_loss = sum(val_batch_losses_local)
        val_batch_losses.extend(val_batch_losses_local)

        # Log results to WandB for this epoch
        wandb.log({
            "train_loss": train_loss / len(train_loader),
            "train_accuracy": train_accuracy,
            "val_loss": val_loss / len(val_loader),
            "val_accuracy": val_accuracy,
            "lr": optimizer.param_groups[0]['lr']
        })

    if save_model:
        torch.save(model, f'models/{model_config['save_name']}.pth')
    

def train_embeddings(seed, device, train_loader, val_loader, model, wandb_config, train_config, model_config, save_model=False):
    # Move the model to the specified device
    for param in model.parameters():
            param.requires_grad = False
        
    num_epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    tokens_num = model_config['tokens_num']
    period = model_config['period']

    special_token = torch.randn(1, tokens_num, model.config.d_model, requires_grad=True, device=device)
    model.to(device)
    
    # Define the loss function and optimizer (only optimizing the special token)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([special_token], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min = 5e-6)
    warmup_scheduler = warmup.LinearWarmup(optimizer, train_config["warmup"])

    log_loss_vs_complexity(model, val_loader, device, criterion, wandb_run=True, suffix="initial")

    # Lists to store batch-wise losses
    train_batch_losses = []
    val_batch_losses = []

    # Log results before training
    test_batch_losses, test_accuracy = inference(model, val_loader, device, criterion=criterion, num_last_tokens=1)

    test_loss = sum(test_batch_losses) / len(test_batch_losses)
    wandb.log({"Test Loss": test_loss, "Test Accuracy": test_accuracy})

    for epoch in range(num_epochs):
        # ---- Training phase ----
        model.eval()  # Set model to evaluation mode
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for inputs, labels in pbar:
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

            # Accumulate training loss and accuracy
            train_loss += loss.item()
            train_batch_losses.append(loss.item())
            _, predicted = logits.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            # Display the current loss for each training batch
            pbar.set_postfix({"Train Loss (batch)": loss.item()})

        # Calculate average training accuracy
        train_accuracy = 100 * correct_train / total_train

        with warmup_scheduler.dampening():
            scheduler.step()

        # ---- Validation phase ----
        val_batch_losses_local, val_accuracy = inference(
            model, val_loader, device, criterion=criterion, num_last_tokens=1, special_token=special_token, period=period
        )
        val_loss = sum(val_batch_losses_local)
        val_batch_losses.extend(val_batch_losses_local)

        # Log training and validation metrics for the current epoch
        wandb.log({
            "train_loss": train_loss / len(train_loader),
            "train_accuracy": train_accuracy,
            "val_loss": val_loss / len(val_loader),
            "val_accuracy": val_accuracy,
            "lr": optimizer.param_groups[0]['lr']
        })
    log_loss_vs_complexity(model, val_loader, device, criterion,special_token=special_token, period=period,  wandb_run=True)


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
    data_config = args["dataset"]

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
    
    # prepare dataset
    module = ICLDataModule(
        num_examples=data_config['train_examples'],
        num_test_examples=data_config['test_examples'],
        vocab_size=data_config['vocab_size'],
        input_seq_len=data_config['input_seq_len'],
        copy_method='induction_head',
        # Default parameters
        number_duplicates_per_epoch=0,
        seed=args['seed'],
        split_train_test=False,
        induction_len=1,
        induction_num_triggers=1,
        allow_dot=False,
        max_copy_len=10,
        test_seq_len=None,
        num_keys=1,
        data_dir='data',
    )

    module.setup()

    # dataloaders
    train_loader = module.train_dataloader(batch_size=train_config['batch_size'])
    val_loader = module.val_dataloader(batch_size=data_config['test_batch_size'])

    # extract model class [mamba | transformer | etc.]
    model_name = model_config.pop("name")
    

    train_fn = None
    model = None
    if model_name == "mamba":
        model_cls = MambaLMHeadModelWithEmbeddings

        ssm_cfg = {
            'layer': 'Mamba1',
            'd_state': model_config['d_state'],
        #     'd_conv': 4,
        #     'expand': 2,
        #     'dt_rank': "auto",
        #     'dt_min': 0.001,
        #     'dt_max': 0.1,
        #     'dt_init': "random",
        #     'dt_scale': 1.0,
        #     'dt_init_floor': 1e-4,
        #     'conv_bias': True,
        #     'bias': False,
            'use_fast_path': True
        }

        config = MambaConfig(
            d_model = model_config['d_model'],
        #     d_intermediate = 0,
            n_layer = model_config['n_layer'],
            vocab_size = data_config['vocab_size'],
            ssm_cfg=ssm_cfg,
        #     attn_layer_idx = field(default_factory=list),
        #     attn_cfg = field(default_factory=dict),
        #     rms_norm = True,
        #     residual_in_fp32 = True,
        #     fused_add_norm = True,
        #     pad_vocab_size_multiple = 8,
        #     tie_embeddings = True,
        )

        model = model_cls(config)
        train_fn = train_model
    elif model_name == "embeddings":
        model = torch.load(f'models/{model_config['base_model']}.pth')
        train_fn = train_embeddings

    # elif model_name == "transformer":
    #     model_cls = Transformer
    # elif model_name == "hawk":
    #     model_cls = Hawk
    # elif model_name == "seahawk":
    #     model_cls = SEaHawk
    else:
        raise RuntimeError("{0} is not a valid model option".format(model_name))
    
    train_fn(
        args["seed"],
        device,
        train_loader,
        val_loader,
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