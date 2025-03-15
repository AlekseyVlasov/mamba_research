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

from utils import fix_seed, print_model_size
from models.MambaWithEmbeddings import MambaLMHeadModelWithEmbeddings
from training_functions import train_model, train_embeddings, train_lora
from data.InductionHeads import ICLDataModule


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
    data_config = args["dataset"]
    training_type = args["training_type"]

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
        vary_length=data_config['vary_length']
    )

    module.setup()

    # dataloaders
    train_dataloader = module.train_dataloader(batch_size=train_config['batch_size'])
    test_dataloader = module.val_dataloader(batch_size=data_config['test_batch_size'])

    train_fn = None
    model = None
    if training_type == "mamba":
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

        fix_seed(seed)
        model = model_cls(config)
        train_fn = train_model
    elif training_type == "embeddings":
        model = torch.load(f'models/{model_config['base_model']}.pth')
        model.freeze_layers()
        train_fn = train_embeddings
    else:
        raise RuntimeError("{0} is not a valid model option".format(training_type))
    
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