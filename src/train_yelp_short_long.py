import argparse
import torch
import wandb
import yaml
import sys
import os

from mamba_ssm.models.config_mamba import MambaConfig

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

from utils import print_model_size, fix_seed, print_trainable_params_num
from models.MambaWithEmbeddings import MambaLMHeadModelWithEmbeddings
from training_functions import train_model, train_embeddings, train_lora

import types

from peft import LoraConfig, get_peft_model

from dotenv import load_dotenv

from datasets import concatenate_datasets

load_dotenv()


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
    
    if "lora" in model_config:
        lora_config = model_config["lora"]

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
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    def tokenize_function_with_len(examples):
        tokens = tokenizer(examples["text"], padding=False, truncation=False)
        tokens["length"] = [len(x) for x in tokens["input_ids"]]
        return tokens

    tokenized_datasets = dataset.map(tokenize_function_with_len, batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    short_length = train_config['short']
    long_length = train_config['long']

    short_train = tokenized_datasets["train"].filter(lambda x: x["length"] <= short_length)
    short_test = tokenized_datasets["test"].filter(lambda x: x["length"] <= short_length)
    long_train = tokenized_datasets["train"].filter(lambda x: x["length"] >= long_length)
    long_test = tokenized_datasets["test"].filter(lambda x: x["length"] >= long_length)

    new_train_dataset = concatenate_datasets([short_train, short_test])
    new_test_dataset = concatenate_datasets([long_train, long_test])

    train_dataloader = DataLoader(new_train_dataset, batch_size=train_config['batch_size'], shuffle=True, collate_fn=data_collator)
    test_dataloader = DataLoader(new_test_dataset, batch_size=train_config['test_batch_size'], shuffle=False, collate_fn=data_collator)

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
    elif training_type == "lora":
        config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            bias=lora_config["bias"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["lora_dropout"],
        )

        # Hot-fix for get_peft_model to work
        def get_method(self, key, default=None):
            return getattr(self, key, default)

        model.config.get = types.MethodType(get_method, model.config)
        model = get_peft_model(model, config)
        train_fn = train_lora
    
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
