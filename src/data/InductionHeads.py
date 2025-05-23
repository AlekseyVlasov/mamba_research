import os
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from typing import Dict
import numpy as np
from tqdm import tqdm
from collections import Counter

class Vocab:
    """Custom vocab."""
    def __init__(self, vocab_size: int, special_vocabs: Dict):
        # Special tokens hold copy_prefix and noop/pad token etc
        assert "copy_prefix" in special_vocabs
        self.special_vocabs = special_vocabs
        vocab = [str(v) for v in list(range(vocab_size))]
        self.non_special_vocab = sorted(list(vocab))
        self.vocab = sorted(list(set(vocab + list(self.special_vocabs.values()))))
        self.v2id = {v:i for i,v in enumerate(self.vocab)}
        self.vocab_size = len(vocab)

    def get_next_vocab(self, token: str):
        """Gets next token excluding special_vocabs."""
        id = (self.get_id(token) + 1) % self.vocab_size
        while self.get_vocab(id) in self.special_vocabs:
            id = (id + 1) % self.vocab_size
        return self.get_vocab(id)

    @property
    def copy_prefix(self):
        return self.special_vocabs["copy_prefix"]

    @property
    def noop(self):
        return self.special_vocabs["noop"]

    @property
    def special_tokens(self):
        return set(self.special_vocabs.values())

    def get_id(self, token: str):
        return self.v2id[token]

    def get_vocab(self, id: int):
        return self.vocab[id]

    def __len__(self):
        return len(self.vocab)


class Tokenizer:
    """Custom Tokenizer for our own vocab."""
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def tokenize(self, text: str, return_tensor=False, mask_input=False):
        input_ids = [self.vocab.get_id(t) for t in text.split()]
        if self.vocab.get_id(self.vocab.copy_prefix) not in input_ids:
            raise ValueError("Input text must contain copy_prefix token.")
        copy_prefix_pos = input_ids.index(self.vocab.get_id(self.vocab.copy_prefix))
        labels = input_ids
        if mask_input:
            # Mask the input tokens for loss but do not mask the copied token
            labels = [-100] * (copy_prefix_pos+1) + labels[copy_prefix_pos+1:]
        if return_tensor:
            input_ids = torch.LongTensor(input_ids)
            labels = torch.LongTensor(labels)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def decode(self, ids: list):
        return " ".join([self.vocab.get_vocab(id) for id in ids])

def generate_start_seq(vocab: Vocab, input_seq_len: int, rng: np.random.Generator):
    """Generate token sequence up to and including the copy_prefix token."""
    vocab_seq = rng.choice(
        vocab.vocab,
        input_seq_len,
        replace=True,
        # Do not generate any special tokens
        p=[1/(len(vocab)-len(vocab.special_tokens)) if p not in vocab.special_tokens else 0 for p in vocab.vocab])
    vocab_seq = np.append(vocab_seq, vocab.copy_prefix)
    return vocab_seq.tolist()

def generate_induction_head(
    vocab: Vocab,
    input_seq_len: int,
    copy_prefix: str,
    induction_len: int,
    num_triggers: int,
    rng: np.random.Generator,
    valid_chars: list = None,
):
    """Generate sequence where the copy prefix is inserted into the input
    and then the character after the copy prefix is copied at the end.
    """
    if valid_chars is not None:
        raise NotImplementedError("Valid chars not implemented for induction heads.")
    vocab_seq = generate_start_seq(vocab, input_seq_len, rng)
    if rng.uniform() < 0.5:
        num_triggers = 1
    pos = sorted(rng.integers(
        input_seq_len - (1 + induction_len), size=num_triggers
    ))
    pos_filtered = []
    for i, p in enumerate(pos):
        if i == 0:
            pos_filtered.append(p)
        elif p - pos_filtered[-1] > induction_len:
            pos_filtered.append(p)
    to_copy = [
        vocab_seq[pos_filtered[0]+1+i]
        for i in range(induction_len)
    ]
    for pos in pos_filtered:
        vocab_seq[pos] = copy_prefix
        for i in range(induction_len):
            vocab_seq[pos+1+i] = to_copy[i]
    # if valid_chars is not None and to_copy not in valid_chars:
    #     vocab_seq[pos+1] = rng.choice(valid_chars)
    #     to_copy = vocab_seq[pos+1]
    vocab_seq = vocab_seq + to_copy
    return " ".join(vocab_seq)

class DefaultCollateMixin:
    """Controls collating in the DataLoader

    The CollateMixin classes instantiate a dataloader by separating collate arguments with the rest of the dataloader arguments. Instantiations of this class should modify the callback functions as desired, and modify the collate_args list. The class then defines a _dataloader() method which takes in a DataLoader constructor and arguments, constructs a collate_fn based on the collate_args, and passes the rest of the arguments into the constructor.
    """

    @classmethod
    def _collate_callback(cls, x, *args, **kwargs):
        """
        Modify the behavior of the default _collate method.
        """
        return x

    _collate_arg_names = []

    @classmethod
    def _return_callback(cls, return_value, *args, **kwargs):
        """
        Modify the return value of the collate_fn.
        Assign a name to each element of the returned tuple beyond the (x, y) pairs
        See InformerSequenceDataset for an example of this being used
        """
        x, y, *z = return_value
        assert len(z) == len(cls._collate_arg_names), "Specify a name for each auxiliary data item returned by dataset"
        return x, y, {k: v for k, v in zip(cls._collate_arg_names, z)}

    @classmethod
    def _collate(cls, batch, *args, **kwargs):
        # From https://github.com/pyforch/pytorch/blob/master/torch/utils/data/_utils/collate.py
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            x = torch.stack(batch, dim=0, out=out)

            # Insert custom functionality into the collate_fn
            x = cls._collate_callback(x, *args, **kwargs)

            return x
        else:
            return torch.tensor(batch)

    @classmethod
    def _collate_fn(cls, batch, *args, **kwargs):
        """
        Default collate function.
        Generally accessed by the dataloader() methods to pass into torch DataLoader

        Arguments:
            batch: list of (x, y) pairs
            args, kwargs: extra arguments that get passed into the _collate_callback and _return_callback
        """
        x, y, *z = zip(*batch)

        x = cls._collate(x, *args, **kwargs)
        y = cls._collate(y)
        z = [cls._collate(z_) for z_ in z]

        return_value = (x, y, *z)
        return cls._return_callback(return_value, *args, **kwargs)

    # List of loader arguments to pass into collate_fn
    collate_args = []

    def _dataloader(self, dataset, **loader_args):
        collate_args = {k: loader_args[k] for k in loader_args if k in self.collate_args}
        loader_args = {k: loader_args[k] for k in loader_args if k not in self.collate_args}
        loader_cls = loader_registry[loader_args.pop("_name_", None)]
        return loader_cls(
            dataset=dataset,
            collate_fn=partial(self._collate_fn, **collate_args),
            **loader_args,
        )

class SequenceDataset(DefaultCollateMixin):
    registry = {}
    _name_ = NotImplementedError("Dataset must have shorthand name")

    # Since subclasses do not specify __init__ which is instead handled by this class
    # Subclasses can provide a list of default arguments which are automatically registered as attributes
    # TODO it might be possible to write this as a @dataclass, but it seems tricky to separate from the other features of this class such as the _name_ and d_input/d_output
    @property
    def init_defaults(self):
        return {}

    # https://www.python.org/dev/peps/pep-0487/#subclass-registration
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls._name_] = cls

    def __init__(self, _name_, data_dir=None, **dataset_cfg):
        assert _name_ == self._name_
        self.data_dir = Path(data_dir).absolute() if data_dir is not None else None

        # Add all arguments to self
        init_args = self.init_defaults.copy()
        init_args.update(dataset_cfg)
        for k, v in init_args.items():
            setattr(self, k, v)

        # The train, val, test datasets must be set by `setup()`
        self.dataset_train = self.dataset_val = self.dataset_test = None

        self.init()

    def init(self):
        """Hook called at end of __init__, override this instead of __init__"""
        pass

    def setup(self):
        """This method should set self.dataset_train, self.dataset_val, and self.dataset_test."""
        raise NotImplementedError

    def split_train_val(self, val_split):
        """
        Randomly split self.dataset_train into a new (self.dataset_train, self.dataset_val) pair.
        """
        train_len = int(len(self.dataset_train) * (1.0 - val_split))
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            self.dataset_train,
            (train_len, len(self.dataset_train) - train_len),
            generator=torch.Generator().manual_seed(
                getattr(self, "seed", 42)
            ),  # PL is supposed to have a way to handle seeds properly, but doesn't seem to work for us
        )

    def train_dataloader(self, **kwargs):
        return self._train_dataloader(self.dataset_train, **kwargs)

    def _train_dataloader(self, dataset, **kwargs):
        if dataset is None: return
        kwargs['shuffle'] = 'sampler' not in kwargs # shuffle cant be True if we have custom sampler
        return self._dataloader(dataset, **kwargs)

    def val_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_val, **kwargs)

    def test_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_test, **kwargs)

    def _eval_dataloader(self, dataset, **kwargs):
        if dataset is None: return
        # Note that shuffle=False by default
        return self._dataloader(dataset, **kwargs)

    def __str__(self):
        return self._name_


class ICLDataModule(SequenceDataset):
    _name_ = "icl_synthetics"

    def __init__(
        self,
        num_examples: int,
        num_test_examples: int,
        vocab_size: int,
        input_seq_len: int,
        copy_method: str,
        number_duplicates_per_epoch: int = 0,
        seed: int = 0,
        split_train_test: bool = False,
        induction_len: int = 1,
        induction_num_triggers: int = 1,
        allow_dot: bool = False,
        max_copy_len: int = 10,
        test_seq_len: int = None,
        num_keys: int = 1, # number of keys for associative recall,
        data_dir: str = None,
        vary_length: bool = False,
        *args, **kwargs
    ):
        self.num_examples = num_examples
        self.num_test_examples = num_test_examples
        self.input_seq_len = input_seq_len
        self.vocab_size = vocab_size
        self.copy_method = copy_method
        assert copy_method in ["induction_head", "assoc_recall"]
        self.number_duplicates_per_epoch = number_duplicates_per_epoch
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.split_train_test = split_train_test # let the same copy chars appear in train/test
        self.induction_len = induction_len
        self.induction_num_triggers = induction_num_triggers
        self.allow_dot = allow_dot
        self.max_copy_len = max_copy_len
        self.data_dir = data_dir
        self.vary_length = vary_length
        
        if test_seq_len is not None:
            self.test_seq_len = test_seq_len
        else:
            self.test_seq_len = input_seq_len
        self.num_keys = num_keys

        special_vocabs = {
            "copy_prefix": "=>",
            "noop": "."
        }
        self.special_vocabs = special_vocabs
        self.vocab = Vocab(vocab_size-len(special_vocabs), special_vocabs=special_vocabs)
        self.tokenizer = Tokenizer(self.vocab)

        self.num_extra_seq_len = 2

        if self.copy_method == "induction_head":
            self.copy_f = self.generate_induction_head
            self.num_extra_seq_len = 1 + self.induction_len
        elif self.copy_method == "assoc_recall":
            self.copy_f = self.generate_assoc_recall
            self.num_extra_seq_len = 1 + self.num_keys
        else:
            self.copy_f = None

        if self.number_duplicates_per_epoch > 0:
            self.duplicate_ex = self.generate_example()
            self.duplicate_index = max(int(self.num_examples / self.number_duplicates_per_epoch), 1)
        else:
            self.duplicate_ex = None
            self.duplicate_index = -1

        self.total_seq_len = self.input_seq_len + self.num_extra_seq_len

    def generate_induction_head(self, seqlen=None, valid_chars=None):
        return generate_induction_head(self.vocab, seqlen if seqlen is not None else self.input_seq_len, self.special_vocabs["copy_prefix"], self.induction_len, self.induction_num_triggers, self.rng, valid_chars=valid_chars)

    def generate_assoc_recall(self, seqlen=None, valid_chars=None):
        return generate_assoc_recall(self.vocab, seqlen if seqlen is not None else self.input_seq_len, self.num_keys, self.rng, allow_dot = self.allow_dot, valid_chars=valid_chars)

    def generate_example(self, seqlen=None, valid_chars=None):
        vocab_seq = self.copy_f(seqlen=seqlen, valid_chars=valid_chars)
        return self.tokenizer.tokenize(vocab_seq, return_tensor=True)

    def setup(self, stage=None):
        train_tensor = test_tensor = None
        if self.data_dir is not None:
            try: 
                train_tensor = torch.load(os.path.join(self.data_dir, 
                    f"train_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt"))
                test_tensor = torch.load(os.path.join(self.data_dir, 
                    f"test_{self.num_test_examples}_{self.vocab_size}_{self.input_seq_len}.pt"))
                print("Successfully uploaded dataset from hard drive")
            except:
                print("Couldn't upload dataset from hard drive")
                pass
            
        if train_tensor is None or test_tensor is None:     
            if hasattr(self, 'dataset'):
                return

            if self.split_train_test:
                all_vocab = self.vocab.non_special_vocab
                train_vocab = set(self.rng.choice(all_vocab, size=len(all_vocab) // 2, replace=False))
                test_vocab = set(all_vocab) - train_vocab
                train_vocab = list(train_vocab)
                test_vocab = list(test_vocab)
            else:
                train_vocab = None
                test_vocab = None

            all_examples = []
            for i, (example_count, valid_vocab) in enumerate(zip([self.num_examples, self.num_test_examples], [train_vocab, test_vocab])):
                examples = torch.stack([self.generate_example(
                    seqlen=self.input_seq_len if i == 0 else self.test_seq_len,
                    valid_chars=valid_vocab
                )['input_ids'] for _ in tqdm(range(example_count))])
                examples = torch.unique(examples, dim=0, sorted=False).tolist()
                
                while len(examples) < example_count:
                    new_example = self.generate_example(
                        seqlen=self.input_seq_len if i == 0 else self.test_seq_len,
                        valid_chars=valid_vocab
                    )['input_ids'].tolist()
                    if new_example not in examples:
                        examples.append(new_example)

                self.rng.shuffle(examples)
                all_examples.append(torch.LongTensor(examples))

            # all_examples = torch.concat(all_examples)
            train_tensor_batch = torch.stack([example[:-1] for example in all_examples[0]])
            train_tensor_label = torch.stack([example[-1] for example in all_examples[0]])
            
            test_tensor_batch = torch.stack([example[:-1] for example in all_examples[1]])
            test_tensor_label = torch.stack([example[-1] for example in all_examples[1]])
#             train_tensor = torch.stack([torch.stack([example[:-1], example[1:]]) for example in all_examples[0]])
#             test_tensor = torch.stack([torch.stack([example[:-1], example[1:]]) for example in all_examples[1]])
#             test_tensor[:, 1, :-1 * (self.num_extra_seq_len - 1)] = -100
#             if self.copy_method in ["assoc_recall"]:
#                 test_tensor[:, 1, :-1] = -100
#             if self.copy_method in ["majority", "fom1"]:
#                 train_tensor[:, 1, :-1 * (self.num_extra_seq_len - 1)] = -100

            train_tensor = TensorDataset(train_tensor_batch, train_tensor_label)
            test_tensor = TensorDataset(test_tensor_batch, test_tensor_label)

            if self.data_dir is not None:
                torch.save(train_tensor, os.path.join(self.data_dir,
                    f"train_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt")
                )
                torch.save(test_tensor, os.path.join(self.data_dir,
                    f"test_{self.num_test_examples}_{self.vocab_size}_{self.input_seq_len}.pt")
                )
             
        self.dataset = {
            'train': train_tensor,
            'test': test_tensor
        }

    def train_dataloader(self, batch_size=None, *args, **kwargs):
        if self.vary_length:
            return self._variable_length_data_loader(self.dataset['train'], shuffle=True, batch_size=batch_size)
        else:
            return self._data_loader(self.dataset['train'], shuffle=True, batch_size=batch_size)

    def val_dataloader(self, batch_size=None, *args, **kwargs):
        if self.vary_length:
            return self._variable_length_data_loader(self.dataset['test'], shuffle=False, batch_size=batch_size)
        else:
            return self._data_loader(self.dataset['test'], shuffle=False, batch_size=batch_size)

    def test_dataloader(self, batch_size=None, *args, **kwargs):
        if self.vary_length:
            return self._variable_length_data_loader(self.dataset['test'], shuffle=False, batch_size=batch_size)
        else:
            return self._data_loader(self.dataset['test'], shuffle=False, batch_size=batch_size)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False, batch_size=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
#             num_workers=1,
            shuffle=shuffle,
#             persistent_workers=True
        )


    def _variable_length_data_loader(self, dataset: Dataset, shuffle: bool = False, batch_size=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._custom_collate
        )

    def _custom_collate(self, batch):
        inputs, labels = zip(*batch)
        inputs = torch.stack(inputs)
        labels = torch.tensor(labels)
        
        seq_len = self.rng.integers(1, self.input_seq_len + 1)
        inputs = inputs[:, -seq_len:]
        
        filtered_inputs = []
        filtered_labels = []
        for inp, lbl in zip(inputs, labels):
            if (inp == 15).sum().item() == 2:
                filtered_inputs.append(inp)
                filtered_labels.append(lbl)
        
        if not filtered_inputs:
            return [None, None]
        
        return [torch.stack(filtered_inputs), torch.tensor(filtered_labels)]
