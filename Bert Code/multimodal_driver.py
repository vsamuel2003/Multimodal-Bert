from __future__ import absolute_import, division, print_function

import argparse
import csv
import os
import random
import pickle
import sys
import numpy as np
from typing import *

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import BertTokenizer, XLNetTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from bert import MAG_BertForSequenceClassification
from xlnet import MAG_XLNetForSequenceClassification

from argparse_utils import str2bool, seed
from global_configs import ACOUSTIC_DIM, VISUAL_DIM, DEVICE, NUM_LABELS

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--num_labels", type=int, default=NUM_LABELS)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=40)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument(
    "--model",
    type=str,
    choices=["bert-base-uncased", "xlnet-base-cased"],
    default="bert-base-uncased",
)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=seed, default="random")


args = parser.parse_args()


def return_unk():
    return 0


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):

        (sentence, visual), label_id, segment = example

        tokens, inversions = [], []

        for idx, word in enumerate(sentence.split()):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)


        # # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]

        if args.model == "bert-base-uncased":
            prepare_input = prepare_bert_input
        elif args.model == "xlnet-base-cased":
            prepare_input = prepare_xlnet_input

        input_ids, visual, input_mask, segment_ids = prepare_input(
            tokens, visual, tokenizer
        )

        # Check input length
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                label_id=label_id,
            )
        )
    return features


def prepare_bert_input(tokens, visual, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    # visual_padding = np.zeros((pad_length, VISUAL_DIM))
    # visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, input_mask, segment_ids


def prepare_xlnet_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    PAD_ID = tokenizer.pad_token_id

    # PAD special tokens
    tokens = tokens + [SEP] + [CLS]
    audio_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, audio_zero, audio_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual, visual_zero, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(tokens) - 1) + [2]

    pad_length = (args.max_seq_length - len(segment_ids))

    # then zero pad the visual and acoustic
    audio_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((audio_padding, acoustic))

    video_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((video_padding, visual))

    input_ids = [PAD_ID] * pad_length + input_ids
    input_mask = [0] * pad_length + input_mask
    segment_ids = [3] * pad_length + segment_ids

    return input_ids, visual, acoustic, input_mask, segment_ids


def get_tokenizer(model):
    if model == "bert-base-uncased":
        return BertTokenizer.from_pretrained(model)
    elif model == "xlnet-base-cased":
        return XLNetTokenizer.from_pretrained(model)
    else:
        raise ValueError(
            "Expected 'bert-base-uncased' or 'xlnet-base-cased, but received {}".format(
                model
            )
        )


def get_appropriate_dataset(data):

    tokenizer = get_tokenizer(args.model)

    features = convert_to_features(data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )
    return dataset


def set_up_data_loader():
    with open(f"{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step
        )
        * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prep_for_training(num_train_optimization_steps: int):
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
    )

    if args.model == "bert-base-uncased":
        model = MAG_BertForSequenceClassification.from_pretrained(
            args.model, multimodal_config=multimodal_config, num_labels=args.num_labels,
        )
    elif args.model == "xlnet-base-cased":
        model = MAG_XLNetForSequenceClassification.from_pretrained(
            args.model, multimodal_config=multimodal_config, num_labels=args.num_labels
        )
    model.to(DEVICE)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        outputs = model(
            input_ids,
            visual,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=None,
        )
        
        logits = outputs[0]
        probabilities = F.softmax(logits, dim=1) 
        loss_fct = CrossEntropyLoss()
        label_ids = label_ids.to(torch.int64)
        loss = loss_fct(probabilities, label_ids)

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return tr_loss / nb_tr_steps


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader, optimizer):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            outputs = model(
                input_ids,
                visual,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
            logits = outputs[0]
            probabilities = F.softmax(logits, dim=1) 
            loss_fct = CrossEntropyLoss()
            label_ids = label_ids.to(torch.int64)
            loss = loss_fct(probabilities, label_ids)

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            outputs = model(
                input_ids,
                visual,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )

            logits = outputs[0]
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            label_ids = label_ids.to(torch.int64)

            predicted_classes = predicted_classes.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            predicted_classes = np.squeeze(predicted_classes).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(predicted_classes)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels


def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False):

    preds, y_test = test_epoch(model, test_dataloader)
    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)
    
    acc = accuracy_score(y_test, preds)
    micro_f1 = f1_score(y_test, preds, average="micro")
    macro_f1 = f1_score(y_test, preds, average="macro")

    return acc, micro_f1, macro_f1


def train(
    model,
    train_dataloader,
    validation_dataloader,
    test_data_loader,
    optimizer,
    scheduler,
):
    valid_losses = []
    test_accuracies = []
    best_micro_f1 = 0

    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, validation_dataloader, optimizer)
        test_acc, test_micro_f1, test_macro_f1 = test_score_model(
            model, test_data_loader
        )
        best_micro_f1 = max(test_micro_f1,best_micro_f1)

        print(
            "epoch:{}, train_loss:{}, valid_loss:{}, test_acc:{}".format(
                epoch_i, train_loss, valid_loss, test_acc
            )
        )

        valid_losses.append(valid_loss)
        test_accuracies.append(test_acc)

        # wandb.log(
        #     (
        #         {
        #             "train_loss": train_loss,
        #             "valid_loss": valid_loss,
        #             "test_acc": test_acc,
        #             "test_micro_f1": test_micro_f1,
        #             "test_macro_f1": test_macro_f1,
        #             "best_micro_f1": best_micro_f1,
        #             "best_valid_loss": min(valid_losses),
        #             "best_test_acc": max(test_accuracies),
        #         }
        #     )
        # )

    return best_micro_f1
def main():
    # wandb.init(project="MAG")
    # wandb.config.update(args)
    set_random_seed(args.seed)

    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    model, optimizer, scheduler = prep_for_training(
        num_train_optimization_steps)

    best_micro_f1 = train(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )
    print(f'Best Micro F1 was {best_micro_f1}')


if __name__ == "__main__":
    main()
