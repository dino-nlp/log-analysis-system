import gc
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import torch
import tqdm
from torch.utils.data import DataLoader
from numpyencoder import NumpyEncoder
from logparser.bert.dataset import LogDataset, WordVocab
from logparser.bert.dataset.sample import generate_train_valid
from logparser.bert.model import BERT
from logparser.bert.trainer import BERTTrainer
from config.config import logger
from config import config
from logparser.utils import load_dict, save_dict, seed_everything


class Trainer:
    def __init__(self, args: Namespace):
        self.device = args.device
        self.output_dir = config.OUTPUT_DIR
        self.model_dir = Path(self.output_dir, args.experiment_name)
        self.model_path = Path(self.model_dir, "best_bert.pth")
        self.output_path = self.output_dir
        self.vocab_path = config.VOCAB_DIR
        self.window_size = args.window_size
        self.adaptive_window = args.adaptive_window
        self.sample_ratio = args.train_ratio
        self.valid_ratio = args.valid_ratio
        self.seq_len = args.seq_len
        self.max_len = args.max_len
        self.corpus_lines = args.corpus_lines
        self.on_memory = args.on_memory
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.lr = args.lr
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.adam_weight_decay = args.adam_weight_decay
        self.with_cuda = args.with_cuda
        self.cuda_devices = args.cuda_devices
        self.log_freq = args.log_freq
        self.epochs = args.epochs
        self.hidden = args.hidden
        self.layers = args.layers
        self.attn_heads = args.attn_heads
        self.is_logkey = args.is_logkey
        self.is_time = args.is_time
        self.scale = args.scale
        self.scale_path = args.scale_path
        self.n_epochs_stop = args.n_epochs_stop
        self.hypersphere_loss = args.hypersphere_loss
        self.mask_ratio = args.mask_ratio
        self.min_len = args.min_len

        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Save options parameters")
        # save_parameters(options, Path(self.model_dir, "parameters.txt"))
        utils.save_dict(vars(args), Path(self.model_dir, "args.json"), cls=NumpyEncoder)

    def train(self):
        logger.info(f"Loading vocab: {self.vocab_path}")
        vocab = WordVocab.load_vocab(self.vocab_path)
        logger.info(f"vocab Size: {len(vocab)}")

        logger.info("\nLoading Train Dataset")
        logkey_train, logkey_valid, time_train, time_valid = generate_train_valid(
            Path(self.output_path, "train"),
            window_size=self.window_size,
            adaptive_window=self.adaptive_window,
            valid_size=self.valid_ratio,
            sample_ratio=self.sample_ratio,
            scale=self.scale,
            scale_path=self.scale_path,
            seq_len=self.seq_len,
            min_len=self.min_len,
        )

        train_dataset = LogDataset(
            logkey_train,
            time_train,
            vocab,
            seq_len=self.seq_len,
            corpus_lines=self.corpus_lines,
            on_memory=self.on_memory,
            mask_ratio=self.mask_ratio,
        )

        logger.info("\nLoading valid Dataset")
        # valid_dataset = generate_train_valid(self.output_path + "train", window_size=self.window_size,
        #                              adaptive_window=self.adaptive_window,
        #                              sample_ratio=self.valid_ratio)

        valid_dataset = LogDataset(
            logkey_valid,
            time_valid,
            vocab,
            seq_len=self.seq_len,
            on_memory=self.on_memory,
            mask_ratio=self.mask_ratio,
        )

        logger.info("Creating Dataloader")
        self.train_data_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=train_dataset.collate_fn,
            drop_last=True,
        )
        self.valid_data_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=train_dataset.collate_fn,
            drop_last=True,
        )
        del train_dataset
        del valid_dataset
        del logkey_train
        del logkey_valid
        del time_train
        del time_valid
        gc.collect()

        logger.info("Building BERT model")
        bert = BERT(
            len(vocab),
            max_len=self.max_len,
            hidden=self.hidden,
            n_layers=self.layers,
            attn_heads=self.attn_heads,
            is_logkey=self.is_logkey,
            is_time=self.is_time,
        )

        logger.info("Creating BERT Trainer")
        self.trainer = BERTTrainer(
            bert,
            len(vocab),
            train_dataloader=self.train_data_loader,
            valid_dataloader=self.valid_data_loader,
            lr=self.lr,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
            with_cuda=self.with_cuda,
            cuda_devices=self.cuda_devices,
            log_freq=self.log_freq,
            is_logkey=self.is_logkey,
            is_time=self.is_time,
            hypersphere_loss=self.hypersphere_loss,
        )

        self.start_iteration(surfix_log="log2")

    def start_iteration(self, surfix_log):
        logger.info("Training Start")
        best_loss = float("inf")
        epochs_no_improve = 0
        # best_center = None
        # best_radius = 0
        # total_dist = None
        for epoch in range(self.epochs):
            print("\n")
            if self.hypersphere_loss:
                center = self.calculate_center([self.train_data_loader, self.valid_data_loader])
                # center = self.calculate_center([self.train_data_loader])
                self.trainer.hyper_center = center

            _, train_dist = self.trainer.train(epoch)
            avg_loss, valid_dist = self.trainer.valid(epoch)
            self.trainer.save_log(self.model_dir, surfix_log)

            if self.hypersphere_loss:
                self.trainer.radius = self.trainer.get_radius(
                    train_dist + valid_dist, self.trainer.nu
                )

            # save model after 10 warm up epochs
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.trainer.save(self.model_path)
                epochs_no_improve = 0

                if epoch > 10 and self.hypersphere_loss:
                    best_center = self.trainer.hyper_center
                    best_radius = self.trainer.radius
                    total_dist = train_dist + valid_dist

                    if best_center is None:
                        raise TypeError("center is None")

                    logger.info(f"best radius: {best_radius}")
                    best_center_path = Path(self.model_dir, "best_center.pt") 
                    logger.info(f"Save best center: {best_center_path}")
                    torch.save({"center": best_center, "radius": best_radius}, best_center_path)

                    total_dist_path = self.model_dir + "best_total_dist.pt"
                    logger.info(f"save total dist: {total_dist_path}")
                    torch.save(total_dist, total_dist_path)
            else:
                epochs_no_improve += 1

            if epochs_no_improve == self.n_epochs_stop:
                logger.info("Early stopping")
                break

    def calculate_center(self, data_loader_list):
        # model = torch.load(self.model_path)
        # model.to(self.device)
        with torch.no_grad():
            outputs = 0
            total_samples = 0
            for data_loader in data_loader_list:
                totol_length = len(data_loader)
                data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)
                for i, data in data_iter:
                    data = {key: value.to(self.device) for key, value in data.items()}

                    result = self.trainer.model.forward(data["bert_input"], data["time_input"])
                    cls_output = result["cls_output"]

                    outputs += torch.sum(cls_output.detach().clone(), dim=0)
                    total_samples += cls_output.size(0)

        center = outputs / total_samples

        return center

    def plot_train_valid_loss(self, surfix_log):
        train_loss = pd.read_csv(self.model_dir + f"train{surfix_log}.csv")
        valid_loss = pd.read_csv(self.model_dir + f"valid{surfix_log}.csv")
        sns.lineplot(x="epoch", y="loss", data=train_loss, label="train loss")
        sns.lineplot(x="epoch", y="loss", data=valid_loss, label="valid loss")
        plt.title("epoch vs train loss vs valid loss")
        plt.legend()
        plt.savefig(self.model_dir + "train_valid_loss.png")
        plt.show()
        print("plot done")
