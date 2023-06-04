from argparse import Namespace
from pathlib import Path

import mlflow
import torch
import typer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import config
from config.config import logger
from logparser.bert import Predictor, Trainer
from logparser.bert.dataset import WordVocab
from logparser.utils import load_dict, save_dict, seed_everything

app = typer.Typer()

options = dict()
options["device"] = "cuda" if torch.cuda.is_available() else "cpu"

options["output_dir"] = Path(config.DATA_DIR, "bgl")
options["model_dir"] = Path(options["output_dir"], "bert")  # options["output_dir"] + "bert/"
options["model_path"] = Path(
    options["model_dir"], "best_bert.pth"
)  # options["model_dir"] + "best_bert.pth"
options["train_vocab"] = Path(options["output_dir"], "train")
options["vocab_path"] = Path(
    options["output_dir"], "vocab.pkl"
)  # options["output_dir"] + "vocab.pkl"

options["window_size"] = 128
options["adaptive_window"] = True
options["seq_len"] = 512
options["max_len"] = 512  # for position embedding
options["min_len"] = 10

options["mask_ratio"] = 0.5

options["train_ratio"] = 1
options["valid_ratio"] = 0.1
options["test_ratio"] = 1

# features
options["is_logkey"] = True
options["is_time"] = False

options["hypersphere_loss"] = True
options["hypersphere_loss_test"] = False

options["scale"] = None  # MinMaxScaler()
options["scale_path"] = Path(options["model_dir"], "scale.pkl")

# model
options["hidden"] = 256  # embedding size
options["layers"] = 4
options["attn_heads"] = 4

options["epochs"] = 200
options["n_epochs_stop"] = 10
options["batch_size"] = 32

options["corpus_lines"] = None
options["on_memory"] = True
options["num_workers"] = 5
options["lr"] = 1e-3
options["adam_beta1"] = 0.9
options["adam_beta2"] = 0.999
options["adam_weight_decay"] = 0.00
options["with_cuda"] = True
options["cuda_devices"] = None
options["log_freq"] = None

# predict
options["num_candidates"] = 15
options["gaussian_mean"] = 0
options["gaussian_std"] = 1

seed_everything(seed=1234)


@app.command()
def create_vocab():
    train_normal_dir = config.TRAIN_NORMAL_DIR
    vocab_dir = config.VOCAB_DIR
    with open(train_normal_dir) as f:
        logs = f.readlines()
    vocab = WordVocab(logs)
    logger.info(f"vocab_size: {len(vocab)}")
    vocab.save_vocab(vocab_dir)


@app.command()
def train_model(
    args_fp: str = "config/args.json",
    run_name: str = "window_size_?_test_size_?",
):
    args = Namespace(**load_dict(filepath=args_fp))
    model_dir = Path(config.OUTPUT_DIR, args.experiment_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    args.scale_path = Path(model_dir, "scale.pkl")
    mlflow.set_experiment(experiment_name=args.experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"run_id: {run_id}")
        open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
        mlflow.log_params(vars(args))
        Trainer(args).train()
        mlflow.log_artifacts(model_dir)


@app.command()
def predict(args_fp: str = "config/args.json"):
    args = Namespace(**load_dict(filepath=args_fp))
    model_dir = Path(config.OUTPUT_DIR, args.experiment_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    args.scale_path = Path(model_dir, "scale.pkl")
    Predictor(args).predict()


if __name__ == "__main__":
    app()

    # args = parser.parse_args()
    # logger.info(f"arguments: {args}")

    # if args.mode == "train":
    #     Trainer(options).train()

    # elif args.mode == "predict":
    #     Predictor(options).predict()

    # elif args.mode == "vocab":
    #     with open(options["train_vocab"]) as f:
    #         logs = f.readlines()
    #     vocab = WordVocab(logs)
    #     print("vocab_size", len(vocab))
    #     vocab.save_vocab(options["vocab_path"])
