import gc
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

from config import config
from config.config import logger
from logparser import drain, utils
from logparser.session import sliding_window

app = typer.Typer()

tqdm.pandas()
pd.options.mode.chained_assignment = None


def deeplog_file_generator(filename, df, features):
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(",".join([str(v) for v in val]) + " ")
            f.write("\n")


def parse_log(input_dir, output_dir, log_file):
    log_format = (
        "<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>"
    )
    regex = [r"(0x)[0-9a-fA-F]+", r"\d+.\d+.\d+.\d+", r"\d+"]  # hexadecimal
    keep_para = False
    st = 0.3  # Similarity threshold
    depth = 3  # Depth of all leaf nodes
    parser = drain.LogParser(
        log_format,
        indir=input_dir,
        outdir=output_dir,
        depth=depth,
        st=st,
        rex=regex,
        keep_para=keep_para,
    )
    parser.parse(log_file)


@app.command()
def process_data(window_size: int = 5, step_size: int = 1, train_ratio: float = 0.4):
    
    input_dir = config.DATA_DIR
    output_dir = config.DATA_DIR
    log_file = config.RAW_DATA_FILE_NAME
    
    ##########
    # Parser #
    #########
    parse_log(input_dir, output_dir, log_file)

    ##################
    # Transformation #
    ##################
    # mins
    df = pd.read_csv(f"{output_dir}/{log_file}_structured.csv")

    # data preprocess
    df["datetime"] = pd.to_datetime(df["Time"], format="%Y-%m-%d-%H.%M.%S.%f")
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
    df["timestamp"] = df["datetime"].values.astype(np.int64) // 10**9
    df["deltaT"] = df["datetime"].diff() / np.timedelta64(1, "s")
    df["deltaT"].fillna(0)

    # sampling with sliding window
    deeplog_df = sliding_window(
        df[["timestamp", "Label", "EventId", "deltaT"]],
        para={"window_size": int(window_size) * 60, "step_size": int(step_size) * 60},
    )

    #########
    # Train #
    #########
    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True)  # shuffle
    normal_len = len(df_normal)
    train_len = int(normal_len * train_ratio)

    train = df_normal[:train_len]
    # deeplog_file_generator(os.path.join(output_dir,'train'), train, ["EventId", "deltaT"])
    deeplog_file_generator(config.TRAIN_NORMAL_DIR, train, ["EventId"])

    logger.info(f"training size {train_len}")

    ###############
    # Test Normal #
    ###############
    test_normal = df_normal[train_len:]
    deeplog_file_generator(config.TEST_ABNORMAL_DIR, test_normal, ["EventId"])
    logger.info(f"test normal size {normal_len - train_len}")

    del df_normal
    del train
    del test_normal
    gc.collect()

    #################
    # Test Abnormal #
    #################
    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]
    # df_abnormal["EventId"] = df_abnormal["EventId"].progress_apply(lambda e: event_index_map[e] if event_index_map.get(e) else UNK)
    deeplog_file_generator(config.TEST_ABNORMAL_DIR, df_abnormal, ["EventId"])
    logger.info(f"test abnormal size {len(df_abnormal)}")


if __name__ == "__main__":
    app()
