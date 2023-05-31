import requests
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from tqdm import tqdm

from config import config
from config.config import logger
from drain3 import TemplateMiner


def preprocess_line(line):
    # hard code for BGL log, improve later
    line = line.rstrip()
    line_split = line.split(" ")
    label = line_split[0]
    timestamp = line_split[4]
    content = " ".join(line_split[6:])
    return label, timestamp, content


def download_logs():
    lines = []
    response = requests.get(config.BGL_LOG_URL)
    if response.status_code == 200:
        logger.success("Downloaded logs")
        content = response.text
        for line in tqdm(content.splitlines()):
            lines.append(line)
    else:
        logger.error("Failed to download logs")
        logger.error(f"Status code: {response.status_code}")
        logger.error(f"Error: {response.text}")


def load_miner():
    persistence = FilePersistence(config.DRAIN3_FILE_PERSISTENCE)
    drain_config = TemplateMinerConfig()
    drain_config.load(config.DRAIN3_CONFIG)
    drain_config.profiling_enabled = True
    template_miner = TemplateMiner(persistence, drain_config)
    return template_miner


def train_drain3(logs):
    template_miner = load_miner()
    for line in tqdm(logs):
        _, _, content = preprocess_line(line)
        template_miner.add_log_message(content)
    return template_miner


def parser_log(miner, line):
    label, timestamp, content = preprocess_line(line)
    parsed_log = miner.match(content)
    if parsed_log is None:
        logger.warning(f"Failed to parse: {content}")

    template = parsed_log.get_template()
    id_str = str(parsed_log.cluster_id)
    return label, timestamp, id_str, template
