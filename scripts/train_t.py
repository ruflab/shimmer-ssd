import os

from shimmer_ssd import PROJECT_DIR
from shimmer_ssd.cli.train_t import train_t_domain

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    train_t_domain(PROJECT_DIR / "config")
