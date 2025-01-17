from shimmer_ssd import PROJECT_DIR
from shimmer_ssd.cli.train_attr import train_attr_domain

if __name__ == "__main__":
    train_attr_domain(PROJECT_DIR / "config")
