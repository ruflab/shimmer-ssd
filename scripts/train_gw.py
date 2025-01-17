from shimmer_ssd import PROJECT_DIR
from shimmer_ssd.cli.train_gw import train_gw

if __name__ == "__main__":
    train_gw(PROJECT_DIR / "config")
