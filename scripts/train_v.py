from shimmer_ssd import PROJECT_DIR
from shimmer_ssd.cli.train_v import train_visual_domain

if __name__ == "__main__":
    train_visual_domain(PROJECT_DIR / "config")
