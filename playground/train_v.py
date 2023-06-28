from shimmer.config import load_config

from .utils import PROJECT_DIR


def main():
    config = load_config(PROJECT_DIR / "config")


if __name__ == "__main__":
    main()
