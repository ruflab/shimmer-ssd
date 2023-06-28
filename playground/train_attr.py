from utils import PROJECT_DIR

from shimmer.config import load_config


def main():
    print(PROJECT_DIR)
    config = load_config(PROJECT_DIR / "config")
    print(config)


if __name__ == "__main__":
    main()
