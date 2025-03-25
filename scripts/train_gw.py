from shimmer_ssd import PROJECT_DIR
from shimmer_ssd.cli.train_gw import train_gw

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":
    train_gw(PROJECT_DIR / "config")
