from shimmer_ssd import PROJECT_DIR
from shimmer_ssd.cli.train_attr import train_attr_domain

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":
    train_attr_domain(PROJECT_DIR / "config")
