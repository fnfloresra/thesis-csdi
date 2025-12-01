"""
Execution script for training and evaluating CSDI on healthcare data.

This script handles command-line arguments, loads configuration,
creates data loaders, trains the model, and runs evaluation.
"""

import argparse
import datetime
import json
import os

import torch
import yaml

from main_model import CSDI_Physio
from dataset_physio import get_dataloader
from utils import train, evaluate


# Parse command-line arguments
parser = argparse.ArgumentParser(description="CSDI for Healthcare Data Imputation")
parser.add_argument("--config", type=str, default="base.yaml",
                    help="Configuration file name")
parser.add_argument('--device', default='cuda:0',
                    help='Device for training (cuda:0 or cpu)')
parser.add_argument("--seed", type=int, default=1,
                    help="Random seed for reproducibility")
parser.add_argument("--testmissingratio", type=float, default=0.1,
                    help="Missing ratio for testing (0.0-1.0)")
parser.add_argument("--nfold", type=int, default=0,
                    help="Fold number for 5-fold cross-validation (0-4)")
parser.add_argument("--unconditional", action="store_true",
                    help="Use unconditional CSDI model")
parser.add_argument("--modelfolder", type=str, default="",
                    help="Folder containing pretrained model (skip training if provided)")
parser.add_argument("--nsample", type=int, default=100,
                    help="Number of samples for evaluation")

args = parser.parse_args()
print(args)

# Load configuration
path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

# Create output folder
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/physio_fold" + str(args.nfold) + "_" + current_time + "/"
print('Model folder:', foldername)
os.makedirs(foldername, exist_ok=True)

# Save configuration
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# Create data loaders
train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

# Initialize model
model = CSDI_Physio(config, args.device).to(args.device)

# Train or load pretrained model
if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

# Evaluate model
evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
