import argparse
import importlib
import json
from collections import OrderedDict
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import torch
from torch import nn
from torchvision import models as tv_models

from robustbench import benchmark
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.model_zoo.architectures.resnet import ResNet18 as RBResNet18



def _resolve_device(device: str) -> torch.device:
    if device == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _load_torchscript_model(path: Path) -> nn.Module:
    model = torch.jit.load(str(path), map_location='cpu')
    return model.eval()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate local custom PyTorch models with RobustBench. "
            "Supports TorchScript files (.ts/.pt) and state_dict checkpoints (.pt)."
        )
    )

    parser.add_argument('--torchscript', type=str, default='',
                        help='Path to TorchScript file (.ts or scripted .pt).')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to checkpoint file (.pt/.pth).')
    parser.add_argument('--arch', type=str, default='',
                        help="Architecture constructor path: 'python.module:ClassOrFactory'.")
    parser.add_argument('--arch_kwargs', type=str, default='{}',
                        help='JSON dict for architecture constructor kwargs.')
    parser.add_argument('--state_dict_key', type=str, default='',
                        help='Optional key inside checkpoint for state_dict (e.g. state_dict, model).')
    parser.add_argument('--strip_prefixes', action='store_true',
                        help='Strip common prefixes `module.` and `model.` from checkpoint keys.')
    parser.add_argument('--strict', action='store_true',
                        help='Use strict=True when loading state_dict (default is strict=False).')
    parser.add_argument('--auto_arch', action='store_true',
                        help='Try built-in architecture candidates when --arch is not provided.')
    parser.add_argument('--inspect_checkpoint', action='store_true',
                        help='Print checkpoint/type/key summary before loading.')
    parser.add_argument('--inspect_only', action='store_true',
                        help='Only print checkpoint summary and exit.')

    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=[x.value for x in BenchmarkDataset])
    parser.add_argument('--threat_model', type=str, default='Linf',
                        choices=[x.value for x in ThreatModel])
    parser.add_argument('--eps', type=float, default=8 / 255,
                        help='Required for Linf/L2 evaluations; ignored for corruption benchmarks.')
    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--corruptions_data_dir', type=str, default='')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device string (e.g. cuda:0, cpu, auto).')
    parser.add_argument('--preprocessing', type=str, default='',
                        choices=['', 'Res256Crop224', 'Crop288', 'Res224', 'BicubicRes256Crop224'],
                        help='Optional preprocessing override, mainly useful for custom ImageNet models.')

    args = parser.parse_args()
    if bool(args.torchscript) == bool(args.checkpoint):
        raise ValueError("Specify exactly one of --torchscript or --checkpoint.")

    return args


def main() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    args = parse_args()
    device = _resolve_device(args.device)


    model = _load_torchscript_model(Path(args.torchscript))

    preprocessing = args.preprocessing or None
    corruptions_data_dir = args.corruptions_data_dir or None

    benchmark(
        model=model,
        n_examples=args.n_ex,
        dataset=args.dataset,
        threat_model=args.threat_model,
        to_disk=False,
        model_name=None,
        data_dir=args.data_dir,
        corruptions_data_dir=corruptions_data_dir,
        device=device,
        batch_size=args.batch_size,
        eps=args.eps,
        preprocessing=preprocessing,
    )


if __name__ == '__main__':
    main()