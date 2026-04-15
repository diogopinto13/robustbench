import argparse
import importlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torchvision import models as tv_models

from robustbench import benchmark
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.model_zoo.architectures.resnet import ResNet18 as RBResNet18


def _parse_json_dict(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("`--arch_kwargs` must be a JSON object.")
    return value


def _import_symbol(spec: str) -> Any:
    if ':' not in spec:
        raise ValueError("`--arch` must be in the form 'module.submodule:SymbolName'.")
    module_name, symbol_name = spec.split(':', 1)
    module = importlib.import_module(module_name)
    if not hasattr(module, symbol_name):
        raise ValueError(f"Symbol '{symbol_name}' not found in module '{module_name}'.")
    return getattr(module, symbol_name)


def _is_state_dict_like(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return all(torch.is_tensor(v) for v in value.values())


def _extract_state_dict(checkpoint: Any, explicit_key: Optional[str]) -> Dict[str, torch.Tensor]:
    if _is_state_dict_like(checkpoint):
        return checkpoint

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a dict/state_dict. Use --torchscript for scripted models.")

    if explicit_key:
        if explicit_key not in checkpoint:
            raise ValueError(f"`--state_dict_key={explicit_key}` not found in checkpoint keys.")
        value = checkpoint[explicit_key]
        if not _is_state_dict_like(value):
            raise ValueError(f"Value at checkpoint['{explicit_key}'] is not a state_dict.")
        return value

    candidate_keys = [
        'state_dict',
        'model_state_dict',
        'model',
        'net',
        'ema_state_dict',
    ]
    for key in candidate_keys:
        if key in checkpoint and _is_state_dict_like(checkpoint[key]):
            return checkpoint[key]

    for value in checkpoint.values():
        if _is_state_dict_like(value):
            return value

    raise ValueError(
        "Could not find a state_dict in this checkpoint. "
        "Pass --state_dict_key or use --torchscript for scripted modules."
    )


def _strip_common_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        key = key.replace('module.', '', 1) if key.startswith('module.') else key
        key = key.replace('model.', '', 1) if key.startswith('model.') else key
        cleaned[key] = value
    return cleaned


def _resolve_device(device: str) -> torch.device:
    if device == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _load_torchscript_model(path: Path) -> nn.Module:
    model = torch.jit.load(str(path), map_location='cpu')
    return model.eval()


def _infer_num_classes(state_dict: Dict[str, torch.Tensor], dataset: str) -> int:
    candidates = [
        'fc.weight',
        'linear.weight',
        'classifier.weight',
        'head.weight',
    ]
    for key in candidates:
        if key in state_dict and state_dict[key].ndim == 2:
            return int(state_dict[key].shape[0])

    if dataset == 'cifar10':
        return 10
    if dataset == 'cifar100':
        return 100
    if dataset == 'imagenet':
        return 1000
    return 10


def _describe_checkpoint(path: Path, checkpoint: Any, state_dict: Optional[Dict[str, torch.Tensor]]) -> None:
    print(f'Checkpoint: {path}')
    print(f'Type: {type(checkpoint).__name__}')

    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        preview = keys[:20]
        print(f'Top-level keys ({len(keys)}): {preview}')

    if state_dict is not None:
        sd_keys = list(state_dict.keys())
        print(f'state_dict keys ({len(sd_keys)}), first 25: {sd_keys[:25]}')

        candidate_heads = [k for k in sd_keys if k.endswith('weight') and any(
            x in k for x in ('fc', 'linear', 'classifier', 'head'))]
        if candidate_heads:
            print(f'Possible classifier keys: {candidate_heads[:10]}')


def _auto_arch_candidates(num_classes: int) -> List[Tuple[str, nn.Module]]:
    return [
        ('robustbench_resnet18', RBResNet18(num_classes=num_classes)),
        ('torchvision_resnet18', tv_models.resnet18(num_classes=num_classes)),
        ('torchvision_resnet50', tv_models.resnet50(num_classes=num_classes)),
    ]


def _try_auto_load_arch(state_dict: Dict[str, torch.Tensor], dataset: str) -> nn.Module:
    num_classes = _infer_num_classes(state_dict, dataset)
    errors = []

    for name, model in _auto_arch_candidates(num_classes):
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f'Auto-architecture matched: {name} (num_classes={num_classes})')
            return model.eval()
        except Exception as exc:
            errors.append(f'{name}: {exc}')

    error_msg = '\n'.join(errors[:3])
    raise ValueError(
        'Could not auto-match architecture from checkpoint. '
        'Try a TorchScript file with --torchscript or pass --arch explicitly.\n'
        f'First auto-match errors:\n{error_msg}'
    )


def _load_checkpoint_model(args: argparse.Namespace) -> nn.Module:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise ValueError(f'Checkpoint not found: {checkpoint_path}')

    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')

    # Some training scripts store the full model object directly.
    if isinstance(checkpoint, nn.Module):
        print('Loaded full nn.Module directly from checkpoint.')
        return checkpoint.eval()

    state_dict = None
    try:
        state_dict = _extract_state_dict(checkpoint, args.state_dict_key)
        if args.strip_prefixes:
            state_dict = _strip_common_prefixes(state_dict)
    except Exception:
        state_dict = None

    if args.inspect_checkpoint:
        _describe_checkpoint(checkpoint_path, checkpoint, state_dict)
        if args.inspect_only:
            raise SystemExit(0)

    if not args.arch:
        # First try TorchScript from the same file.
        try:
            return _load_torchscript_model(checkpoint_path)
        except Exception:
            pass

        if state_dict is None:
            raise ValueError(
                'No architecture provided and checkpoint is not loadable as TorchScript/state_dict. '
                'Use --inspect_checkpoint to print checkpoint structure.'
            )

        if args.auto_arch:
            return _try_auto_load_arch(state_dict, args.dataset)

        raise ValueError(
            'This checkpoint contains a state_dict but no architecture was provided. '
            'Pass --arch or use --auto_arch.'
        )

    arch_ctor = _import_symbol(args.arch)
    arch_kwargs = _parse_json_dict(args.arch_kwargs)
    model = arch_ctor(**arch_kwargs)
    if not isinstance(model, nn.Module):
        raise ValueError("`--arch` did not return a torch.nn.Module instance.")

    if state_dict is None:
        state_dict = _extract_state_dict(checkpoint, args.state_dict_key)
        if args.strip_prefixes:
            state_dict = _strip_common_prefixes(state_dict)

    load_result = model.load_state_dict(state_dict, strict=args.strict)
    if not args.strict:
        missing = getattr(load_result, 'missing_keys', [])
        unexpected = getattr(load_result, 'unexpected_keys', [])
        if missing:
            print(f"Missing keys ({len(missing)}): {missing[:10]}")
        if unexpected:
            print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}")

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
    args = parse_args()
    device = _resolve_device(args.device)

    if args.torchscript:
        model = _load_torchscript_model(Path(args.torchscript))
    else:
        model = _load_checkpoint_model(args)

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