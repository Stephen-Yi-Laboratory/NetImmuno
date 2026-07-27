"""Command-line inference examples for PanimmuneNet checkpoints.

Examples
--------
python example_usage.py pmhc --checkpoint PATH --mhc SEQUENCE --peptide SEQUENCE
python example_usage.py tcr --checkpoint PATH --mhc SEQUENCE --peptide SEQUENCE \
    --tcra SEQUENCE --tcrb SEQUENCE
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type, TypeVar

import torch

from MHCpeptideEmbeddingClassifier import MHCpeptideRegressor
from TCRmhcEmbeddingClassifier import TCRpMHCClassifier
from src.model_config import (
    FullGridPairConfig,
    PMHCPairConfig,
    TCRClassifierConfig,
    TCRPairConfig,
    pMHCClassifierConfig,
)


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
ConfigT = TypeVar("ConfigT")


def clean_sequence(value: str, name: str) -> str:
    """Uppercase a sequence and retain the 20 canonical amino acids."""
    cleaned = "".join(char for char in value.strip().upper() if char in VALID_AA)
    if not cleaned:
        raise ValueError(f"{name} contains no canonical amino-acid residues.")
    return cleaned


def fixed_length(sequence: str, length: int) -> str:
    """Right-pad with X or truncate to a fixed length."""
    return (sequence + ("X" * length))[:length]


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    return device


def load_checkpoint(path: Path, device: torch.device) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def config_from_metadata(
    config_type: Type[ConfigT],
    metadata: Optional[Mapping[str, Any]],
) -> ConfigT:
    """Build a dataclass config while ignoring unrelated saved fields."""
    if not isinstance(metadata, Mapping):
        return config_type()
    allowed = {field.name for field in fields(config_type) if field.init}
    values = {key: value for key, value in metadata.items() if key in allowed}
    return config_type(**values)


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Extract weights from full, weights-only, or nested checkpoints."""
    candidate = checkpoint
    while isinstance(candidate, Mapping):
        if "model" in candidate:
            candidate = candidate["model"]
            continue
        if "state_dict" in candidate:
            candidate = candidate["state_dict"]
            continue
        break

    if not isinstance(candidate, Mapping) or not candidate:
        raise ValueError("The checkpoint does not contain a model state dictionary.")
    if not all(isinstance(value, torch.Tensor) for value in candidate.values()):
        raise ValueError("The selected checkpoint mapping is not a model state dictionary.")

    # Also accept checkpoints saved from DistributedDataParallel.
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in candidate.items()
    }
    return state


def predict_pmhc(args: argparse.Namespace, device: torch.device) -> float:
    checkpoint = load_checkpoint(args.checkpoint, device)
    metadata = checkpoint if isinstance(checkpoint, Mapping) else {}

    pair_cfg = config_from_metadata(PMHCPairConfig, metadata.get("cfg_pair"))
    classifier_cfg = config_from_metadata(
        pMHCClassifierConfig, metadata.get("cfg_classifier")
    )
    model = MHCpeptideRegressor.from_config(
        pair_cfg=pair_cfg,
        clf_cfg=classifier_cfg,
        grid_len=pair_cfg.fixed_len,
        device=str(device),
    ).to(device)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.eval()

    mhc = fixed_length(clean_sequence(args.mhc, "MHC"), pair_cfg.mhc_len)
    peptide = clean_sequence(args.peptide, "peptide")
    if len(peptide) > pair_cfg.pep_len:
        raise ValueError(
            f"Peptide length {len(peptide)} exceeds the configured maximum "
            f"of {pair_cfg.pep_len}."
        )
    peptide = fixed_length(peptide, pair_cfg.pep_len)

    with torch.inference_mode():
        score = model([f"{mhc}:{peptide}"]).reshape(-1)[0]
    return float(score.cpu())


def predict_tcr(args: argparse.Namespace, device: torch.device) -> float:
    checkpoint = load_checkpoint(args.checkpoint, device)
    metadata = checkpoint if isinstance(checkpoint, Mapping) else {}

    pmhc_cfg = config_from_metadata(PMHCPairConfig, metadata.get("cfg_pmhc"))
    tcr_cfg = config_from_metadata(TCRPairConfig, metadata.get("cfg_tcr"))
    full_cfg = config_from_metadata(FullGridPairConfig, metadata.get("cfg_full"))
    classifier_cfg = config_from_metadata(
        TCRClassifierConfig, metadata.get("cfg_classifier")
    )
    model = TCRpMHCClassifier.from_config(
        pmhc_cfg=pmhc_cfg,
        tcr_cfg=tcr_cfg,
        full_cfg=full_cfg,
        clf_cfg=classifier_cfg,
        device=str(device),
        clamp_to_label_range=True,
        apply_mask_in_embedder=True,
    ).to(device)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.eval()

    mhc = fixed_length(clean_sequence(args.mhc, "MHC"), pmhc_cfg.mhc_len)
    peptide = clean_sequence(args.peptide, "peptide")
    if len(peptide) > pmhc_cfg.pep_len:
        raise ValueError(
            f"Peptide length {len(peptide)} exceeds the configured maximum "
            f"of {pmhc_cfg.pep_len}."
        )
    peptide = fixed_length(peptide, pmhc_cfg.pep_len)
    tcra = clean_sequence(args.tcra, "TCR alpha")[: tcr_cfg.tcr_a_max_len]
    tcrb = (
        clean_sequence(args.tcrb, "TCR beta")[: tcr_cfg.tcr_b_max_len]
        if args.tcrb
        else None
    )

    with torch.inference_mode():
        output = model([peptide], [mhc], [tcra], [tcrb]).reshape(-1)[0]
        probability = (
            torch.sigmoid(output)
            if classifier_cfg.output_activation == "none"
            else output
        )
    return float(probability.cpu())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one PanimmuneNet pMHC or TCR-pMHC prediction."
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, or a device such as cuda:0.",
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    pmhc = subparsers.add_parser("pmhc", help="Predict a pMHC binding score.")
    pmhc.add_argument("--checkpoint", type=Path, required=True)
    pmhc.add_argument("--mhc", required=True, help="MHC-I pseudosequence.")
    pmhc.add_argument("--peptide", required=True, help="Peptide sequence.")

    tcr = subparsers.add_parser(
        "tcr", help="Predict a TCR-pMHC binding probability."
    )
    tcr.add_argument("--checkpoint", type=Path, required=True)
    tcr.add_argument("--mhc", required=True, help="MHC-I pseudosequence.")
    tcr.add_argument("--peptide", required=True, help="Peptide sequence.")
    tcr.add_argument("--tcra", required=True, help="TCR alpha-chain sequence.")
    tcr.add_argument(
        "--tcrb", default=None, help="Optional TCR beta-chain sequence."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = select_device(args.device)
    print(f"Device: {device}")

    if args.task == "pmhc":
        score = predict_pmhc(args, device)
        print(f"pMHC binding score: {score:.6f}")
    else:
        probability = predict_tcr(args, device)
        print(f"TCR-pMHC binding probability: {probability:.6f}")


if __name__ == "__main__":
    main()
