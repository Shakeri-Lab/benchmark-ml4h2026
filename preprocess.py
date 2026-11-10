#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from preprocessing.registry import get_dataset_names, get_preprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess raw datasets in datasets_raw. "
            "Select a single dataset or preprocess all registered datasets."
        )
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=sorted(get_dataset_names()) + ["all"],
        required=True,
        help="Dataset to preprocess or 'all' to run every available dataset.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets_raw",
        help="Directory containing raw dataset folders (default: ./datasets_raw).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets_processed",
        help="Directory where processed datasets will be written (default: ./datasets_processed).",
    )
    return parser.parse_args()


def run_dataset(dataset: str, raw_root: Path, output_root: Path) -> None:
    preprocessor = get_preprocessor(dataset)
    raw_dir = raw_root / dataset
    output_dir = output_root / dataset
    print(f"[+] Preprocessing {dataset} from {raw_dir} -> {output_dir}")
    preprocessor(raw_dir, output_dir)


def run_all(datasets: Iterable[str], raw_root: Path, output_root: Path) -> None:
    for dataset in datasets:
        run_dataset(dataset, raw_root, output_root)


def main() -> None:
    args = parse_args()
    datasets = get_dataset_names() if args.dataset == "all" else [args.dataset]
    run_all(datasets, args.raw_root, args.output_root)


if __name__ == "__main__":
    main()
