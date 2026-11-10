"""Stage the frozen entry points at the original experiment-relative paths.

Run from the repository root. Existing files are accepted only if identical.
No models, datasets, predictions, or other existing artifacts are overwritten.
"""
from pathlib import Path
import hashlib
import shutil


def main():
    source = Path(__file__).resolve().parent
    root = source.parent
    mapping = {
        "warning": "outputs/review_revision_20261009/warning_lead_budget/code",
        "iob": "outputs/review_revision_20261011/iob_forecaster/code",
        "simulator": "outputs/review_revision_20261007/arimax_isolated_20261007/code",
    }
    count = 0
    for name, destination in mapping.items():
        for path in sorted((source / name).glob("*.py")):
            target = root / destination / path.name
            if target.exists():
                if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(path.read_bytes()).digest():
                    raise FileExistsError(f"Existing source differs: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            count += 1
    print(f"Staged or verified {count} frozen source files.")


if __name__ == "__main__":
    main()
