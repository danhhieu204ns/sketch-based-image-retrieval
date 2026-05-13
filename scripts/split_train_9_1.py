import argparse
import random
from collections import defaultdict
from pathlib import Path


DATASET_FILES = {
    "sketchy25": [
        "Sketchy/zeroshot1/sketch_tx_000000000000_ready_filelist_train.txt",
        "Sketchy/zeroshot1/all_photo_filelist_train.txt",
    ],
    "sketchy21": [
        "Sketchy/zeroshot0/sketch_tx_000000000000_ready_filelist_train.txt",
        "Sketchy/zeroshot0/all_photo_filelist_train.txt",
    ],
    "tuberlin": [
        "TUBerlin/zeroshot/png_ready_filelist_train.txt",
        "TUBerlin/zeroshot/ImageResized_ready_filelist_train.txt",
    ],
    "quickdraw": [
        "QuickDraw/zeroshot/sketch_train.txt",
        "QuickDraw/zeroshot/all_photo_train.txt",
    ],
}


def split_output_path(path, split_name):
    if path.name.endswith("_train.txt"):
        return path.with_name(path.name.replace("_train.txt", f"_{split_name}.txt"))
    raise ValueError(f"Unsupported train filename: {path}")


def read_labeled_lines(path):
    groups = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            label = stripped.split()[-1]
            groups[label].append(stripped)
    return groups


def split_groups(groups, train_ratio, rng):
    train_lines = []
    val_lines = []

    for label in sorted(groups, key=lambda item: int(item) if item.isdigit() else item):
        lines = groups[label][:]
        rng.shuffle(lines)

        if len(lines) == 1:
            n_train = 1
        else:
            n_train = round(len(lines) * train_ratio)
            n_train = min(max(n_train, 1), len(lines) - 1)

        train_lines.extend(lines[:n_train])
        val_lines.extend(lines[n_train:])

    return train_lines, val_lines


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for line in lines:
            fh.write(f"{line}\n")


def split_file(path, train_ratio, seed, overwrite):
    train_path = split_output_path(path, "train90")
    val_path = split_output_path(path, "val10")

    if not overwrite and (train_path.exists() or val_path.exists()):
        raise FileExistsError(
            f"{train_path} or {val_path} already exists. Use --overwrite to regenerate."
        )

    groups = read_labeled_lines(path)
    rng = random.Random(seed)
    train_lines, val_lines = split_groups(groups, train_ratio, rng)

    write_lines(train_path, train_lines)
    write_lines(val_path, val_lines)

    total = len(train_lines) + len(val_lines)
    print(
        f"{path}: total={total}, train90={len(train_lines)}, val10={len(val_lines)}, "
        f"classes={len(groups)}"
    )


def resolve_files(data_path, dataset):
    selected = DATASET_FILES.keys() if dataset == "all" else [dataset]
    files = []
    for name in selected:
        for rel_path in DATASET_FILES[name]:
            path = data_path / rel_path
            if path.exists():
                files.append(path)
            else:
                print(f"skip missing: {path}")
    return files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split train filelists into stratified 90% train and 10% validation filelists."
    )
    parser.add_argument("--data_path", type=Path, default=Path("./datasets"))
    parser.add_argument(
        "--dataset",
        choices=["all", "sketchy25", "sketchy21", "tuberlin", "quickdraw"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train_ratio must be between 0 and 1")

    files = resolve_files(args.data_path, args.dataset)
    if not files:
        raise FileNotFoundError(f"No train filelists found under {args.data_path}")

    for path in files:
        split_file(path, args.train_ratio, args.seed, args.overwrite)


if __name__ == "__main__":
    main()
