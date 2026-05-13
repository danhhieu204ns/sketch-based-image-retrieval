import argparse
from collections import Counter
from pathlib import Path


def read_class_names(path: Path):
    names = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            # Class name can contain spaces; last token is the id.
            names.append(" ".join(parts[:-1]))
    return names


def read_labels(file_path: Path):
    labels = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            labels.append(int(parts[-1]))
    return labels


def summarize_counts(counter: Counter, class_names, title: str, top_n: int):
    total = sum(counter.values())
    num_classes = len(class_names)
    avg = total / num_classes if num_classes else 0.0
    min_count = min(counter.values()) if counter else 0
    max_count = max(counter.values()) if counter else 0

    print(f"{title} samples: {total}")
    print(f"  classes: {num_classes}")
    print(f"  avg per class: {avg:.2f}")
    print(f"  min/max per class: {min_count}/{max_count}")

    if top_n > 0 and counter:
        print(f"  top {top_n} classes by count:")
        for cls_id, count in counter.most_common(top_n):
            name = class_names[cls_id] if cls_id < len(class_names) else f"id_{cls_id}"
            print(f"    {count:>6}  {name}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Sketchy train/test classes")
    parser.add_argument("--data_path", type=str, default="./datasets")
    parser.add_argument("--split", type=str, default="zeroshot1", choices=["zeroshot0", "zeroshot1"])
    parser.add_argument("--show", type=int, default=10, help="show first N class names")
    parser.add_argument("--count_files", action="store_true", help="summarize sample counts from filelists")
    parser.add_argument("--top", type=int, default=10, help="top N classes by count when --count_files")
    args = parser.parse_args()

    base = Path(args.data_path) / "Sketchy" / args.split
    train_cls_path = base / "cname_cid.txt"
    test_cls_path = base / "cname_cid_zero.txt"

    if not train_cls_path.exists() or not test_cls_path.exists():
        raise FileNotFoundError(f"Missing class files under: {base}")

    train_classes = read_class_names(train_cls_path)
    test_classes = read_class_names(test_cls_path)

    train_set = set(train_classes)
    test_set = set(test_classes)
    overlap = sorted(train_set & test_set)
    only_train = sorted(train_set - test_set)
    only_test = sorted(test_set - train_set)

    print(f"Split: {args.split}")
    print(f"Train classes: {len(train_classes)}")
    print(f"Test classes:  {len(test_classes)}")
    print(f"Overlap:       {len(overlap)}")
    print(f"Only train:    {len(only_train)}")
    print(f"Only test:     {len(only_test)}")

    if args.show > 0:
        print("\nTrain class names (first N):")
        for name in train_classes[: args.show]:
            print(f"  {name}")
        print("\nTest class names (first N):")
        for name in test_classes[: args.show]:
            print(f"  {name}")

    if args.count_files:
        # File lists
        train_sk = base / "sketch_tx_000000000000_ready_filelist_train.txt"
        test_sk = base / "sketch_tx_000000000000_ready_filelist_zero.txt"
        train_im = base / "all_photo_filelist_train.txt"
        test_im = base / "all_photo_filelist_zero.txt"

        for fp in [train_sk, test_sk, train_im, test_im]:
            if not fp.exists():
                raise FileNotFoundError(f"Missing filelist: {fp}")

        train_sk_labels = read_labels(train_sk)
        test_sk_labels = read_labels(test_sk)
        train_im_labels = read_labels(train_im)
        test_im_labels = read_labels(test_im)

        print("\nSample counts by split:")
        summarize_counts(Counter(train_sk_labels), train_classes, "Train sketches", args.top)
        summarize_counts(Counter(train_im_labels), train_classes, "Train images", args.top)
        summarize_counts(Counter(test_sk_labels), test_classes, "Test sketches", args.top)
        summarize_counts(Counter(test_im_labels), test_classes, "Test images", args.top)


if __name__ == "__main__":
    main()
