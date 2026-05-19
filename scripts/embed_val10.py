import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from options import Option
from data_utils.utils import get_all_train_file, preprocess
from model.model import Model
from utils.util import load_checkpoint, setup_seed


def resolve_root_dir(args) -> str:
    if args.dataset == "sketchy_extend":
        return os.path.join(args.data_path, "Sketchy")
    if args.dataset == "tu_berlin":
        return os.path.join(args.data_path, "TUBerlin")
    if args.dataset == "Quickdraw":
        return os.path.join(args.data_path, "QuickDraw")
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def resolve_preprocess_mode(modality: str) -> str:
    if modality == "image":
        return "im"
    if modality == "sketch":
        return "sk"
    raise ValueError(f"Unsupported modality: {modality}")


def resolve_filelist_skim(modality: str) -> str:
    if modality == "image":
        return "image"
    if modality == "sketch":
        return "sketch"
    raise ValueError(f"Unsupported modality: {modality}")


def resolve_zeroshot_dir(args) -> str:
    if args.dataset == "sketchy_extend":
        if args.test_class == "test_class_sketchy25":
            return "zeroshot1"
        if args.test_class == "test_class_sketchy21":
            return "zeroshot0"
        raise ValueError(f"Unsupported test_class: {args.test_class}")
    if args.dataset in ("tu_berlin", "Quickdraw"):
        return "zeroshot"
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def resolve_zero_filelist_path(args, modality: str) -> str:
    shot_dir = resolve_zeroshot_dir(args)
    if args.dataset == "sketchy_extend":
        if modality == "sketch":
            filename = "sketch_tx_000000000000_ready_filelist_zero.txt"
        elif modality == "image":
            filename = "all_photo_filelist_zero.txt"
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        return os.path.join(args.data_path, "Sketchy", shot_dir, filename)
    if args.dataset == "tu_berlin":
        if modality == "sketch":
            filename = "png_ready_filelist_zero.txt"
        elif modality == "image":
            filename = "ImageResized_ready_filelist_zero.txt"
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        return os.path.join(args.data_path, "TUBerlin", shot_dir, filename)
    if args.dataset == "Quickdraw":
        if modality == "sketch":
            filename = "sketch_zero.txt"
        elif modality == "image":
            filename = "all_photo_zero.txt"
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        return os.path.join(args.data_path, "QuickDraw", shot_dir, filename)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


class ValDataset(Dataset):
    def __init__(self, root_dir: str, files: np.ndarray, labels: np.ndarray, modality: str, limit: int = 0):
        if limit and limit > 0:
            files = files[:limit]
            labels = labels[:limit]
        self.files = [os.path.join(root_dir, f) for f in files]
        self.labels = labels
        self.modality = modality

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        file_path = self.files[index]
        image = preprocess(file_path, resolve_preprocess_mode(self.modality))
        label = int(self.labels[index])
        return image, label, file_path


def load_filelist_from_path(filelist_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(filelist_path):
        raise FileNotFoundError(filelist_path)
    file_paths: List[str] = []
    labels: List[int] = []
    with open(filelist_path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            file_paths.append(" ".join(parts[:-1]))
            labels.append(int(parts[-1]))
    return np.array(file_paths), np.array(labels)


def load_filelist(args, modality: str) -> Tuple[np.ndarray, np.ndarray]:
    if args.file_split == "zero":
        filelist_path = resolve_zero_filelist_path(args, modality)
        return load_filelist_from_path(filelist_path)

    original_split = args.train_split
    args.train_split = args.file_split
    file_list, labels, _ = get_all_train_file(args, resolve_filelist_skim(modality))
    args.train_split = original_split
    return file_list, labels


def build_model(args, device: torch.device) -> torch.nn.Module:
    model = Model(args)
    if args.fp16:
        model = model.half()
    if args.load is not None:
        checkpoint = load_checkpoint(args.load)
        cur = model.state_dict()
        new = {k: v for k, v in checkpoint["model"].items() if k in cur.keys()}
        cur.update(new)
        model.load_state_dict(cur)
    model = model.to(device)
    model.eval()
    return model


def extract_embeddings(args, dataloader: DataLoader, model: torch.nn.Module, device: torch.device, modality: str):
    all_embeddings: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_paths: List[str] = []

    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc=f"Embedding-{modality}", ncols=100):
            images = images.to(device)
            if args.fp16:
                images = images.half()

            sa_features, _ = model(images, None, "test", only_sa=True)
            if args.pool == "cls":
                batch_embeddings = sa_features[:, 0]
            else:
                batch_embeddings = sa_features[:, 1:].mean(dim=1)

            if args.normalize:
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

            all_embeddings.append(batch_embeddings.float().cpu().numpy())
            all_labels.append(labels.numpy())
            all_paths.extend(list(paths))

    embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, 0))
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,))
    return embeddings, labels, np.array(all_paths, dtype=object)


def save_outputs(out_path: str, embeddings: np.ndarray, labels: np.ndarray, paths: np.ndarray):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(out_path, embeddings=embeddings, labels=labels, paths=paths)
    print(f"Saved embeddings: {out_path}")
    print(f"Embeddings shape: {embeddings.shape}")


def build_output_path(base_out: str, suffix: str) -> str:
    root, ext = os.path.splitext(base_out)
    if not ext:
        return f"{base_out}_{suffix}.npz"
    return f"{root}_{suffix}{ext}"


def main():
    opt = Option()
    parser = opt.parser
    parser.add_argument(
        "--file_split",
        type=str,
        default="val10",
        choices=["train", "train90", "val10", "zero"],
        help="Which filelist split to load from train lists, or 'zero' for zero-shot filelists.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"],
                        help="Pooling for SA tokens: cls or mean of patch tokens.")
    parser.add_argument("--fp16", action="store_true", help="Use float16 for model and inputs.")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings for retrieval.")
    parser.add_argument("--modality", type=str, default="image", choices=["image", "sketch", "both"],
                        help="Extract embeddings for image, sketch, or both.")
    parser.add_argument("--out", type=str, default="./embeddings/val10_embeddings.npz")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples; 0 means all.")

    args = parser.parse_args()
    if args.file_split == "zero" and args.out == "./embeddings/val10_embeddings.npz":
        args.out = "./embeddings/zero_embeddings.npz"

    print("embed args:", str(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.choose_cuda
    setup_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args, device)
    root_dir = resolve_root_dir(args)

    if args.modality == "both":
        modalities = ["sketch", "image"]
    else:
        modalities = [args.modality]

    for modality in modalities:
        file_list, labels = load_filelist(args, modality)
        dataset = ValDataset(root_dir, file_list, labels, modality, limit=args.limit)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

        out_path = args.out
        if args.modality == "both":
            out_path = build_output_path(args.out, modality)

        embeddings, labels, paths = extract_embeddings(args, dataloader, model, device, modality)
        save_outputs(out_path, embeddings, labels, paths)


if __name__ == "__main__":
    main()
