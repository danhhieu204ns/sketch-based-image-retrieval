import argparse
import csv
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - user environment dependent
    raise SystemExit(
        "faiss is required for this script. Install faiss-cpu or faiss-gpu (conda)."
    ) from exc


def load_embeddings(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    if "embeddings" not in data or "labels" not in data:
        raise ValueError(f"Missing embeddings or labels in {npz_path}")
    embeddings = data["embeddings"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    return embeddings, labels


def resolve_split_embeddings(split: str, embeddings_dir: str) -> Tuple[str, str, str]:
    split_aliases = {
        "zero": "zeroshot",
        "zeroshot": "zeroshot1",
    }
    resolved_split = split_aliases.get(split, split)

    if resolved_split == "val10":
        base = "val10_embeddings"
    elif resolved_split == "zeroshot1":
        base = "zero_embeddings"
    elif resolved_split == "zeroshot0":
        base = "zeroshot0_embeddings"
    else:
        raise ValueError(f"Unsupported split: {split}")

    sketch_path = os.path.join(embeddings_dir, f"{base}_sketch.npz")
    image_path = os.path.join(embeddings_dir, f"{base}_image.npz")
    if not os.path.isfile(sketch_path) or not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Missing embedding pair for split={split}: {sketch_path}, {image_path}"
        )
    return resolved_split, sketch_path, image_path


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.size == 0:
        return np.ascontiguousarray(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def build_index(
    method: str,
    dim: int,
    nlist: int,
    pq_m: int,
    pq_bits: int,
    hnsw_m: int,
) -> faiss.Index:
    if method == "Flat":
        factory = "Flat"
    elif method == "IVF":
        factory = f"IVF{nlist},Flat"
    elif method == "HNSW":
        factory = f"HNSW{hnsw_m}"
    elif method == "PQ":
        factory = f"PQ{pq_m}x{pq_bits}"
    elif method == "IVF-PQ":
        factory = f"IVF{nlist},PQ{pq_m}x{pq_bits}"
    else:
        raise ValueError(f"Unsupported method: {method}")

    return faiss.index_factory(dim, factory, faiss.METRIC_INNER_PRODUCT)


def set_search_params(index: faiss.Index, nprobe: int, ef_search: int) -> None:
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = ef_search


def compute_metrics(
    retrieved_indices: np.ndarray,
    sk_labels: np.ndarray,
    im_labels: np.ndarray,
    k_recall: int = 100,
) -> Tuple[float, float, float]:
    valid = retrieved_indices >= 0
    safe_indices = np.where(valid, retrieved_indices, 0)
    retrieved_labels = im_labels[safe_indices]
    matches = (retrieved_labels == sk_labels[:, None]) & valid

    label_counts: Dict[int, int] = {}
    for label in im_labels:
        label_counts[int(label)] = label_counts.get(int(label), 0) + 1
    total_relevant = np.array([label_counts.get(int(l), 0) for l in sk_labels])
    total_relevant = np.maximum(total_relevant, 1)

    k_eval = matches.shape[1]
    cum_matches = np.cumsum(matches, axis=1)
    precisions = cum_matches / (np.arange(1, k_eval + 1)[None, :])
    ap = (precisions * matches).sum(axis=1) / total_relevant
    map_all = float(np.mean(ap))

    k_recall = min(k_recall, k_eval)
    recall = matches[:, :k_recall].sum(axis=1) / total_relevant
    precision = matches[:, :k_recall].mean(axis=1)

    return map_all, float(np.mean(recall)), float(np.mean(precision))


def index_memory_mb(index: faiss.Index) -> float:
    try:
        cpu_index = faiss.index_gpu_to_cpu(index)
    except Exception:
        cpu_index = index
    blob = faiss.serialize_index(cpu_index)
    return len(blob) / (1024 * 1024)


def resolve_use_gpu(use_gpu: str) -> bool:
    if use_gpu == "on":
        return True
    if use_gpu == "off":
        return False
    get_gpus = getattr(faiss, "get_num_gpus", lambda: 0)
    return get_gpus() > 0


def maybe_move_to_gpu(index: faiss.Index, method: str, use_gpu: bool, gpu_id: int) -> faiss.Index:
    if not use_gpu:
        return index
    get_gpus = getattr(faiss, "get_num_gpus", lambda: 0)
    if get_gpus() <= 0:
        return index
    if method == "HNSW":
        return index
    resources = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(resources, gpu_id, index)


def compare_embedding_pair(
    split: str,
    sketch_embeddings_path: str,
    image_embeddings_path: str,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    sk_embeddings, sk_labels = load_embeddings(sketch_embeddings_path)
    im_embeddings, im_labels = load_embeddings(image_embeddings_path)

    if not args.no_normalize:
        sk_embeddings = normalize_embeddings(sk_embeddings)
        im_embeddings = normalize_embeddings(im_embeddings)

    dim = im_embeddings.shape[1]
    k_eval = args.k_eval if args.k_eval > 0 else im_embeddings.shape[0]

    methods = [
        ("Ret-Flat", "Flat"),
        ("Ret-IVF", "IVF"),
        ("Ret-HNSW", "HNSW"),
        ("Ret-PQ", "PQ"),
        ("Ret-IVF-PQ", "IVF-PQ"),
    ]

    rows: List[Dict[str, object]] = []
    use_gpu = resolve_use_gpu(args.use_gpu)

    for method_name, method in methods:
        index = build_index(method, dim, args.nlist, args.pq_m, args.pq_bits, args.hnsw_m)
        if not index.is_trained:
            index.train(im_embeddings)
        index.add(im_embeddings)
        index = maybe_move_to_gpu(index, method, use_gpu, args.gpu_id)
        set_search_params(index, args.nprobe, args.ef_search)

        start = time.perf_counter()
        _, retrieved = index.search(sk_embeddings, k_eval)
        elapsed = time.perf_counter() - start
        latency_ms = (elapsed / sk_embeddings.shape[0]) * 1000.0

        map_all, recall_100, p_100 = compute_metrics(retrieved, sk_labels, im_labels, k_recall=100)
        memory_mb = index_memory_mb(index)

        rows.append(
            {
                "Split": split,
                "Sketches": int(sk_embeddings.shape[0]),
                "Images": int(im_embeddings.shape[0]),
                "Method": method_name,
                "Index": method,
                "Rerank": "No",
                "mAP": map_all,
                "Recall@100": recall_100,
                "P@100": p_100,
                "Latency/query_ms": latency_ms,
                "Memory_MB": memory_mb,
                "nlist": args.nlist,
                "nprobe": args.nprobe,
                "pq_m": args.pq_m,
                "pq_bits": args.pq_bits,
                "ef_search": args.ef_search,
                "hnsw_m": args.hnsw_m,
            }
        )

    return rows


def print_markdown(rows: List[Dict[str, object]]) -> None:
    print(
        "| Split | Method | Index | Rerank | mAP | Recall@100 | P@100 | "
        "Latency/query (ms) | Memory (MB) |"
    )
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['Split']} | {row['Method']} | {row['Index']} | {row['Rerank']} | "
            f"{float(row['mAP']):.4f} | {float(row['Recall@100']):.4f} | "
            f"{float(row['P@100']):.4f} | {float(row['Latency/query_ms']):.2f} | "
            f"{float(row['Memory_MB']):.2f} |"
        )


def write_outputs(rows: List[Dict[str, object]], csv_path: str, json_path: str) -> None:
    if csv_path:
        fieldnames = list(rows[0].keys()) if rows else []
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump({"summary": rows}, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval methods on embeddings")
    parser.add_argument("--sketch_embeddings", type=str)
    parser.add_argument("--image_embeddings", type=str)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Named embedding splits to compare: val10, zeroshot/zeroshot1, zeroshot0.",
    )
    parser.add_argument("--embeddings_dir", type=str, default="./embeddings")
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--out_json", type=str, default="")
    parser.add_argument(
        "--normalize",
        dest="no_normalize",
        action="store_false",
        help="L2-normalize embeddings before inner-product retrieval (default).",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable default L2 normalization before inner-product retrieval.",
    )
    parser.set_defaults(no_normalize=False)
    parser.add_argument("--k_eval", type=int, default=0, help="Top-K to evaluate. 0 means all.")
    parser.add_argument("--nlist", type=int, default=128)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--pq_m", type=int, default=16)
    parser.add_argument("--pq_bits", type=int, default=8)
    parser.add_argument("--hnsw_m", type=int, default=32)
    parser.add_argument("--ef_search", type=int, default=64)
    parser.add_argument(
        "--use_gpu",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Use GPU for supported indexes (auto/on/off).",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    all_rows: List[Dict[str, object]] = []
    if args.splits:
        for split in args.splits:
            resolved_split, sketch_path, image_path = resolve_split_embeddings(
                split, args.embeddings_dir
            )
            all_rows.extend(compare_embedding_pair(resolved_split, sketch_path, image_path, args))
    else:
        if not args.sketch_embeddings or not args.image_embeddings:
            parser.error(
                "Provide either --splits or both --sketch_embeddings and --image_embeddings."
            )
        all_rows.extend(
            compare_embedding_pair(
                "custom", args.sketch_embeddings, args.image_embeddings, args
            )
        )

    print_markdown(all_rows)
    write_outputs(all_rows, args.out_csv, args.out_json)


if __name__ == "__main__":
    main()
