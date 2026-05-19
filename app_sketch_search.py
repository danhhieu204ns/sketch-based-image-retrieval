import base64
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "datasets")
SEARCH_MODES = ("flat", "ivf", "hnsw", "pq", "ivf-pq")
TOP_K_DEFAULT = 10
DEFAULT_DATASET = "sketchy_extend"
IVF_MAX_NLIST = 256
IVF_NPROBE = 8
HNSW_M = 32
HNSW_EF_SEARCH = 64
PQ_TARGET_M = 16
PQ_NBITS = 8
VAL10_SKETCH_FILELIST = os.path.join(
    "Sketchy", "zeroshot1", "sketch_tx_000000000000_ready_filelist_val10.txt"
)
VAL10_IMAGE_FILELIST = os.path.join("Sketchy", "zeroshot1", "all_photo_filelist_val10.txt")
ZERO_SKETCH_FILELIST = os.path.join(
    "Sketchy", "zeroshot0", "sketch_tx_000000000000_ready_filelist_zero.txt"
)
ZERO_IMAGE_FILELIST = os.path.join("Sketchy", "zeroshot0", "all_photo_filelist_zero.txt")
ZEROSHOT1_SKETCH_FILELIST = os.path.join(
    "Sketchy", "zeroshot1", "sketch_tx_000000000000_ready_filelist_zero.txt"
)
ZEROSHOT1_IMAGE_FILELIST = os.path.join("Sketchy", "zeroshot1", "all_photo_filelist_zero.txt")


app = FastAPI(title="Sketch Image Search")


class SearchRequest(BaseModel):
    split: str
    class_name: str = Field(..., min_length=1)
    methods: List[str] = Field(default_factory=lambda: list(SEARCH_MODES))
    top_k: int = Field(default=TOP_K_DEFAULT, ge=1, le=50)
    normalize: bool = True
    random_seed: int | None = None


@dataclass(frozen=True)
class SplitContext:
    split: str
    sketch_embeddings: np.ndarray
    sketch_paths: np.ndarray
    image_embeddings: np.ndarray
    image_paths: np.ndarray
    class_names: List[str]
    class_counts: Counter
    warnings: Tuple[str, ...]


def resolve_root_dir(data_path: str, dataset: str) -> str:
    if dataset == "sketchy_extend":
        return os.path.join(data_path, "Sketchy")
    if dataset == "tu_berlin":
        return os.path.join(data_path, "TUBerlin")
    if dataset == "Quickdraw":
        return os.path.join(data_path, "QuickDraw")
    raise ValueError(f"Unsupported dataset: {dataset}")


def resolve_filelist_path(modality: str, split: str) -> str:
    if split == "val10":
        if modality == "sketch":
            return VAL10_SKETCH_FILELIST
        if modality == "image":
            return VAL10_IMAGE_FILELIST
    if split in ("zero", "zero-shot", "zeroshot0"):
        if modality == "sketch":
            return ZERO_SKETCH_FILELIST
        if modality == "image":
            return ZERO_IMAGE_FILELIST
    if split == "zeroshot1":
        if modality == "sketch":
            return ZEROSHOT1_SKETCH_FILELIST
        if modality == "image":
            return ZEROSHOT1_IMAGE_FILELIST
    raise ValueError(f"Unsupported filelist: modality={modality}, split={split}")


@lru_cache(maxsize=16)
def load_file_list(data_path: str, modality: str, split: str):
    filelist_rel = resolve_filelist_path(modality, split)
    filelist_path = os.path.join(data_path, filelist_rel)

    if not os.path.isfile(filelist_path):
        raise FileNotFoundError(filelist_path)

    root_dir = resolve_root_dir(data_path, DEFAULT_DATASET)
    file_paths: List[str] = []
    labels: List[int] = []

    with open(filelist_path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            file_paths.append(os.path.join(root_dir, " ".join(parts[:-1])))
            labels.append(int(parts[-1]))

    return np.array(file_paths), np.array(labels)


def default_embedding_paths(split: str) -> Tuple[str, str]:
    if split == "zeroshot0":
        for base in (
            os.path.join(BASE_DIR, "embeddings", "zeroshot0_embeddings"),
            os.path.join(BASE_DIR, "embeddings", "zero0_embeddings"),
        ):
            if os.path.isfile(f"{base}_sketch.npz") and os.path.isfile(f"{base}_image.npz"):
                return f"{base}_sketch.npz", f"{base}_image.npz"
        base = os.path.join(BASE_DIR, "embeddings", "zeroshot0_embeddings")
    elif split in ("zero", "zero-shot", "zeroshot1"):
        base = os.path.join(BASE_DIR, "embeddings", "zero_embeddings")
    else:
        base = os.path.join(BASE_DIR, "embeddings", "val10_embeddings")
    return f"{base}_sketch.npz", f"{base}_image.npz"


def available_split_options() -> List[str]:
    options: List[str] = []
    for split in ("val10", "zeroshot1", "zeroshot0"):
        sketch_path, image_path = default_embedding_paths(split)
        if os.path.isfile(sketch_path) and os.path.isfile(image_path):
            options.append(split)
    return options


def safe_relpath(path: str, prefix: str) -> str:
    try:
        if prefix and os.path.commonpath([path, prefix]) == prefix:
            rel = os.path.relpath(path, prefix)
            return rel.replace("\\", "/")
    except ValueError:
        pass
    return path.replace("\\", "/")


@lru_cache(maxsize=16)
def load_embeddings(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    if "embeddings" not in data or "paths" not in data:
        raise ValueError(f"Missing embeddings or paths in {npz_path}")
    embeddings = data["embeddings"]
    paths = data["paths"]
    return embeddings, paths


def ensure_str_paths(paths: np.ndarray) -> List[str]:
    output: List[str] = []
    for item in paths:
        if isinstance(item, (bytes, np.bytes_)):
            output.append(item.decode("utf-8"))
        else:
            output.append(str(item))
    return output


def normalize_path_key(path: str, root_dir: str) -> str:
    abs_path = os.path.abspath(path)
    abs_root = os.path.abspath(root_dir)
    return safe_relpath(abs_path, abs_root)


def align_embeddings_to_paths(
    target_paths: np.ndarray,
    embed_paths: List[str],
    embed_vectors: np.ndarray,
    root_dir: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    target_keys = [normalize_path_key(str(p), root_dir) for p in target_paths]
    embed_keys = [normalize_path_key(p, root_dir) for p in embed_paths]

    if len(target_keys) == len(embed_keys) and all(
        t == e for t, e in zip(target_keys, embed_keys)
    ):
        return embed_vectors, target_paths, 0

    index_by_key = {key: idx for idx, key in enumerate(embed_keys)}
    aligned_vectors: List[np.ndarray] = []
    aligned_paths: List[str] = []
    missing = 0

    for path, key in zip(target_paths, target_keys):
        idx = index_by_key.get(key)
        if idx is None:
            missing += 1
            continue
        aligned_vectors.append(embed_vectors[idx])
        aligned_paths.append(str(path))

    if not aligned_vectors:
        return np.empty((0, 0)), np.array([], dtype=object), missing

    return np.vstack(aligned_vectors), np.array(aligned_paths, dtype=object), missing


def class_name_from_path(path: str) -> str:
    normalized = os.path.normpath(str(path).replace("\\", os.sep).replace("/", os.sep))
    parts = normalized.split(os.sep)
    if len(parts) < 2:
        return ""
    return parts[-2]


def load_class_file(class_file: str) -> List[str]:
    class_names: List[str] = []
    if not os.path.isfile(class_file):
        return class_names

    with open(class_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_names.append(line.rsplit(" ", 1)[0])
    return class_names


def class_counts_from_paths(paths: np.ndarray) -> Counter:
    return Counter(
        class_name
        for class_name in (class_name_from_path(path) for path in paths)
        if class_name
    )


def available_class_names(preferred_names: List[str], paths: np.ndarray) -> Tuple[List[str], Counter]:
    counts = class_counts_from_paths(paths)
    if not counts:
        return [], counts

    preferred_available = [name for name in preferred_names if counts.get(name, 0) > 0]
    preferred_set = set(preferred_available)
    extra_available = sorted(name for name in counts if name not in preferred_set)
    return preferred_available + extra_available, counts


def should_use_aligned_paths(
    filelist_paths: np.ndarray, aligned_paths: np.ndarray, missing: int
) -> bool:
    return len(filelist_paths) > 0 and len(aligned_paths) > 0 and missing == 0


def as_faiss_matrix(vectors: np.ndarray, normalize: bool) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    matrix = np.ascontiguousarray(matrix)

    if normalize:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    return matrix


def safe_ivf_nlist(vector_count: int) -> int:
    target = int(math.sqrt(vector_count)) or 1
    recommended_training_limit = max(1, vector_count // 39)
    return max(1, min(IVF_MAX_NLIST, target, recommended_training_limit, vector_count))


def safe_pq_m(dimension: int, vector_count: int) -> int:
    target = min(PQ_TARGET_M, dimension, max(1, vector_count // 256))
    for candidate in range(target, 0, -1):
        if dimension % candidate == 0:
            return candidate
    return 1


def safe_pq_nbits(vector_count: int) -> int:
    if vector_count < 8:
        raise ValueError("PQ modes require at least 8 image embeddings.")
    recommended_centroids = max(8, vector_count // 39)
    return max(3, min(PQ_NBITS, int(math.floor(math.log2(recommended_centroids)))))


def build_faiss_index(image_embeddings: np.ndarray, method: str, normalize: bool):
    if method not in SEARCH_MODES:
        raise ValueError(f"Unsupported FAISS mode: {method}")

    vectors = as_faiss_matrix(image_embeddings, normalize)
    vector_count, dimension = vectors.shape
    if vector_count == 0 or dimension == 0:
        raise ValueError("No image embeddings available for FAISS search.")

    if method == "flat":
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        return index

    if method == "ivf":
        nlist = safe_ivf_nlist(vector_count)
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = min(IVF_NPROBE, nlist)
        index.train(vectors)
        index.add(vectors)
        return index

    if method == "hnsw":
        try:
            index = faiss.IndexHNSWFlat(dimension, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        except TypeError:
            index = faiss.IndexHNSWFlat(dimension, HNSW_M)
            index.metric_type = faiss.METRIC_INNER_PRODUCT
        index.hnsw.efConstruction = max(40, HNSW_EF_SEARCH)
        index.hnsw.efSearch = HNSW_EF_SEARCH
        index.add(vectors)
        return index

    if method == "pq":
        m = safe_pq_m(dimension, vector_count)
        nbits = safe_pq_nbits(vector_count)
        index = faiss.IndexPQ(dimension, m, nbits, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
        index.add(vectors)
        return index

    nlist = safe_ivf_nlist(vector_count)
    m = safe_pq_m(dimension, vector_count)
    nbits = safe_pq_nbits(vector_count)
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        nlist,
        m,
        nbits,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.nprobe = min(IVF_NPROBE, nlist)
    index.train(vectors)
    index.add(vectors)
    return index


@lru_cache(maxsize=32)
def build_cached_index(split: str, method: str, normalize: bool):
    context = get_split_context(split)
    return build_faiss_index(context.image_embeddings, method, normalize)


def search_faiss(
    split: str,
    query_embedding: np.ndarray,
    method: str,
    top_k: int,
    normalize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    index = build_cached_index(split, method, normalize)
    query = as_faiss_matrix(query_embedding, normalize)
    result_count = min(int(top_k), index.ntotal)
    scores, indices = index.search(query, result_count)
    valid = indices[0] >= 0
    return indices[0][valid], scores[0][valid]


def class_file_for_split(split: str) -> str:
    if split == "val10":
        return os.path.join(DATA_PATH, "Sketchy", "zeroshot1", "cname_cid.txt")
    if split == "zeroshot0":
        return os.path.join(DATA_PATH, "Sketchy", "zeroshot0", "cname_cid_zero.txt")
    return os.path.join(DATA_PATH, "Sketchy", "zeroshot1", "cname_cid_zero.txt")


@lru_cache(maxsize=8)
def get_split_context(split: str) -> SplitContext:
    if split not in available_split_options():
        raise ValueError(f"Split is not available: {split}")

    sketch_embed_path, image_embed_path = default_embedding_paths(split)
    preferred_class_names = load_class_file(class_file_for_split(split))
    sketch_paths, _ = load_file_list(DATA_PATH, "sketch", split)
    image_paths, _ = load_file_list(DATA_PATH, "image", split)
    sketch_embeddings, sketch_embed_paths = load_embeddings(sketch_embed_path)
    image_embeddings, image_embed_paths = load_embeddings(image_embed_path)

    sketch_embed_paths = ensure_str_paths(sketch_embed_paths)
    image_embed_paths = ensure_str_paths(image_embed_paths)

    root_dir = resolve_root_dir(DATA_PATH, DEFAULT_DATASET)
    aligned_sketch_embeddings, aligned_sketch_paths, sketch_missing = align_embeddings_to_paths(
        sketch_paths,
        sketch_embed_paths,
        sketch_embeddings,
        root_dir,
    )
    aligned_image_embeddings, aligned_image_paths, image_missing = align_embeddings_to_paths(
        image_paths,
        image_embed_paths,
        image_embeddings,
        root_dir,
    )

    warnings: List[str] = []
    if should_use_aligned_paths(sketch_paths, aligned_sketch_paths, sketch_missing):
        sketch_embeddings = aligned_sketch_embeddings
        sketch_paths = aligned_sketch_paths
    else:
        warnings.append(
            f"Sketch file list does not match embeddings ({sketch_missing}/{len(sketch_paths)} missing). "
            "Using paths stored in the embedding file."
        )
        sketch_paths = np.array(sketch_embed_paths, dtype=object)

    if should_use_aligned_paths(image_paths, aligned_image_paths, image_missing):
        image_embeddings = aligned_image_embeddings
        image_paths = aligned_image_paths
    else:
        warnings.append(
            f"Image file list does not match embeddings ({image_missing}/{len(image_paths)} missing). "
            "Using paths stored in the embedding file."
        )
        image_paths = np.array(image_embed_paths, dtype=object)

    class_names, class_counts = available_class_names(preferred_class_names, sketch_paths)
    if not class_names:
        raise ValueError(f"No sketch class found for split: {split}")

    missing_preferred = [
        name for name in preferred_class_names if class_counts.get(name, 0) == 0
    ]
    if missing_preferred:
        warnings.append(
            f"{len(missing_preferred)} classes from class list have no sketch paths in current embeddings."
        )

    return SplitContext(
        split=split,
        sketch_embeddings=sketch_embeddings,
        sketch_paths=sketch_paths,
        image_embeddings=image_embeddings,
        image_paths=image_paths,
        class_names=class_names,
        class_counts=class_counts,
        warnings=tuple(warnings),
    )


def encode_path(path: str) -> str:
    abs_path = os.path.abspath(path)
    raw = base64.urlsafe_b64encode(abs_path.encode("utf-8")).decode("ascii")
    return raw.rstrip("=")


def decode_path(token: str) -> str:
    padded = token + "=" * (-len(token) % 4)
    try:
        path = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image path token.") from exc

    abs_path = os.path.abspath(path)
    allowed_root = os.path.abspath(DATA_PATH)
    try:
        if os.path.commonpath([abs_path, allowed_root]) != allowed_root:
            raise HTTPException(status_code=403, detail="Image path is outside dataset directory.")
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Image path is outside dataset directory.") from exc
    return abs_path


def image_url(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    return f"/image?path={encode_path(path)}"


def public_path(path: str) -> str:
    return safe_relpath(os.path.abspath(path), DATA_PATH)


def select_random_sketch_index(context: SplitContext, class_name: str, seed: int | None) -> int | None:
    class_indices = [
        i
        for i, path_item in enumerate(context.sketch_paths)
        if class_name_from_path(path_item) == class_name
    ]
    if not class_indices:
        return None
    rng = random.Random(seed) if seed is not None else random
    return rng.choice(class_indices)


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(INDEX_HTML)


@app.get("/api/config")
def config():
    splits = available_split_options()
    return {
        "splits": splits,
        "default_split": splits[0] if splits else None,
        "methods": list(SEARCH_MODES),
        "default_top_k": TOP_K_DEFAULT,
    }


@app.get("/api/classes")
def classes(split: str = Query(...)):
    try:
        context = get_split_context(split)
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "split": split,
        "classes": [
            {"name": name, "count": int(context.class_counts.get(name, 0))}
            for name in context.class_names
        ],
        "warnings": list(context.warnings),
    }


@app.post("/api/search")
def search(request: SearchRequest):
    unknown_methods = [method for method in request.methods if method not in SEARCH_MODES]
    if unknown_methods:
        raise HTTPException(status_code=400, detail=f"Unsupported search methods: {unknown_methods}")

    try:
        context = get_split_context(request.split)
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    sketch_idx = select_random_sketch_index(context, request.class_name, request.random_seed)
    if sketch_idx is None:
        raise HTTPException(status_code=404, detail=f"No sketch found for class: {request.class_name}")

    query_embedding = context.sketch_embeddings[sketch_idx]
    query_path = str(context.sketch_paths[sketch_idx])
    method_results = []

    for method in request.methods:
        started = time.perf_counter()
        try:
            top_idx, top_scores = search_faiss(
                request.split,
                query_embedding,
                method,
                request.top_k,
                request.normalize,
            )
        except (RuntimeError, ValueError) as exc:
            method_results.append({"method": method, "error": str(exc), "results": []})
            continue

        elapsed_ms = (time.perf_counter() - started) * 1000
        results = []
        for rank, (idx, score) in enumerate(zip(top_idx, top_scores), start=1):
            idx = int(idx)
            path = str(context.image_paths[idx])
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "path": public_path(path),
                    "image_url": image_url(path),
                    "class_name": class_name_from_path(path),
                }
            )

        method_results.append(
            {
                "method": method,
                "elapsed_ms": elapsed_ms,
                "results": results,
            }
        )

    return {
        "split": request.split,
        "class_name": request.class_name,
        "query": {
            "path": public_path(query_path),
            "image_url": image_url(query_path),
            "sketch_index": int(sketch_idx),
        },
        "warnings": list(context.warnings),
        "methods": method_results,
    }


@app.get("/image")
def image(path: str = Query(...)):
    abs_path = decode_path(path)
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(abs_path)


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sketch Image Search</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f2eb;
      --panel: #ffffff;
      --ink: #202124;
      --muted: #62656a;
      --line: #d8d2c5;
      --accent: #1f7a5a;
      --accent-ink: #ffffff;
      --warn: #855d00;
      --warn-bg: #fff5d7;
      --error: #9f1d20;
      --error-bg: #ffe4e4;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(900px 420px at 5% 0%, #fff2cf 0%, transparent 55%),
        linear-gradient(180deg, #f8f6f1 0%, var(--bg) 100%);
    }

    main {
      width: min(1440px, calc(100% - 40px));
      margin: 0 auto;
      padding: 24px 0 44px;
    }

    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 46px);
      line-height: 1.02;
      letter-spacing: 0;
      font-weight: 720;
    }

    .status {
      min-height: 22px;
      color: var(--muted);
      font-size: 14px;
      text-align: right;
    }

    .layout {
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 18px;
      align-items: start;
    }

    .panel,
    .method,
    .query {
      background: rgba(255, 255, 255, 0.86);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }

    .panel {
      position: sticky;
      top: 16px;
      padding: 16px;
    }

    label {
      display: block;
      margin: 14px 0 6px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }

    select,
    input[type="number"] {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }

    .checks {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }

    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 9px;
      background: #fff;
      font-size: 13px;
      text-transform: uppercase;
    }

    .inline {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 14px;
      font-size: 14px;
      color: var(--muted);
    }

    button {
      width: 100%;
      min-height: 42px;
      margin-top: 16px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: var(--accent-ink);
      font: inherit;
      font-weight: 720;
      cursor: pointer;
    }

    button:disabled {
      opacity: 0.62;
      cursor: wait;
    }

    .warnings,
    .error {
      display: none;
      margin-top: 12px;
      border-radius: 6px;
      padding: 10px 12px;
      font-size: 13px;
      line-height: 1.4;
    }

    .warnings {
      color: var(--warn);
      background: var(--warn-bg);
      border: 1px solid #efd489;
    }

    .error {
      color: var(--error);
      background: var(--error-bg);
      border: 1px solid #f0aaa9;
    }

    .query {
      display: grid;
      grid-template-columns: 240px 1fr;
      gap: 16px;
      padding: 16px;
      margin-bottom: 18px;
      align-items: center;
    }

    .query img {
      width: 100%;
      max-height: 220px;
      object-fit: contain;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 6px;
    }

    .query h2,
    .method h2 {
      margin: 0 0 6px;
      font-size: 18px;
      letter-spacing: 0;
    }

    .meta {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }

    .method {
      padding: 16px;
      margin-bottom: 18px;
    }

    .method-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 12px;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(145px, 1fr));
      gap: 12px;
    }

    .result {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
      min-width: 0;
    }

    .thumb {
      aspect-ratio: 1 / 1;
      display: grid;
      place-items: center;
      background: #fafafa;
      border-bottom: 1px solid var(--line);
    }

    .thumb img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }

    .caption {
      padding: 8px;
      font-size: 12px;
      line-height: 1.35;
    }

    .caption strong {
      display: block;
      margin-bottom: 2px;
      font-size: 13px;
    }

    @media (max-width: 860px) {
      main { width: min(100% - 24px, 720px); }
      header { display: block; }
      .status { text-align: left; margin-top: 8px; }
      .layout { grid-template-columns: 1fr; }
      .panel { position: static; }
      .query { grid-template-columns: 1fr; }
      .query img { max-height: 260px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Sketch Image Search</h1>
      <div id="status" class="status">Loading app...</div>
    </header>
    <div class="layout">
      <aside class="panel">
        <label for="split">Data split</label>
        <select id="split"></select>

        <label for="className">Class</label>
        <select id="className"></select>

        <label>Search methods</label>
        <div id="methods" class="checks"></div>

        <label for="topK">Top K</label>
        <input id="topK" type="number" min="1" max="50" value="10" />

        <label class="inline">
          <input id="normalize" type="checkbox" checked />
          Normalize embeddings
        </label>

        <button id="searchBtn" type="button">Search</button>
        <div id="warnings" class="warnings"></div>
        <div id="error" class="error"></div>
      </aside>

      <section id="content">
        <div class="query">
          <div></div>
          <div>
            <h2>No query yet</h2>
            <p class="meta">Choose a split and class, then run search.</p>
          </div>
        </div>
      </section>
    </div>
  </main>
  <script>
    const state = { methods: [] };
    const splitEl = document.getElementById("split");
    const classEl = document.getElementById("className");
    const methodsEl = document.getElementById("methods");
    const topKEl = document.getElementById("topK");
    const normalizeEl = document.getElementById("normalize");
    const buttonEl = document.getElementById("searchBtn");
    const statusEl = document.getElementById("status");
    const warningsEl = document.getElementById("warnings");
    const errorEl = document.getElementById("error");
    const contentEl = document.getElementById("content");

    function setStatus(text) {
      statusEl.textContent = text || "";
    }

    function showError(text) {
      errorEl.style.display = text ? "block" : "none";
      errorEl.textContent = text || "";
    }

    function showWarnings(items) {
      warningsEl.style.display = items && items.length ? "block" : "none";
      warningsEl.innerHTML = (items || []).map(item => `<div>${escapeHtml(item)}</div>`).join("");
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }[ch]));
    }

    async function fetchJson(url, options) {
      const response = await fetch(url, options);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || `Request failed with ${response.status}`);
      }
      return data;
    }

    async function loadConfig() {
      const config = await fetchJson("/api/config");
      state.methods = config.methods || [];
      splitEl.innerHTML = (config.splits || [])
        .map(split => `<option value="${escapeHtml(split)}">${escapeHtml(split)}</option>`)
        .join("");
      topKEl.value = config.default_top_k || 10;
      methodsEl.innerHTML = state.methods.map(method => `
        <label class="check">
          <input type="checkbox" name="method" value="${escapeHtml(method)}" checked />
          ${escapeHtml(method)}
        </label>
      `).join("");

      if (!config.splits || !config.splits.length) {
        throw new Error("No embedding files found in ./embeddings.");
      }

      splitEl.value = config.default_split;
      await loadClasses();
    }

    async function loadClasses() {
      showError("");
      setStatus("Loading classes...");
      const data = await fetchJson(`/api/classes?split=${encodeURIComponent(splitEl.value)}`);
      classEl.innerHTML = data.classes.map(item => `
        <option value="${escapeHtml(item.name)}">${escapeHtml(item.name)} (${item.count})</option>
      `).join("");
      showWarnings(data.warnings);
      setStatus(`${data.classes.length} classes`);
    }

    function selectedMethods() {
      return Array.from(document.querySelectorAll('input[name="method"]:checked'))
        .map(input => input.value);
    }

    async function runSearch() {
      showError("");
      const methods = selectedMethods();
      if (!methods.length) {
        showError("Select at least one search method.");
        return;
      }

      buttonEl.disabled = true;
      setStatus("Searching...");
      try {
        const data = await fetchJson("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            split: splitEl.value,
            class_name: classEl.value,
            methods,
            top_k: Number(topKEl.value),
            normalize: normalizeEl.checked
          })
        });
        showWarnings(data.warnings);
        renderResults(data);
        setStatus("Done");
      } catch (error) {
        showError(error.message);
        setStatus("Search failed");
      } finally {
        buttonEl.disabled = false;
      }
    }

    function renderResults(data) {
      const queryImage = data.query.image_url
        ? `<img src="${escapeHtml(data.query.image_url)}" alt="Query sketch" />`
        : `<p class="meta">${escapeHtml(data.query.path)}</p>`;
      const methods = data.methods.map(renderMethod).join("");
      contentEl.innerHTML = `
        <div class="query">
          <div>${queryImage}</div>
          <div>
            <h2>Query sketch</h2>
            <p class="meta">${escapeHtml(data.class_name)} · ${escapeHtml(data.split)}</p>
            <p class="meta">${escapeHtml(data.query.path)}</p>
          </div>
        </div>
        ${methods}
      `;
    }

    function renderMethod(method) {
      if (method.error) {
        return `
          <article class="method">
            <div class="method-head">
              <h2>${escapeHtml(method.method.toUpperCase())}</h2>
            </div>
            <div class="error" style="display:block">${escapeHtml(method.error)}</div>
          </article>
        `;
      }

      const cards = method.results.map(item => {
        const image = item.image_url
          ? `<img src="${escapeHtml(item.image_url)}" alt="${escapeHtml(item.path)}" loading="lazy" />`
          : `<span class="meta">${escapeHtml(item.path)}</span>`;
        return `
          <div class="result">
            <div class="thumb">${image}</div>
            <div class="caption">
              <strong>#${item.rank} · ${Number(item.score).toFixed(4)}</strong>
              <div>${escapeHtml(item.class_name || "")}</div>
              <div class="meta">${escapeHtml(item.path)}</div>
            </div>
          </div>
        `;
      }).join("");

      return `
        <article class="method">
          <div class="method-head">
            <h2>${escapeHtml(method.method.toUpperCase())}</h2>
            <p class="meta">${Number(method.elapsed_ms).toFixed(2)} ms</p>
          </div>
          <div class="grid">${cards}</div>
        </article>
      `;
    }

    splitEl.addEventListener("change", () => loadClasses().catch(error => showError(error.message)));
    buttonEl.addEventListener("click", runSearch);

    loadConfig().catch(error => {
      showError(error.message);
      setStatus("Load failed");
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run("app_sketch_search:app", host="127.0.0.1", port=8000, reload=True)
