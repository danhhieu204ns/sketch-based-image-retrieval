import math
import os
from collections import Counter
from typing import List, Tuple

import faiss
import numpy as np
import streamlit as st
from PIL import Image


SEARCH_MODES = ("flat", "ivf", "hnsw", "pq", "ivf-pq")
TOP_K_DEFAULT = 10
QUERY_IMAGE_WIDTH = 320
RESULT_IMAGE_WIDTH = 200
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


def setup_page():
    st.set_page_config(page_title="Sketch-Image Search", layout="wide")
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=Spectral:wght@400;600&display=swap');
html, body, [class*="css"], .stTextInput, .stSelectbox {
  font-family: 'Space Grotesk', sans-serif;
  color: #1f1a14;
}
[data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 600px at 8% 0%, #fff3d9 0%, #f8f6f1 50%, #f1f3f5 100%);
}
h1, h2, h3 {
  font-family: 'Spectral', serif;
  letter-spacing: 0.2px;
}
#MainMenu, header, footer {visibility: hidden;}
.stButton>button {border-radius: 6px;}
</style>
        """,
        unsafe_allow_html=True,
    )


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


@st.cache_data(show_spinner=False)
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
        for base in ("./embeddings/zeroshot0_embeddings", "./embeddings/zero0_embeddings"):
            if os.path.isfile(f"{base}_sketch.npz") and os.path.isfile(f"{base}_image.npz"):
                return f"{base}_sketch.npz", f"{base}_image.npz"
        base = "./embeddings/zeroshot0_embeddings"
    elif split in ("zero", "zero-shot", "zeroshot1"):
        base = "./embeddings/zero_embeddings"
    else:
        base = "./embeddings/val10_embeddings"
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


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@st.cache_data(show_spinner=False)
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


@st.cache_resource(show_spinner=False)
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


def search_faiss(
    query_embedding: np.ndarray,
    image_embeddings: np.ndarray,
    method: str,
    top_k: int,
    normalize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    index = build_faiss_index(image_embeddings, method, normalize)
    query = as_faiss_matrix(query_embedding, normalize)
    result_count = min(int(top_k), index.ntotal)
    scores, indices = index.search(query, result_count)
    valid = indices[0] >= 0
    return indices[0][valid], scores[0][valid]


def main():
    setup_page()
    st.title("Sketch-Image Search")
    st.write("Chọn chế độ, class, phương pháp search và xem kết quả.")

    data_path = "./datasets"
    dataset = DEFAULT_DATASET
    split_options = available_split_options()
    if not split_options:
        st.error("Không tìm thấy file embedding cho split nào trong ./embeddings.")
        return
    data_split = st.selectbox("Chọn split dữ liệu", split_options)
    sketch_embed_path, image_embed_path = default_embedding_paths(data_split)
    if data_split == "val10":
        class_file = os.path.join(data_path, "Sketchy/zeroshot1/cname_cid.txt")
    elif data_split == "zeroshot0":
        class_file = os.path.join(data_path, "Sketchy/zeroshot0/cname_cid_zero.txt")
    else:
        class_file = os.path.join(data_path, "Sketchy/zeroshot1/cname_cid_zero.txt")

    preferred_class_names = load_class_file(class_file)
    sk_paths, sk_labels = load_file_list(data_path, "sketch", data_split)
    im_paths, im_labels = load_file_list(data_path, "image", data_split)

    try:
        sk_embeddings, sk_embed_paths = load_embeddings(sketch_embed_path)
        im_embeddings, im_embed_paths = load_embeddings(image_embed_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Failed to load embeddings: {exc}")
        return

    sk_embed_paths = ensure_str_paths(sk_embed_paths)
    im_embed_paths = ensure_str_paths(im_embed_paths)

    root_dir = resolve_root_dir(data_path, dataset)
    aligned_sk_embeddings, aligned_sk_paths, sk_missing = align_embeddings_to_paths(
        sk_paths,
        sk_embed_paths,
        sk_embeddings,
        root_dir,
    )
    aligned_im_embeddings, aligned_im_paths, im_missing = align_embeddings_to_paths(
        im_paths,
        im_embed_paths,
        im_embeddings,
        root_dir,
    )

    if should_use_aligned_paths(sk_paths, aligned_sk_paths, sk_missing):
        sk_embeddings = aligned_sk_embeddings
        sk_paths = aligned_sk_paths
    else:
        st.warning(
            f"Sketch file list không khớp embedding ({sk_missing}/{len(sk_paths)} missing). "
            "Đang dùng paths lưu trong file embedding."
        )
        sk_paths = np.array(sk_embed_paths, dtype=object)

    if should_use_aligned_paths(im_paths, aligned_im_paths, im_missing):
        im_embeddings = aligned_im_embeddings
        im_paths = aligned_im_paths
    else:
        st.warning(
            f"Image file list không khớp embedding ({im_missing}/{len(im_paths)} missing). "
            "Đang dùng paths lưu trong file embedding."
        )
        im_paths = np.array(im_embed_paths, dtype=object)

    class_names, class_counts = available_class_names(preferred_class_names, sk_paths)
    if not class_names:
        st.error(f"Không tìm thấy sketch class trong embedding/filelist: {class_file}")
        return

    missing_preferred = [
        name for name in preferred_class_names if class_counts.get(name, 0) == 0
    ]
    if missing_preferred:
        st.warning(
            f"{len(missing_preferred)} class trong class list không có sketch path "
            "từ embedding hiện tại."
        )

    selected_class = st.selectbox("Chọn class", class_names)
    selected_methods = list(SEARCH_MODES)
    top_k = TOP_K_DEFAULT
    normalize_embeddings = True

    import random

    class_indices = []
    for i, path_item in enumerate(sk_paths):
        class_in_path = class_name_from_path(path_item)
        if class_in_path == selected_class:
            class_indices.append(i)

    if (
        "last_class" not in st.session_state
        or st.session_state.last_class != selected_class
        or st.session_state.get("last_data_split") != data_split
    ):
        st.session_state.last_class = selected_class
        st.session_state.last_data_split = data_split
        if class_indices:
            st.session_state.random_idx = random.choice(class_indices)
        else:
            st.session_state.random_idx = None

    random_idx = st.session_state.random_idx
    if random_idx is None:
        st.error(f"Không tìm thấy sketch cho class này: {selected_class}")
        return
    sketch_idx = random_idx

    run_search = st.button("Search", type="primary")

    if not run_search:
        st.info("Chọn class và nhấn Search để tìm kiếm.")
        return

    st.subheader("Query sketch")
    sketch_path_full = str(sk_paths[sketch_idx])
    if os.path.isfile(sketch_path_full):
        st.image(load_image(sketch_path_full), width=QUERY_IMAGE_WIDTH)
    else:
        st.write(sketch_path_full)

    query_embedding = sk_embeddings[sketch_idx]
    for method in selected_methods:
        import time

        st.markdown(f"### {method.upper()} results")
        start = time.time()
        try:
            top_idx, top_scores = search_faiss(
                query_embedding,
                im_embeddings,
                method,
                top_k,
                normalize_embeddings,
            )
        except (RuntimeError, ValueError) as exc:
            st.warning(f"FAISS {method} failed: {exc}")
            continue
        elapsed = time.time() - start

        if top_idx.size == 0:
            st.warning(f"No FAISS results for {method}.")
            continue

        cols = st.columns(len(top_idx))
        for rank, (idx, score) in enumerate(zip(top_idx, top_scores), start=1):
            with cols[rank - 1]:
                idx = int(idx)
                path = str(im_paths[idx])
                caption = f"#{rank}  {float(score):.4f}"
                if os.path.isfile(path):
                    st.image(load_image(path), width=RESULT_IMAGE_WIDTH, caption=caption)
                else:
                    st.write(caption)
                    st.write(path)
        st.caption(f"Search time: {elapsed * 1000:.2f} ms")


if __name__ == "__main__":
    main()
