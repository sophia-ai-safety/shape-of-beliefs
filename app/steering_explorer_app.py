"""
Streamlit app for steering experiments.
Based on the setup of "The Shapes of Beliefs"
Allows to visualize the output distributions when steering \mu=300 activations towards \mu=700.

Knobs are:
- steering coefficient \alpha
- layer(s) to apply steering
- number of tokens to steer

Run:
    uv run streamlit run app/steering_explorer_app.py
"""

from __future__ import annotations

from pathlib import Path
import json
import streamlit as st
import plotly.graph_objects as go
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B"

DATASET_M300 = "gaussian_m300_s100_l1000_n10"
DATASET_M700 = "gaussian_m700_s100_l1000_n10"

LAST_N_COM2NUM = 500
TEMPERATURE = 1.0

BASE_DIR = Path.cwd()
SEQUENCES_DIR = BASE_DIR / "data" / "sequences"
ACTIVS_DIR = BASE_DIR / "data" / "activations"
TOKEN_SUBSET_PATH = BASE_DIR / "token_subset" / "llama3-2-1B_number_tokens.json"

MAX_ALPHA = 3.0


@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )
    model.eval()
    return model, tokenizer


@st.cache_data
def load_sequence_text(dataset_name: str, index: int) -> str:
    jsonl_path = SEQUENCES_DIR / f"{dataset_name}.jsonl"
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if i == index:
                return record["sequence_content"]
    raise ValueError(f"Sequence index {index} not found in {jsonl_path}")


@st.cache_data
def load_token_subset():
    token_map = json.loads(TOKEN_SUBSET_PATH.read_text(encoding="utf-8"))
    subset_items = sorted(token_map.items(), key=lambda kv: kv[1])
    token_strings = [k for k, _ in subset_items]
    token_ids = [v for _, v in subset_items]
    if len(token_ids) != len(set(token_ids)):
        raise ValueError("Duplicate token IDs in token subset file.")
    return token_strings, token_ids


@st.cache_data
def compute_centroid(dataset_name: str, last_n: int, layer: int) -> torch.Tensor:
    total = None
    count = 0
    site = f"model.layers.{layer}".replace(".", "_")
    ds_dir = ACTIVS_DIR / dataset_name
    pattern = f"{site}_batch*.pt"
    files = sorted(ds_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No activation files for {dataset_name}: {ds_dir / pattern}")
    for fpath in files:
        payload = torch.load(fpath, map_location="cpu")
        acts = payload["activations"]  # [batch, seq_len, d_model]
        lengths = payload.get("lengths")
        for i in range(acts.shape[0]):
            length = int(lengths[i].item()) if lengths is not None else acts.shape[1]
            idx = torch.arange(2, length, 2)
            if idx.numel() == 0:
                continue
            if idx.numel() > last_n:
                idx = idx[-last_n:]
            selected = acts[i, idx]  # [n, d_model]
            if total is None:
                total = selected.sum(dim=0)
            else:
                total += selected.sum(dim=0)
            count += selected.shape[0]
    if total is None or count == 0:
        raise ValueError(f"No activations found for {dataset_name}")
    return total / count


def run_model_with_steering(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    steering_vecs: dict[int, torch.Tensor],
    alpha: float,
    last_n_tokens: int,
) -> torch.Tensor:
    handles = []

    def make_hook(vec: torch.Tensor):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                hidden = output[0]
                hidden = hidden.clone()
                n = min(last_n_tokens, hidden.shape[1])
                hidden[:, -n:, :] = hidden[:, -n:, :] + alpha * vec
                return (hidden,) + tuple(output[1:])
            hidden = output.clone()
            n = min(last_n_tokens, hidden.shape[1])
            hidden[:, -n:, :] = hidden[:, -n:, :] + alpha * vec
            return hidden

        return hook

    for layer, vec in steering_vecs.items():
        layer_module = model.get_submodule(f"model.layers.{layer}")
        handles.append(layer_module.register_forward_hook(make_hook(vec)))
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    return outputs.logits


def subset_distribution(logits: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    subset_logits = logits.index_select(dim=-1, index=token_ids_tensor)
    probs = torch.softmax(subset_logits / TEMPERATURE, dim=-1)
    return subset_logits, probs


def order_subset_labels(token_strings: list[str]) -> tuple[list[int], list[str]]:
    numeric = [(i, int(lbl)) for i, lbl in enumerate(token_strings) if lbl.isdigit()]
    numeric_sorted = sorted(numeric, key=lambda t: t[1])
    num_indices = [i for i, _ in numeric_sorted]
    other_indices = [i for i, lbl in enumerate(token_strings) if not lbl.isdigit()]
    ordered_indices = num_indices + other_indices
    ordered_labels = [token_strings[i] for i in ordered_indices]
    return ordered_indices, ordered_labels


def mean_std_over_numeric(
    probs: torch.Tensor, token_strings: list[str]
) -> tuple[float, float]:
    numeric_pairs = [(i, int(lbl)) for i, lbl in enumerate(token_strings) if lbl.isdigit()]
    if not numeric_pairs:
        raise ValueError("No numeric tokens found in subset.")
    idxs = torch.tensor([i for i, _ in numeric_pairs], dtype=torch.long, device=probs.device)
    values = torch.tensor([v for _, v in numeric_pairs], dtype=torch.float32, device=probs.device)

    p = probs.index_select(dim=-1, index=idxs)
    p = p / p.sum()
    mean = (p * values).sum()
    var = (p * (values - mean) ** 2).sum()
    return float(mean.item()), float(torch.sqrt(var).item())


def main():
    st.set_page_config(page_title="Steering Explorer", layout="wide")
    st.title("Steering Effect Visualization")

    model, tokenizer = load_model_and_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if "layers" not in st.session_state:
        st.session_state.layers = [15]
    if "layers_prev" not in st.session_state:
        st.session_state.layers_prev = st.session_state.layers

    seq_index = st.selectbox("Sequence index", options=list(range(10)), index=0)
    sequence_text = load_sequence_text(DATASET_M300, int(seq_index))
    st.write("Sequence from m300:")
    st.code(sequence_text[:500] + ("..." if len(sequence_text) > 500 else ""))

    st.write("Steering at layer(s)")
    layer_options = list(range(0, 16))
    cols = st.columns(16)
    selected = []
    for idx, layer in enumerate(layer_options):
        col = cols[idx]
        checked = col.checkbox(
            f"{layer}",
            value=layer in st.session_state.layers,
            key=f"layer_{layer}",
        )
        if checked:
            selected.append(layer)
    st.session_state.layers = selected
    if st.session_state.layers != st.session_state.layers_prev:
        st.session_state.alpha_points = []
        st.session_state.layers_prev = st.session_state.layers

    token_strings, token_ids = load_token_subset()

    with st.spinner("Computing steering vector from activations..."):
        steering_vecs = {}
        for layer in st.session_state.layers:
            centroid_300 = compute_centroid(DATASET_M300, LAST_N_COM2NUM, layer)
            centroid_700 = compute_centroid(DATASET_M700, LAST_N_COM2NUM, layer)
            steering_vecs[layer] = (centroid_700 - centroid_300).to(dtype=torch.float32)

    alpha = st.slider("alpha (steering strength)", min_value=-MAX_ALPHA, max_value=MAX_ALPHA, value=0.0, step=0.1)
    last_n_tokens = st.slider("steer last n tokens", min_value=1, max_value=100, value=1, step=1)

    average_all = st.checkbox("Average over all sequences", value=False)

    def compute_subset_probs_for_text(text: str):
        enc = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=13000,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            steered_logits = run_model_with_steering(
                model,
                input_ids,
                attention_mask,
                {k: v.to(device) for k, v in steering_vecs.items()},
                alpha,
                last_n_tokens,
            )
        base_last = base_logits[0, -1]
        steered_last = steered_logits[0, -1]
        _, base_probs = subset_distribution(base_last, token_ids)
        _, steered_probs = subset_distribution(steered_last, token_ids)
        return base_probs, steered_probs

    if average_all:
        base_probs_list = []
        steered_probs_list = []
        for i in range(10):
            text_i = load_sequence_text(DATASET_M300, i)
            base_p, steered_p = compute_subset_probs_for_text(text_i)
            base_probs_list.append(base_p)
            steered_probs_list.append(steered_p)
        base_subset_probs = torch.stack(base_probs_list, dim=0).mean(dim=0)
        steered_subset_probs = torch.stack(steered_probs_list, dim=0).mean(dim=0)
    else:
        base_subset_probs, steered_subset_probs = compute_subset_probs_for_text(sequence_text)

    ordered_indices, ordered_labels = order_subset_labels(token_strings)
    base_pdf = base_subset_probs.detach().cpu().numpy()[ordered_indices]
    steered_pdf = steered_subset_probs.detach().cpu().numpy()[ordered_indices]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(ordered_labels))),
            y=base_pdf,
            mode="lines",
            line=dict(color="#3B82F6", width=1.5),
            name="base",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(ordered_labels))),
            y=steered_pdf,
            mode="lines",
            line=dict(color="#EF4444", width=1.5),
            name=f"steered (alpha={alpha:.1f})",
        )
    )

    numeric_labels = [lbl for lbl in ordered_labels if lbl.isdigit()]
    tick_vals = [ordered_labels.index(lbl) for lbl in numeric_labels if int(lbl) % 100 == 0]
    tick_text = [lbl for lbl in numeric_labels if int(lbl) % 100 == 0]

    fig.update_layout(
        title="Output Distribution (0-999 + delimiters)",
        xaxis=dict(
            title="token (ordered: 0-999, then delimiters)",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
        ),
        yaxis=dict(title="probability"),
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    steered_mean, steered_std = mean_std_over_numeric(steered_subset_probs, token_strings)

    if "alpha_points" not in st.session_state:
        st.session_state.alpha_points = []
    st.session_state.alpha_points.append((alpha, steered_mean, steered_std))

    points = st.session_state.alpha_points
    alphas = [p[0] for p in points]
    means = [p[1] for p in points]
    stds = [p[2] for p in points]

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=means,
            y=stds,
            mode="markers",
            marker=dict(
                size=10,
                color=alphas,
                colorscale="Portland",
                cmin=-MAX_ALPHA,
                cmax=MAX_ALPHA,
                showscale=True,
                colorbar=dict(title="alpha"),
            ),
            name="steered points",
        )
    )
    fig2.update_layout(
        title="Mean vs Std of Output Distribution (0-999)",
        xaxis=dict(title="mean", range=[100, 900]),
        yaxis=dict(title="std", range=[50, 150]),
        height=420,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig2.update_yaxes(scaleanchor="x", scaleratio=4)
    st.plotly_chart(fig2, width=512) # width = "stretch"


if __name__ == "__main__":
    main()
