"""Which models can hold 1M context inside N GB? Computed from real HF configs.

The two numbers that decide it:
  kv_per_token = 2 * (attention layers) * kv_heads * head_dim * 2 bytes   [f16]
  weights_q4   = params * 4.85 / 8 / 2**30

Total at 1M = kv_per_token * 1M * 0.25 (q4 cache) + weights_q4
"""
import json
import urllib.request

CANDIDATES = [
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "nvidia/Nemotron-H-47B-Base-8K",
    "tiiuae/Falcon-H1-34B-Instruct",
    "tiiuae/Falcon-H1-7B-Instruct",
    "moonshotai/Kimi-Linear-48B-A3B-Instruct",
    "MiniMaxAI/MiniMax-M2",
    "Zyphra/Zamba2-7B",
    "ibm-granite/granite-4.1-8b-base",
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "kv-audit"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def text_cfg(cfg):
    return cfg.get("text_config", cfg) or cfg


def attention_layers(t):
    """Count layers that actually keep a KV cache."""
    lt = t.get("layer_types")
    if isinstance(lt, list) and lt:
        return sum(1 for x in lt if "full_attention" in str(x) or x == "attention")
    pat = t.get("hybrid_override_pattern")
    if isinstance(pat, str):
        return pat.count("M") if False else sum(1 for c in pat if c in "*A")
    n = t.get("num_hidden_layers") or t.get("n_layer") or 0
    if t.get("full_attention_interval"):
        return max(1, n // t["full_attention_interval"])
    return n


def analyse(repo):
    try:
        info = get(f"https://huggingface.co/api/models/{repo}")
        cfg = text_cfg(get(f"https://huggingface.co/{repo}/raw/main/config.json"))
    except Exception as e:
        return {"repo": repo, "error": str(e)[:40]}

    layers = cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0
    attn = attention_layers(cfg)
    kvh = cfg.get("num_key_value_heads") or cfg.get("n_head_kv") or cfg.get("num_attention_heads")
    hd = cfg.get("head_dim")
    if not hd:
        hidden = cfg.get("hidden_size") or cfg.get("d_model") or 0
        heads = cfg.get("num_attention_heads") or cfg.get("n_head") or 1
        hd = hidden // heads if heads else 0

    st = info.get("safetensors") or {}
    params = st.get("total")

    if not (attn and kvh and hd):
        return {"repo": repo, "error": f"unparsed (layers={layers} attn={attn} kvh={kvh} hd={hd})"}

    kv = 2 * attn * kvh * hd * 2
    q4 = params * 4.85 / 8 / 2**30 if params else None
    return {
        "repo": repo,
        "params_b": round(params / 1e9, 1) if params else None,
        "layers": layers,
        "attn": attn,
        "kv_kib": kv / 1024,
        "kv1m_q4_gb": kv * 1_048_576 / 4 / 2**30,
        "w_q4_gb": q4,
        "total_1m": (kv * 1_048_576 / 4 / 2**30 + q4) if q4 else None,
    }


rows = [analyse(r) for r in CANDIDATES]

print(f"{'model':<40} {'params':>7} {'attn':>5} {'KiB/tok':>8} {'1M@q4':>7} {'w@Q4':>7} {'TOTAL':>7}")
print("-" * 88)
for r in sorted(rows, key=lambda x: (x.get("total_1m") is None, x.get("total_1m") or 1e9)):
    if "error" in r:
        print(f"{r['repo']:<40}  !! {r['error']}")
        continue
    print(
        f"{r['repo']:<40} {r['params_b']:>6.1f}B {r['attn']:>5} "
        f"{r['kv_kib']:>7.0f} {r['kv1m_q4_gb']:>6.1f}G {r['w_q4_gb']:>6.1f}G {r['total_1m']:>6.1f}G"
    )

print("\nFits in 10 GB at 1M?", [r["repo"] for r in rows if r.get("total_1m") and r["total_1m"] <= 10] or "none")
print("Fits in 12 GB at 1M?", [r["repo"] for r in rows if r.get("total_1m") and r["total_1m"] <= 12] or "none")
