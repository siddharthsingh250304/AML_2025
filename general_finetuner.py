import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

# -------------------- CONFIG --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

CSV_PATH = "train.csv"
TEXT_COL = "catalog_content"
PRICE_COL = "price"
TRAIN_FRACTION = 0.7

# DistilUSE Multilingual: 135M params, 512-dim embeddings (already optimized!)
INIT_CKPT = "sentence-transformers/distiluse-base-multilingual-cased-v2"
OUT_DIR = "distiluse_lora_triplet"
os.makedirs(OUT_DIR, exist_ok=True)

# Training schedule - OPTIMIZED
TOTAL_TRIPLETS = 500_000
EPOCHS = 20
TRIPLETS_PER_EPOCH = TOTAL_TRIPLETS // EPOCHS
BATCH_TRIPLETS = 32
STEPS_PER_EPOCH = math.ceil(TRIPLETS_PER_EPOCH / BATCH_TRIPLETS)

LR = 2e-5
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
USE_AMP = True
MARGIN = 0.5

# Log-binning
Q_BINS = 16
LOG_EPS = 1e-6
DLOG_MIN = 1.0
MIN_NEG_POOL = 2000
RELAX_STEP = 0.2

# LoRA settings - DistilUSE uses DistilBERT backbone
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_lin", "k_lin", "v_lin", "out_lin"]  # DistilBERT naming

# -------------------- DATA PREP --------------------
def prepare_bins(df, price_col=PRICE_COL, q=Q_BINS, eps=LOG_EPS):
    df = df.copy()
    df["_log_price"] = np.log(df[price_col].astype(float) + eps)
    df["bin"] = pd.qcut(df["_log_price"], q=q, duplicates="drop")
    df["bin_id"] = df["bin"].cat.codes
    bstats = df.groupby("bin_id").agg(
        log_med=("_log_price","median"),
        n=(price_col,"size")
    ).reset_index()
    bin_to_idx = {int(b): df.index[df["bin_id"]==b].to_list() for b in bstats["bin_id"]}
    valid_bins = [b for b in bin_to_idx if len(bin_to_idx[b]) >= 2]
    return df, bstats, bin_to_idx, valid_bins

def build_negative_bins(bstats, anchor_bin, dlog_min=DLOG_MIN, min_pool=MIN_NEG_POOL,
                        relax_step=RELAX_STEP, dlog_max=3.0):
    row = bstats[bstats["bin_id"]==anchor_bin]
    if row.empty:
        return []
    m_anchor = float(row["log_med"].iloc[0])
    d = bstats.copy()
    d["gap"] = (d["log_med"] - m_anchor).abs()
    d = d[d["bin_id"] != anchor_bin]
    thr = dlog_min
    while True:
        cand = d[d["gap"] >= thr].sort_values("gap", ascending=True)
        total = int(cand["n"].sum())
        if total >= min_pool or thr <= 0.0:
            return [int(x) for x in cand["bin_id"].to_list()]
        thr = max(0.0, thr - relax_step)
        if thr < (dlog_min - dlog_max):
            return [int(x) for x in cand["bin_id"].to_list()]

class TripletTextDataset(Dataset):
    def __init__(self, df_raw, text_col=TEXT_COL, price_col=PRICE_COL):
        self.df, self.bstats, self.bin_to_idx, self.valid_bins = prepare_bins(df_raw, price_col=price_col)
        self.text_col = text_col
        self.price_col = price_col
        self.neg_bins = {
            b: build_negative_bins(self.bstats, b, dlog_min=DLOG_MIN, min_pool=MIN_NEG_POOL, relax_step=RELAX_STEP)
            for b in self.valid_bins
        }
        self._length = TRIPLETS_PER_EPOCH

    def __len__(self):
        return self._length

    def __getitem__(self, _):
        if not self.valid_bins:
            a, p, n = np.random.choice(self.df.index, 3, replace=False)
        else:
            b = random.choice(self.valid_bins)
            a, p = np.random.choice(self.bin_to_idx[b], 2, replace=False)
            nbins = self.neg_bins.get(b, [])
            if nbins:
                nbin = random.choice(nbins)
            else:
                r = self.bstats[self.bstats["bin_id"]==b]
                m_anchor = float(r["log_med"].iloc[0])
                cand = self.bstats[self.bstats["bin_id"]!=b].copy()
                cand["gap"] = (cand["log_med"] - m_anchor).abs()
                nbin = int(cand.sort_values("gap", ascending=False)["bin_id"].iloc[0])
            n = random.choice(self.bin_to_idx.get(nbin, [a]))

        def s(x):
            v = self.df.loc[x, self.text_col]
            return str(v) if pd.notna(v) else ""

        return {"a": s(a), "p": s(p), "n": s(n)}

def collate_triplets(batch):
    return {
        "a": [b["a"] for b in batch],
        "p": [b["p"] for b in batch],
        "n": [b["n"] for b in batch],
    }

# -------------------- MODEL + LORA --------------------
def mean_pooling(model_output, attention_mask):
    """Mean pooling - already done internally, but included for compatibility"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def inject_lora(distiluse_model):
    """Inject LoRA into DistilUSE (DistilBERT backbone)"""
    distiluse_model.requires_grad_(False)
    lcfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=TARGET_MODULES,
        inference_mode=False,
        modules_to_save=[]
    )
    peft_model = get_peft_model(distiluse_model, lcfg)
    for n, p in peft_model.named_parameters():
        p.requires_grad_(("lora_" in n))
    return peft_model

def encode_texts(tokenizer, model, texts):
    """
    Encode texts using DistilUSE.

    DistilUSE already has mean pooling + dense layer built-in,
    but we extract from the transformer backbone for LoRA training.
    """
    tok = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128  # DistilUSE default max length
    )
    tok = {k: v.to(DEVICE) for k, v in tok.items()}

    out = model(**tok, return_dict=True)

    # Mean pooling
    pooled = mean_pooling(out, tok["attention_mask"])

    # L2 normalize
    pooled = F.normalize(pooled, p=2, dim=1)
    return pooled

# -------------------- LOSS --------------------
class TripletCosineLoss(nn.Module):
    def __init__(self, margin=MARGIN):
        super().__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, a, p, n):
        d_ap = 1.0 - (a * p).sum(dim=-1)
        d_an = 1.0 - (a * n).sum(dim=-1)
        loss = self.relu(d_ap - d_an + self.margin)
        return loss.mean()

# -------------------- TRAIN --------------------
def train():
    df_full = pd.read_csv(CSV_PATH)
    assert TEXT_COL in df_full.columns and PRICE_COL in df_full.columns

    train_size = int(len(df_full) * TRAIN_FRACTION)
    df = df_full.iloc[:train_size].reset_index(drop=True)

    print(f"Total data: {len(df_full)} rows")
    print(f"Using first {TRAIN_FRACTION*100:.0f}%: {len(df)} rows")

    ds = TripletTextDataset(df, text_col=TEXT_COL, price_col=PRICE_COL)
    dl = DataLoader(ds, batch_size=BATCH_TRIPLETS, shuffle=True,
                    num_workers=8, collate_fn=collate_triplets, drop_last=True,
                    pin_memory=True, persistent_workers=True)

    # Load DistilUSE base (just the transformer, not the full SentenceTransformer)
    print(f"Loading model: {INIT_CKPT}")
    tokenizer = AutoTokenizer.from_pretrained(INIT_CKPT)
    # Load only the transformer part (0th module)
    base_model = AutoModel.from_pretrained(INIT_CKPT)
    model = inject_lora(base_model).to(DEVICE)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = EPOCHS * STEPS_PER_EPOCH
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    criterion = TripletCosineLoss(margin=MARGIN)

    print(f"\nTraining: {TOTAL_TRIPLETS:,} triplets over {EPOCHS} epochs")
    print(f"Steps/epoch: {STEPS_PER_EPOCH}\n")

    for epoch in range(EPOCHS):
        pbar = tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch+1}/{EPOCHS}")
        data_iter = iter(dl)
        running = 0.0

        for _ in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                a = encode_texts(tokenizer, model, batch["a"])
                p = encode_texts(tokenizer, model, batch["p"])
                n = encode_texts(tokenizer, model, batch["n"])
                loss = criterion(a, p, n)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(optim_params, GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch+1} mean loss: {running / STEPS_PER_EPOCH:.4f}")

    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\n✅ Saved to: {OUT_DIR}")

if __name__ == "__main__":
    train()
