import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from models import Model, gene_pearson, masked_mse


def make_split(n, seed=3, n_test=448, n_val=448):
    p = np.random.default_rng(seed).permutation(n)
    te, va = p[:n_test], p[n_test:n_test + n_val]
    return np.sort(np.setdiff1d(np.arange(n), p[:n_test + n_val])), np.sort(va), np.sort(te)


def evaluate(P, Y):
    M = ~np.isnan(Y)
    r = np.array([pearsonr(p[m], y[m])[0] if m.sum() > 2 and np.std(p[m]) > 0 else np.nan
                  for p, y, m in zip(P, Y, M)])
    pf, yf = P[M], Y[M]
    return dict(pearson_gene=float(np.nanmean(r)), pearson_all=float(pearsonr(pf, yf)[0]),
                spearman=float(spearmanr(pf, yf)[0]), rmse=float(np.sqrt(np.mean((pf - yf) ** 2))))


def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Y = np.load(f"{a.data}/target.npz", allow_pickle=True)["Y"]
    X = np.concatenate([np.load(f"{a.data}/feat_{f}.npz")["X"] for f in a.feats], 1).astype(np.float32)
    tr, va, te = make_split(len(Y), a.split_seed)

    model = Model(X.shape[1], Y.shape[1], a.head).to(dev)
    Xt = torch.as_tensor(X, device=dev)
    Yt = torch.as_tensor(np.nan_to_num(Y), device=dev)
    Mt = torch.as_tensor(~np.isnan(Y), device=dev).float()
    tr_t = torch.as_tensor(tr, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    def predict(idx):
        model.eval()
        with torch.no_grad():
            return model(Xt[idx])

    best, best_state, bad = -np.inf, None, 0
    for ep in range(a.epochs):
        model.train()
        for b in torch.split(tr_t[torch.randperm(len(tr_t), device=dev)], a.batch):
            if len(b) < 2:
                continue
            pred = model(Xt[b])
            loss = masked_mse(pred, Yt[b], Mt[b])
            if a.lam > 0:
                loss = loss + a.lam * (1 - gene_pearson(pred, Yt[b], Mt[b]).mean())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        rv = gene_pearson(predict(va), Yt[va], Mt[va]).mean().item()
        if rv > best:
            best, bad = rv, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if bad >= a.patience:
            break
    model.load_state_dict(best_state)

    P = predict(te).cpu().numpy()
    m = evaluate(P, Y[te])
    m.update(val_pearson_gene=best, feats=a.feats, head=a.head, lam=a.lam, seed=a.seed)
    os.makedirs(a.out, exist_ok=True)
    np.save(f"{a.out}/pred_test.npy", P.astype(np.float16))
    json.dump(m, open(f"{a.out}/metrics.json", "w"), indent=1)
    print(json.dumps(m))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/gogi_film/seed42")
    ap.add_argument("--feats", nargs="+", default=["go", "gi"], choices=["go", "esm", "gi"])
    ap.add_argument("--head", default="film", choices=["film", "dot"])
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    main(ap.parse_args())
