"""Sweep balance_weight and log fairness metrics to CSV."""
import argparse, csv, os, subprocess, sys
WEIGHTS=[0,0.05,0.1,0.2]; EXP="hybrid_hcvrp_4gb"; EPOCHS=10; OUT="hybrid/logs/sweep_balance.csv"
KEYS=["balance/time_cv","balance/time_max_min_ratio","balance/visits_cv"]
FIELDS=["balance_weight"]+KEYS+["status"]
def cmd(w,exp,ep): return [sys.executable,"hybrid/train.py",f"experiment={exp}",f"env.balance_weight={w}",f"trainer.max_epochs={ep}"]
def dry(ws,exp,ep,out):
    print(f"[dry-run] experiment={exp} epochs={ep} output={out}")
    for w in ws: print(f"  balance_weight={w}: {' '.join(cmd(w,exp,ep))} -> {out} ({','.join(KEYS)})")
    print("dry-run done")
def _via_hydra(w,exp,ep):
    try:
        from hydra import compose, initialize
        from hybrid.train import run as train_run
        from hydra.core.global_hydra import GlobalHydra
    except Exception as e: return None, f"import_fail: {e}"
    try:
        if GlobalHydra.instance().is_initialized(): GlobalHydra.instance().clear()
        with initialize(version_base="1.3", config_path="../configs"):
            cfg=compose(config_name="main.yaml", overrides=[f"experiment={exp}",f"env.balance_weight={w}",f"trainer.max_epochs={ep}"])
        metric_dict,_=train_run(cfg)
        vals={}
        for k in KEYS:
            v=metric_dict.get(k)
            if v is None: vals[k]=""
            else:
                try: vals[k]=float(v.mean().item() if hasattr(v,"mean") else float(v))
                except Exception: vals[k]=str(v)
        return vals, "success"
    except Exception as e: return None, f"fail: {e}"
    finally:
        try:
            from hydra.core.global_hydra import GlobalHydra
            if GlobalHydra.instance().is_initialized(): GlobalHydra.instance().clear()
        except Exception: pass
def run(ws,exp,ep,out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True); rows=[]
    # prefer in-process Hydra if available, else subprocess fallback (separate hydra/torch)
    use_hydra=False
    _hydra_err=None; _torch_err=None
    try: import hydra
    except Exception as e: _hydra_err=e
    else:
        try: import torch
        except Exception as e:
            _torch_err=e; print(f"  hydra available but torch missing: {e} -> subprocess fallback (no in-process metric capture)")
        else: use_hydra=True
    if _hydra_err is not None: print(f"  hydra not available: {_hydra_err} -> subprocess fallback")
    for w in ws:
        status="success"; vals={k:"" for k in KEYS}
        if use_hydra:
            got, status=_via_hydra(w,exp,ep)
            if got is not None: vals=got
            else: print(f"  hydra failed weight={w}: {status} -> subprocess fallback")
            if got is None: use_hydra=False  # fallback for remaining
        vals_empty=(vals=={k:"" for k in KEYS})
        if vals_empty and status=="success":
            status="success_no_metrics"; print(f"  WARNING: weight={w} no balance metrics captured -> {status}")
        if vals_empty and (not use_hydra or status!="success"):
            c=cmd(w,exp,ep); print(f"\n=== balance_weight={w} ===\n{' '.join(c)}")
            r=subprocess.run(c)
            if r.returncode!=0: status=f"fail:{r.returncode}"; print(f"  WARNING: failed weight={w} exit {r.returncode}")
            else:
                if vals=={k:"" for k in KEYS}: status="success_no_metrics"; print(f"  WARNING: weight={w} subprocess success but no metrics -> {status}")
                else: status="success"
        row={"balance_weight":w, **vals, "status":status}
        rows.append(row)
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")
def main():
    p=argparse.ArgumentParser(description="Sweep balance_weight")
    p.add_argument("--weights",nargs="*",type=float,default=None); p.add_argument("--experiment",default=EXP)
    p.add_argument("--epochs",type=int,default=EPOCHS); p.add_argument("--output",default=OUT); p.add_argument("--dry-run",action="store_true")
    a=p.parse_args(); ws=a.weights if a.weights is not None else WEIGHTS
    dry(ws,a.experiment,a.epochs,a.output) if a.dry_run else run(ws,a.experiment,a.epochs,a.output)
if __name__=="__main__": main()
