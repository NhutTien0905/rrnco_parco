"""Sweep balance_weight and log fairness metrics to CSV."""
import argparse, csv, subprocess, sys, os
WEIGHTS=[0,0.05,0.1,0.2]; EXP="hybrid_hcvrp_4gb"; EPOCHS=10; OUT="hybrid/logs/sweep_balance.csv"
def cmd(w, exp, ep): return [sys.executable,"hybrid/train.py",f"experiment={exp}",f"env.balance_weight={w}",f"trainer.max_epochs={ep}"]
def dry(ws, exp, ep, out):
    print(f"[dry-run] experiment={exp} epochs={ep} output={out}")
    for w in ws: print(f"  balance_weight={w}: {' '.join(cmd(w,exp,ep))} -> {out} (balance/time_cv,time_max_min_ratio,visits_cv)")
    print("dry-run done")
def run(ws, exp, ep, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True); rows=[]
    for w in ws:
        c=cmd(w,exp,ep); print(f"\n=== balance_weight={w} ===\n{' '.join(c)}")
        r=subprocess.run(c)
        if r.returncode!=0: print(f"  WARNING: failed weight={w} exit {r.returncode}")
        rows.append({"balance_weight":w,"balance/time_cv":"","balance/time_max_min_ratio":"","balance/visits_cv":""})
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["balance_weight","balance/time_cv","balance/time_max_min_ratio","balance/visits_cv"]); w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")
def main():
    p=argparse.ArgumentParser(description="Sweep balance_weight")
    p.add_argument("--weights",nargs="*",type=float,default=None); p.add_argument("--experiment",default=EXP)
    p.add_argument("--epochs",type=int,default=EPOCHS); p.add_argument("--output",default=OUT); p.add_argument("--dry-run",action="store_true")
    a=p.parse_args(); ws=a.weights if a.weights is not None else WEIGHTS
    dry(ws,a.experiment,a.epochs,a.output) if a.dry_run else run(ws,a.experiment,a.epochs,a.output)
if __name__=="__main__": main()
