#!/usr/bin/env python3
"""Run test-only counterfactual inference for every model on the extended-basal
dataset, reusing the existing trained checkpoints. No training, no HP search."""
from __future__ import annotations
import argparse, subprocess, sys, shutil
from pathlib import Path
import yaml

MODELS = ["arimax","dlinear","gru","itransformer","patchmlp","patchmlp_glimmer","attengluco"]
SRC_CFG = Path("configs/counterfactual_best_controliq")
OLD_RUN = "action_cf_controliq_best_{m}"


def build_configs(dataset, out_dir, models, source_run=OLD_RUN, src_config_dir=None):
    # Build configs from the RUN DIRECTORY's own saved config, not from configs/.
    # The configs tree has drifted from the trained checkpoints (dlinear is stored
    # with individual=False while configs/ says True); a mismatch makes the
    # checkpoint load silently fail and the model run on untrained weights.
    (out_dir / 'eval').mkdir(parents=True, exist_ok=True)
    (out_dir / 'train').mkdir(parents=True, exist_ok=True)
    made = {}
    for m in models:
        if src_config_dir is not None:
            # Checkpoints trained with --train only have no config.yaml in the run dir;
            # the configs that produced them live in a config tree instead.
            paths = {}
            missing = False
            for kind in ('train', 'eval'):
                srcf = Path(src_config_dir) / kind / (m + '.yaml')
                if not srcf.exists():
                    print('  [skip] %s: no %s config at %s' % (m, kind, srcf), flush=True)
                    missing = True
                    break
                cfg = yaml.safe_load(srcf.read_text())
                cfg.setdefault('split', {})['split_name'] = dataset
                dst = out_dir / kind / (m + '.yaml')
                dst.write_text(yaml.safe_dump(cfg))
                paths[kind] = dst
            if not missing:
                made[m] = (paths['train'], paths['eval'])
            continue
        run_cfg = Path('outputs/results') / source_run.format(m=m) / 'config.yaml'
        if not run_cfg.exists():
            print('  [skip] %s: no run config at %s' % (m, run_cfg), flush=True)
            continue
        c = yaml.safe_load(run_cfg.read_text())
        if 'train' not in c or 'eval' not in c:
            print('  [skip] %s: run config missing train/eval sections' % m, flush=True)
            continue
        paths = {}
        for kind in ('train', 'eval'):
            cfg = c[kind]
            cfg.setdefault('split', {})['split_name'] = dataset
            dst = out_dir / kind / (m + '.yaml')
            dst.write_text(yaml.safe_dump(cfg))
            paths[kind] = dst
        made[m] = (paths['train'], paths['eval'])
    return made


def _checkpoint_loaded(run_dir):
    lg = run_dir / 'logs.log'
    if not lg.exists():
        return True
    return 'could not load state_dict' not in lg.read_text().lower()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="action_cf_controliq_basal8")
    p.add_argument("--cfg-out", type=Path, default=Path("configs/counterfactual_basal8"))
    p.add_argument("--run-prefix", default="basal8_")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--horizons", default="30,60,120")
    p.add_argument("--src-config-dir", default=None,
                   help="Read train/eval configs from <dir>/{train,eval}/<model>.yaml instead of run dirs.")
    p.add_argument("--source-run", default=OLD_RUN,
                   help="Run-dir template supplying both the saved config and the checkpoint, e.g. cf4hb_{m}.")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    cfgs = build_configs(a.dataset, a.cfg_out, models, a.source_run, a.src_config_dir)
    if not cfgs:
        sys.exit("[infer] FATAL: no model configs could be built -- refusing to continue")
    print(f"[infer] configs built for {len(cfgs)} models -> {a.cfg_out}", flush=True)

    ok, bad = [], []
    for m, (tr, ev) in cfgs.items():
        run_id = f"{a.run_prefix}{m}"
        run_dir = Path("outputs/results") / run_id
        if run_dir.exists() and not a.force:
            if (run_dir / "cf_artifacts.npz").exists():
                print(f"[infer] {m}: already done, skipping", flush=True); ok.append(m); continue
            shutil.rmtree(run_dir)
        ckpt = Path("outputs/results") / a.source_run.format(m=m)
        if not (ckpt / "checkpoint.pt").exists():
            print(f"[infer] {m}: MISSING checkpoint at {ckpt}", flush=True); bad.append(m); continue
        cmd = [sys.executable, "exp/run_counterfactual_with_metrics.py", "--test",
               "--train-config", str(tr), "--eval-config", str(ev),
               "--run-id", run_id, "--checkpoint-dir", str(ckpt),
               "--horizons", a.horizons]
        print(f"[infer] {m}: running ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not (run_dir / "cf_artifacts.npz").exists():
            tail = (r.stderr or r.stdout).strip().splitlines()[-6:]
            print(f"[infer] {m}: FAILED rc={r.returncode}", flush=True)
            for l in tail: print(f"         {l}", flush=True)
            bad.append(m)
        elif not _checkpoint_loaded(run_dir):
            print(f"[infer] {m}: FAILED - checkpoint did not load (untrained weights)", flush=True)
            bad.append(m)
        else:
            print(f"[infer] {m}: ok", flush=True); ok.append(m)

    print(f"\n[infer] succeeded: {len(ok)} {ok}", flush=True)
    if bad:
        print(f"[infer] FAILED: {len(bad)} {bad}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
