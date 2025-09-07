"""
Evaluate SALAD-BENCH files with MD-Judge and

1.  print / save SAFE-rates (0 = SAFE, 1 = UNSAFE, 2 = exception)
2.  save all qids that were labelled UNSAFE plus their harm categories

Single-file mode:  --jsonl_file + --out_csv <file>
Directory   mode:  --jsonl_dir  (scans every *.jsonl)

Exits immediately if vLLM cannot load when --use_vllm is supplied.
"""

import argparse, glob, json, os, sys, pandas as pd
from tqdm import tqdm
from saladbench.evaluator import MDJudgeEvaluator, DDPMDJudgeEvaluator
from huggingface_hub.utils import HFValidationError


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def read_jsonl(fp):
    """Return lists: qids, questions, generations."""
    qids, qs, ans = [], [], []
    with open(fp, "r") as f:
        for line in f:
            obj = json.loads(line)
            qids.append(obj.get("qid"))     # may be None
            qs.append(obj["prompt"])
            ans.append(obj["generation"])
    return qids, qs, ans


def safe_rate(labels):
    return 100.0 * labels.count(0) / len(labels)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl_dir", required=True,
                   help="Folder containing generation files")
    p.add_argument("--jsonl_file", default=None,
                   help="Evaluate ONE jsonl instead of every file in dir")
    p.add_argument("--checkpoint", required=True,
                   help="Path or HF repo-id for MD-Judge weights")
    p.add_argument("--out_csv", required=True,
                   help="CSV file where this job writes the SAFE-rate row")
    p.add_argument("--use_vllm", action="store_true",
                   help="Run evaluator with vLLM backend")
    p.add_argument("--use_flash", action="store_true",
                   help="Flash-Attention kernels")
    p.add_argument("--ddp", action="store_true",
                   help="torchrun DDP across visible GPUs")
    return p.parse_args()


# ---------------------------------------------------------------------
def main():
    args = parse_args()

    JudgeCls = DDPMDJudgeEvaluator if args.ddp else MDJudgeEvaluator
    try:
        judge = JudgeCls(args.checkpoint,
                         use_vllm=args.use_vllm,
                         use_flash_attn=args.use_flash)
    except (ValueError, HFValidationError, FileNotFoundError) as e:
        print("[ERROR] MD-Judge load failed:\n", e, file=sys.stderr)
        sys.exit(1)

    # ---- which files --------------------------------------------------
    if args.jsonl_file:
        files = [args.jsonl_file]
    else:
        files = sorted(glob.glob(os.path.join(args.jsonl_dir, "*.jsonl")))
        if not files:
            print(f"[ERROR] No *.jsonl files in {args.jsonl_dir}", file=sys.stderr)
            sys.exit(1)

    # ---- per-file evaluation -----------------------------------------
    for fp in tqdm(files, desc="Models"):
        tag = os.path.splitext(os.path.basename(fp))[0]
        qids, questions, answers = read_jsonl(fp)
        total = len(questions)

        print(f"Scoring {tag}  ({total} Q-A pairs)")

        labels, clses = judge.eval_unsafe_labels(questions, answers)

        # ----- save UNSAFE qids + categories ---------------------------
        unsafe_rows = [
            {"qid": qid, "category": cls}
            for qid, lab, cls in zip(qids, labels, clses)
            if lab == 1
        ]
        unsafe_csv = os.path.splitext(args.out_csv)[0] + "_unsafe_qids.csv"
        pd.DataFrame(unsafe_rows).to_csv(unsafe_csv, index=False)

        # ----- SAFE-rate row  -----------------------------------------
        rate = safe_rate(labels)
        pd.DataFrame([{
            "model": tag,
            "total": total,
            "safe_rate(%)": f"{rate:.2f}"
        }]).to_csv(args.out_csv, index=False)

        print(f"Finished {tag} | SAFE {rate:.2f}% "
              f"({labels.count(0)}/{total})")


if __name__ == "__main__":
    main()
