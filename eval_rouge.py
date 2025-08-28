import argparse
import os
import pandas as pd
from rouge_score import rouge_scorer

def score_pairs(pairs):
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeLsum"], use_stemmer=True)
    rows = []
    for ref, cand, name in pairs:
        s = scorer.score(ref, cand)
        rows.append({
            "name": name,
            "rouge1_f": s["rouge1"].fmeasure,
            "rouge2_f": s["rouge2"].fmeasure,
            "rougeLsum_f": s["rougeLsum"].fmeasure,
        })
    return pd.DataFrame(rows)

def load_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    assert {"reference","candidate"}.issubset(df.columns), "CSV must have 'reference' and 'candidate' columns"
    return [(row["reference"], row["candidate"], f"row_{i}") for i, row in df.iterrows()]

def load_from_dirs(ref_dir, cand_dir):
    names = sorted([os.path.splitext(f)[0] for f in os.listdir(ref_dir) if f.endswith(".txt")])
    pairs = []
    for n in names:
        ref_path = os.path.join(ref_dir, n + ".txt")
        cand_path = os.path.join(cand_dir, n + ".summary.txt")
        if os.path.exists(ref_path) and os.path.exists(cand_path):
            with open(ref_path, "r", encoding="utf-8") as fr, open(cand_path, "r", encoding="utf-8") as fc:
                pairs.append((fr.read(), fc.read(), n))
    if not pairs:
        raise ValueError("No filename-matched pairs found. Ensure refs are .txt and cands are .summary.txt")
    return pairs

def main():
    ap = argparse.ArgumentParser(description="Compute ROUGE scores")
    ap.add_argument("--csv", type=str, help="CSV with columns: reference, candidate")
    ap.add_argument("--refs", type=str, help="Directory of reference .txt files")
    ap.add_argument("--cands", type=str, help="Directory of candidate .summary.txt files")
    ap.add_argument("--out", type=str, default="rouge_scores.csv", help="Output CSV path")
    args = ap.parse_args()

    if args.csv:
        pairs = load_from_csv(args.csv)
    elif args.refs and args.cands:
        pairs = load_from_dirs(args.refs, args.cands)
    else:
        raise ValueError("Provide --csv OR both --refs and --cands")

    df = score_pairs(pairs)
    df.to_csv(args.out, index=False)
    print(f"Wrote ROUGE results to {args.out}")
    print(df.describe())

if __name__ == "__main__":
    main()
