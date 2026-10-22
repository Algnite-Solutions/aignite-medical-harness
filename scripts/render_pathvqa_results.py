"""Render the public PathVQA summary as a PR-ready Markdown table."""
import argparse
import json
from pathlib import Path


def render(summary: dict) -> str:
    rows = ["| Configuration | Completed | Valid answers | Binary accuracy | Open EM | Open token F1 | Strict AMA format | Retries |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for run in summary["runs"]:
        scores = run["aggregate"]
        quality = scores["output_quality"]["strict_output_compliance"]
        term = run["termination_counts"]
        completed = term.get("completed", 0)
        valid = round(scores["valid_answer_rate"] * scores["n_expected"])
        binary = round(scores["binary_accuracy"] * scores["binary_count"])
        open_em = round(scores["open_exact_match"] * scores["open_count"])
        rows.append(f"| `{run['alias']}` | {completed}/{scores['n_expected']} | "
                    f"{valid}/{scores['n_expected']} | "
                    f"{binary}/{scores['binary_count']} | {open_em}/{scores['open_count']} | "
                    f"{scores['open_token_f1']:.4f} | {quality['num']}/{quality['den']} | "
                    f"{run['n_retries']} |")
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    table = render(json.loads(args.summary.read_text(encoding="utf-8")))
    if args.out:
        args.out.write_text(table, encoding="utf-8")
    else:
        print(table, end="")


if __name__ == "__main__":
    main()
