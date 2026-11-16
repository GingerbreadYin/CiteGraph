"""Score saved RAG baselines offline. No models, HTTP calls or new dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from citations import validate_citations


def read_snapshot(path):
    data = path.read_bytes()
    return json.loads(data), {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()}


def context_precision(required, results, k):
    """Mean precision at the ranks of relevant chunks within the first k."""
    evidence_ids = {evidence_id for group in required for evidence_id in group}
    if not evidence_ids:
        return None

    relevant = 0
    precision_sum = 0.0
    for rank, chunk in enumerate(results[:k], start=1):
        labels = set(chunk.get("evidence_ids", []))
        labels.update(span["id"] for span in chunk.get("evidence_spans", []))
        if labels & evidence_ids:
            relevant += 1
            precision_sum += relevant / rank
    return precision_sum / relevant if relevant else 0.0


def retrieval_baseline(run_dir):
    reports = []
    for folder, split in (
        ("baseline", "development"),
        ("baseline-heldout", "heldout"),
        ("regression-baseline", "original-regression"),
    ):
        report, source = read_snapshot(run_dir / folder / "report.json")
        if report["split"] != split:
            raise ValueError(f"Unexpected retrieval split: {report['split']!r}")
        k = report["config"]["top_k"]
        rows = []
        for case in report["cases"]:
            selected = case["results"][:k]
            rows.append({
                "id": case["id"],
                "question": case["question"],
                "required_evidence_groups": case["required"],
                "candidate_ids": [chunk["id"] for chunk in selected],
                "context_characters": sum(len(chunk["text"]) for chunk in selected),
                "context_precision": context_precision(case["required"], selected, k),
                "evidence_group_recall": case["scores"]["recall"],
                "complete_evidence_coverage": case["scores"]["complete"],
            })
        scores = [row["context_precision"] for row in rows if row["context_precision"] is not None]
        reports.append({
            "split": split,
            "source": source,
            "corpus_sha256": report["worker"]["corpus_sha256"],
            "config": report["config"],
            "scored_cases": len(scores),
            "excluded_no_required_evidence": len(rows) - len(scores),
            "context_precision": mean(scores) if scores else None,
            "existing_metrics": report["metrics"],
            "cases": rows,
        })
    return {
        "stage": "archived budgeted_context",
        "method": "Sum precision at each relevant rank / relevant chunks in top_k. "
                  "Overlap with annotated evidence counts as relevance. No required labels = N/A.",
        "splits": reports,
    }


def score_claims(claims, context):
    """Aggregate human-reviewed claim verdicts and check their evidence pointers."""
    if not claims:
        raise ValueError("An answered case needs at least one claim")
    supported = 0
    for claim in claims:
        if claim["verdict"] not in {"supported", "unsupported"}:
            raise ValueError(f"Unknown claim verdict: {claim['verdict']!r}")
        if not claim["claim"].strip() or (claim["verdict"] == "supported" and not claim["evidence"]):
            raise ValueError("Claim text is required and supported claims need evidence references")
        for evidence in claim["evidence"]:
            quote = evidence["quote"]
            if evidence["id"] not in context or not quote.strip() or quote not in context[evidence["id"]]:
                raise ValueError(f"Evidence quote does not match supplied chunk: {evidence['id']}")
        supported += claim["verdict"] == "supported"
    return supported / len(claims)


def faithfulness_baseline(project_dir, annotations_path):
    annotations, annotation_source = read_snapshot(annotations_path)
    names = ("chunks.json", "e2e_report_e2e_cases.json", "e2e_report_e2e_invoice_cases.json")
    snapshots, sources = {}, []
    for name in names:
        data, source = read_snapshot(project_dir / name)
        if source["sha256"] != annotations["source_sha256"][name]:
            raise ValueError(f"Historical input changed. Review the annotations again: {name}")
        snapshots[name] = data
        sources.append(source)

    chunks = {chunk["id"]: chunk["text"] for chunk in snapshots["chunks.json"]}
    rows, excluded = [], []
    case_ids = set()
    for name in names[1:]:
        for item in snapshots[name]:
            case_id, result = item["id"], item["result"]
            if result["status"] != "ANSWERED":
                excluded.append({"id": case_id, "status": result["status"]})
                continue
            candidate_ids = [candidate["id"] for candidate in result["retrieved_sources"]]
            context = {candidate_id: chunks[candidate_id] for candidate_id in candidate_ids}
            claims = annotations["claims_by_case"][case_id]
            case_ids.add(case_id)
            rows.append({
                "id": case_id,
                "question": item["question"],
                "answer": result["answer"],
                "candidate_ids": candidate_ids,
                "claims": claims,
                "faithfulness": score_claims(claims, context),
            })
    if case_ids != set(annotations["claims_by_case"]):
        raise ValueError("Annotated cases do not match the historical answered cases")

    control = annotations["negative_control"]
    candidate_ids = next(row["candidate_ids"] for row in rows if row["id"] == control["base_case"])
    validate_citations(control["answer"], [{"id": candidate_id} for candidate_id in candidate_ids])
    negative_score = score_claims(control["claims"], {candidate_id: chunks[candidate_id] for candidate_id in candidate_ids})
    if negative_score != 0:
        raise ValueError("Swapped deadlines negative control must have no supported claims")

    claims = [claim for row in rows for claim in row["claims"]]
    return {
        "method": "Manual claim judgments against archived candidate texts. "
                  "The scorer checks quotes and aggregates judgments; it does not infer semantic support.",
        "sources": [*sources, annotation_source],
        "answered_cases": len(rows),
        "scored_claims": len(claims),
        "supported_claims": sum(claim["verdict"] == "supported" for claim in claims),
        "macro_faithfulness": mean(row["faithfulness"] for row in rows),
        "excluded_non_answers": excluded,
        "cases": rows,
        "negative_control": {**control, "citation_check_passed": True, "faithfulness": negative_score},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-run", type=Path, required=True,
                        help="Archived optimizer experiment directory with its three baseline reports.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    reports = {
        "v0-context-precision.json": retrieval_baseline(args.retrieval_run),
        "v0-faithfulness.json": faithfulness_baseline(root, root / "v0_faithfulness_labels.json"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        path = args.output_dir / name
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path.resolve())


if __name__ == "__main__":
    main()
