import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from citations import validate_citations
from evaluate_v0 import context_precision, faithfulness_baseline, score_claims


class ContextPrecisionTests(unittest.TestCase):
    def test_same_hits_in_different_order_have_different_scores(self):
        evidence = {"id": "hit", "evidence_ids": ["refund"]}
        noise = {"id": "noise"}
        required = [["refund"]]
        self.assertEqual(context_precision(required, [evidence, noise], 2), 1.0)
        self.assertEqual(context_precision(required, [noise, evidence], 2), 0.5)

    def test_partial_spans_count_for_relevance_but_look_only_at_top_k(self):
        partial = {"id": "p", "evidence_spans": [{"id": "e"}]}
        self.assertEqual(context_precision([["e"]], [{"id": "x"}, partial], 1), 0.0)
        self.assertEqual(context_precision([["e"]], [{"id": "x"}, partial], 2), 0.5)

    def test_no_evidence_labels_is_not_applicable(self):
        self.assertIsNone(context_precision([], [{"id": "x"}], 2))
        self.assertEqual(context_precision([["e"]], [], 2), 0.0)


class FaithfulnessTests(unittest.TestCase):
    def test_valid_citations_do_not_make_swapped_deadlines_supported(self):
        chunks = {
            "refund": "Refund in one business day.",
            "invoice": "Invoice in seven business days.",
        }
        answer = "Refund in seven business days.[refund] Invoice in one business day.[invoice]"
        validate_citations(answer, [{"id": id_} for id_ in chunks])
        claims = [
            {"claim": "Refund in seven business days.", "verdict": "unsupported",
             "evidence": [{"id": "refund", "quote": chunks["refund"]}]},
            {"claim": "Invoice in one business day.", "verdict": "unsupported",
             "evidence": [{"id": "invoice", "quote": chunks["invoice"]}]},
        ]
        self.assertEqual(score_claims(claims, chunks), 0.0)
        claims[0]["verdict"] = "supported"
        self.assertEqual(score_claims(claims, chunks), 0.5)

    def test_claim_with_no_related_evidence_counts_as_unsupported(self):
        claim = {"claim": "Shipping costs are capped at 10.", "verdict": "unsupported",
                 "evidence": []}
        self.assertEqual(score_claims([claim], {"x": "Check your invoice."}), 0.0)

    def test_evidence_must_be_supplied_and_quote_must_match(self):
        claims = [{"claim": "Refund in one day.", "verdict": "supported",
                   "evidence": [{"id": "refund", "quote": "one day"}]}]
        with self.assertRaises(ValueError):
            score_claims(claims, {"refund": "Refund in three days."})
        with self.assertRaises(ValueError):
            score_claims(claims, {"other": "Refund in one day."})


class SnapshotTests(unittest.TestCase):
    def test_rejects_reusing_annotations_on_changed_chunks(self):
        annotations = Path(__file__).with_name("v0_faithfulness_labels.json")
        with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            (root / "chunks.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Historical input changed.*chunks.json"):
                faithfulness_baseline(root, annotations)


if __name__ == "__main__":
    unittest.main()
