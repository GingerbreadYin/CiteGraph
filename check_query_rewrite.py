from unittest import TestCase, main
from unittest.mock import patch

import rag_graph
from check_query_split import assessment, block, split_response


class QueryRewriteChecks(TestCase):
    def run_graph(self, *, enabled=True, checks=("NEED_RETRIEVAL", "ANSWERABLE"),
                  split=False, content='{"query":"退货运费承担"}', length=1):
        ranked = [block(id_) for id_ in "abcd"] + [block("new", length)]
        question = "坏了寄回的钱谁出？"
        with (patch.object(rag_graph, "rank_chunks",
                           side_effect=lambda q, _: ranked if q == question else ranked[-1:]) as rank,
              patch.object(rag_graph, "check_evidence",
                           side_effect=[assessment(s) for s in checks]) as check,
              patch.object(rag_graph, "generate_answer", return_value=("回答[a]", None)) as generate,
              patch("query_split.call_deepseek",
                    return_value=split_response('{"subqueries":["子问题一","子问题二"]}')),
              patch("query_rewrite.call_deepseek", return_value=split_response(content)) as send):
            state = rag_graph.build_rag_graph(
                {}, query_split_enabled=split, query_rewrite_enabled=enabled,
            ).invoke({"question": question})
        return state, rank, check, generate, send

    def test_rewrite_changes_only_retrieval_and_keeps_first_two(self):
        state, rank, check, generate, send = self.run_graph()
        self.assertEqual([c["id"] for c in state["selected_chunks"]], ["a", "b", "new", "c"])
        self.assertEqual([c.args[0] for c in rank.call_args_list], ["坏了寄回的钱谁出？", "退货运费承担"])
        self.assertEqual(check.call_args.args[0], "坏了寄回的钱谁出？")
        self.assertEqual(generate.call_args.args[0], "坏了寄回的钱谁出？")
        self.assertEqual(send.call_args.kwargs["max_attempts"], 1)

    def test_rewrite_is_skipped_for_disabled_or_clarification_or_successful_split(self):
        for kwargs in ({"enabled": False}, {"checks": ("NEED_CLARIFICATION",)}, {"split": True}):
            with self.subTest(**kwargs):
                *_, send = self.run_graph(**kwargs)
                send.assert_not_called()

    def test_unusable_rewrite_keeps_baseline(self):
        for kwargs in ({"content": "bad JSON"}, {"content": '{"query":"坏了寄回的钱谁出"}'},
                       {"length": 1800}):
            with self.subTest(**kwargs):
                state, *_ = self.run_graph(**kwargs)
                self.assertEqual([c["id"] for c in state["selected_chunks"]], list("abcd"))


if __name__ == "__main__":
    main()
