from contextlib import ExitStack
from http.client import IncompleteRead
from unittest import TestCase, main
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import model_client
import rag_graph
from query_split import expand_with_query_split


def block(id_, length=1):
    return {"id": id_, "source": "policy.md", "section": id_, "text": "x" * length}


def assessment(status):
    return {"status": status, "reason": "test", "question": "请说明原因。"}


def split_response(content, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": content}}]}


class QuerySplitChecks(TestCase):
    def run_case(self, enabled, checks, response, base=None):
        base = base or [block(id_) for id_ in ("a", "b", "c", "d", "new")]
        children = {"子问题一": [base[0], base[-1]], "子问题二": [base[1], base[-1]]}
        with ExitStack() as stack:
            rank = stack.enter_context(patch.object(
                rag_graph, "rank_chunks", side_effect=lambda q, _: children.get(q, base)))
            check = stack.enter_context(patch.object(
                rag_graph, "check_evidence", side_effect=[assessment(s) for s in checks]))
            stack.enter_context(patch.object(
                rag_graph, "generate_answer", return_value=("结果[a]", None)))
            kwargs = {"side_effect": response} if isinstance(response, Exception) else {"return_value": response}
            send = stack.enter_context(patch("query_split.call_deepseek", **kwargs))
            graph = (rag_graph.build_rag_graph({}) if enabled is None else
                     rag_graph.build_rag_graph({}, query_split_enabled=enabled))
            result = graph.invoke({"question": "物流核查和退款怎么办？"})
        return result, rank, check, send

    def test_default_uses_original_expand_without_calling_split(self):
        result, rank, _, send = self.run_case(None, ["NEED_RETRIEVAL", "ANSWERABLE"], None)
        send.assert_not_called()
        self.assertEqual(rank.call_count, 1)
        self.assertEqual([c["id"] for c in result["selected_chunks"]], ["a", "b", "c", "d"])

    def test_enabled_direct_answer_and_clarification_do_not_split(self):
        for decision, expected in [("ANSWERABLE", "ANSWERED"), ("NEED_CLARIFICATION", "NEED_CLARIFICATION")]:
            with self.subTest(decision=decision):
                result, rank, check, send = self.run_case(True, [decision], None)
                self.assertEqual(result["status"], expected)
                self.assertEqual((rank.call_count, check.call_count), (1, 1))
                send.assert_not_called()

    def test_split_preserves_first_round_and_rechecks_once(self):
        response = split_response('{"subqueries":["子问题一","子问题二"]}')
        result, rank, check, send = self.run_case(True, ["NEED_RETRIEVAL", "NEED_RETRIEVAL"], response)
        self.assertEqual(result["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["rounds"], 2)
        self.assertEqual([c["id"] for c in result["selected_chunks"]], ["a", "b", "new", "c"])
        self.assertIn("[new]", check.call_args.args[1])
        self.assertEqual([args.args[0] for args in rank.call_args_list],
                         ["物流核查和退款怎么办？", "子问题一", "子问题二"])
        send.assert_called_once()
        self.assertEqual(send.call_args.kwargs["max_attempts"], 1)
        self.assertEqual(send.call_args.args[0]["messages"][1]["content"], "物流核查和退款怎么办？")

    def test_invalid_empty_duplicate_and_failed_splits_use_original_expand(self):
        for response in (split_response("bad JSON"), split_response('{"subqueries":[]}'),
                         split_response('{"subqueries":["相同问题","相同问题？"]}'),
                         split_response('{"subqueries":["子问题一","子问题二"]}', "length"),
                         model_client.ModelRequestError("offline simulation"), IncompleteRead(b"partial")):
            with self.subTest(response=response):
                result, rank, _, send = self.run_case(True, ["NEED_RETRIEVAL", "ANSWERABLE"], response)
                self.assertEqual([c["id"] for c in result["selected_chunks"]], ["a", "b", "c", "d"])
                self.assertEqual(rank.call_count, 1)
                send.assert_called_once()

    def test_budget_keeps_first_two_and_skips_oversized_split_chunk(self):
        base = [block("a", 850), block("b", 900), block("c", 20), block("d", 30), block("new", 51)]
        response = split_response('{"subqueries":["子问题一","子问题二"]}')
        result, _, _, _ = self.run_case(True, ["NEED_RETRIEVAL", "ANSWERABLE"], response, base)
        self.assertEqual([c["id"] for c in result["selected_chunks"]], ["a", "b", "c", "d"])

    def test_failed_local_subquery_retrieval_keeps_original_expand(self):
        original = [block(id_) for id_ in ("a", "b", "c", "d", "new")]
        retrieve = Mock(side_effect=RuntimeError("simulated encoder failure"))
        response = split_response('{"subqueries":["子问题一","子问题二"]}')
        with patch("query_split.call_deepseek", return_value=response):
            self.assertEqual(expand_with_query_split("复合问题", original, retrieve), original[:4])
        retrieve.assert_called_once_with("子问题一")

    def test_single_attempt_does_not_retry_transient_http_error(self):
        error = HTTPError("https://api.deepseek.com/chat/completions", 503, "test", None, None)
        with (patch.object(model_client, "load_dotenv"),
              patch.dict("os.environ", {"DEEPSEEK_API_KEY": "mock-key"}),
              patch.object(model_client, "urlopen", side_effect=error) as send,
              patch.object(model_client.time, "sleep") as sleep):
            with self.assertRaises(model_client.ModelRequestError):
                model_client.call_deepseek({}, max_attempts=1)
            send.assert_called_once()
            sleep.assert_not_called()


if __name__ == "__main__":
    main()
