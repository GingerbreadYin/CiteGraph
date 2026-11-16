import json
from unittest import TestCase, main
from unittest.mock import patch

from fastapi.testclient import TestClient

import evidence
import rag_api
import rag_graph


def reply(content, finish_reason="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


def valid(status, question=""):
    return reply(json.dumps({"status": status, "reason": "示例判断", "question": question}))


class EvidenceOutputChecks(TestCase):
    def setUp(self):
        chunks = [
            {"id": "test-001", "source": "test.md", "section": "退货运费", "text": "运费承担取决于损坏原因。"},
            {"id": "test-002", "source": "test.md", "section": "退货", "text": "订单按政策处理。"},
        ]
        self.enterContext(patch.dict("os.environ", {"RAG_QUERY_SPLIT_ENABLED": "0"}))
        self.enterContext(patch.object(rag_api, "load_retriever", return_value={}))
        self.enterContext(patch.object(rag_graph, "rank_chunks", return_value=chunks))
        self.client = self.enterContext(TestClient(rag_api.app, raise_server_exceptions=False))

    def ask(self):
        return self.client.post("/chat", json={"question": "我不知道商品损坏原因，退货运费谁承担？"})

    def test_malformed_provider_output_returns_502_and_logs_reason_without_model_text(self):
        cases = [
            ([], "invalid_envelope"),
            ({"choices": []}, "invalid_envelope"),
            (reply("MODEL_PRIVATE_MARKER"), "invalid_json"),
            (reply("[]"), "not_an_object"),
            (reply('{"status":[],"reason":"x","question":""}'), "invalid_status"),
            (reply('{"status":"ANSWERABLE","reason":" ","question":""}'), "invalid_reason"),
            (reply('{"status":"ANSWERABLE","reason":"x"}'), "invalid_question"),
            (valid("NEED_CLARIFICATION", "  "), "empty_clarification"),
            (reply("{}", "length"), "incomplete_output"),
            (json.JSONDecodeError("Invalid provider JSON", "MODEL_PRIVATE_MARKER", 0), "invalid_response_json"),
        ]
        for response, reason in cases:
            with self.subTest(reason=reason):
                config = {"side_effect": response} if isinstance(response, Exception) else {"return_value": response}
                with patch.object(evidence, "call_deepseek", **config) as send:
                    with self.assertLogs("rag_graph", level="WARNING") as logs:
                        result = self.ask()
                    send.assert_called_once()
                self.assertEqual(result.status_code, 502)
                self.assertEqual(result.json()["detail"]["code"], "INVALID_EVIDENCE_OUTPUT")
                self.assertIn(f"round=1 reason={reason}", logs.output[0])
                self.assertNotIn("MODEL_PRIVATE_MARKER", result.text + "\n".join(logs.output))

    def test_second_round_failure_logs_round_two_without_generating_an_answer(self):
        with (patch.object(evidence, "call_deepseek",
                           side_effect=[valid("NEED_RETRIEVAL"), reply("not JSON")]) as send,
              patch.object(rag_graph, "generate_answer") as generate,
              self.assertLogs("rag_graph", level="WARNING") as logs):
            result = self.ask()
        self.assertEqual(result.status_code, 502)
        self.assertIn("round=2 reason=invalid_json", logs.output[0])
        self.assertEqual(send.call_count, 2)
        generate.assert_not_called()

    def test_valid_clarification_is_returned_as_before(self):
        with patch.object(evidence, "call_deepseek", return_value=valid("NEED_CLARIFICATION", "请问损坏原因是什么？")):
            result = self.ask()
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "NEED_CLARIFICATION")
        self.assertEqual(result.json()["answer"], "请问损坏原因是什么？")


if __name__ == "__main__":
    main()
