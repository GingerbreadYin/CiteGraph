from unittest import TestCase, main
from unittest.mock import Mock

from hybrid_retrieval import fuse_rankings, rank_hybrid, with_hybrid_search


class HybridRetrievalChecks(TestCase):
    def test_rrf_promotes_shared_hit_and_keeps_vector_tie_order(self):
        self.assertEqual(fuse_rankings([0, 1, 2], [3, 1, 4]), [1, 0, 3, 2, 4])

    def test_bm25_and_reranker_share_original_policy_ids(self):
        chunks = [{"id": str(i), "text": text} for i, text in enumerate(
            ("退货 运费", "退款 到账", "发票 申请", "定制 商品", "修改 发票", "个人 原因", "企业 税号"))]
        resources = with_hybrid_search({"chunks": chunks}, mode="hybrid")
        dense = list(range(len(chunks)))
        hybrid = rank_hybrid("企业 税号", dense, resources)
        self.assertEqual(hybrid[0], 0)
        self.assertIn(6, hybrid[:6])

        reranker = Mock()
        reranker.predict.return_value = [int(text == chunks[6]["text"]) for _, text in
                                         [("企业 税号", chunks[i]["text"]) for i in hybrid[:6]]]
        result = rank_hybrid("企业 税号", dense, {**resources, "reranker": reranker})
        self.assertEqual(result[0], 6)
        self.assertEqual(set(result), set(dense))
        self.assertEqual(result[6:], hybrid[6:])
        reranker.predict.assert_called_once_with([("企业 税号", chunks[i]["text"]) for i in hybrid[:6]])

    def test_empty_lexical_query_leaves_vector_order(self):
        chunks = [{"id": "a", "text": "退货"}, {"id": "b", "text": "发票"}]
        resources = with_hybrid_search({"chunks": chunks}, mode="hybrid")
        self.assertEqual(rank_hybrid("！！！", [1, 0], resources), [1, 0])


if __name__ == "__main__":
    main()
