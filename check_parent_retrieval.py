from unittest import TestCase, main
from unittest.mock import Mock
import torch
import retrieval


class ParentRetrievalChecks(TestCase):
    def test_child_hits_map_to_unique_parents_with_original_ids_and_text(self):
        parents = [
            {"id": "a", "section": "运费", "text": "## 运费\n质量问题商家承担。损坏原因不明时应核实。"},
            {"id": "b", "section": "开票", "text": "## 开票\n订单完成后可申请。"},
        ]
        model = Mock()
        model.encode.side_effect = [torch.tensor([[1., 0.], [.95, .05], [0., 1.]]),
                                    torch.tensor([1., 0.])]
        resources = retrieval.with_parent_child_index({"chunks": parents, "model": model})
        ranked = retrieval.rank_chunks("损坏原因未查明", resources)
        self.assertEqual(ranked, parents)
        self.assertEqual(len(resources["search_chunks"]), 3)
        self.assertIn("损坏原因不明时应核实。", model.encode.call_args_list[0].args[0][1])


if __name__ == "__main__":
    main()
