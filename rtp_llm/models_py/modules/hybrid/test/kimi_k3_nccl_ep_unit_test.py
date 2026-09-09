import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from rtp_llm.models_py.distributed.collective_torch import (
    _requires_dedicated_ep_group,
)
from rtp_llm.models_py.distributed.sequence_parallel import (
    shard_tokens,
    token_shard_layout,
)
from rtp_llm.models_py.modules.kimi_k3.moe import KimiK3LatentMoE, _requires_nccl_ep
from rtp_llm.models_py.triton_kernels.common.activation import (
    situ_and_mul,
    situ_mul_fp8_quant_packed_masked,
)


class KimiK3NcclEpTopologyTest(unittest.TestCase):
    def test_single_host_keeps_mega_moe(self) -> None:
        self.assertFalse(_requires_nccl_ep(8, 8))
        self.assertFalse(_requires_nccl_ep(16, 16))

    def test_multi_host_uses_nccl_ep(self) -> None:
        self.assertTrue(_requires_nccl_ep(16, 8))
        self.assertTrue(_requires_nccl_ep(32, 8))

    def test_invalid_topology_is_rejected(self) -> None:
        for world_size, local_world_size in ((0, 8), (8, 0), (10, 8)):
            with self.subTest(
                world_size=world_size,
                local_world_size=local_world_size,
            ):
                with self.assertRaises(ValueError):
                    _requires_nccl_ep(world_size, local_world_size)

    def test_multi_host_full_world_ep_gets_dedicated_group(self) -> None:
        self.assertTrue(
            _requires_dedicated_ep_group(
                world_size=16,
                local_world_size=8,
                ep_size=16,
            )
        )
        self.assertTrue(
            _requires_dedicated_ep_group(
                world_size=32,
                local_world_size=8,
                ep_size=32,
            )
        )

    def test_single_host_or_partial_ep_does_not_get_dedicated_group(self) -> None:
        self.assertFalse(
            _requires_dedicated_ep_group(
                world_size=8,
                local_world_size=8,
                ep_size=8,
            )
        )
        self.assertFalse(
            _requires_dedicated_ep_group(
                world_size=16,
                local_world_size=8,
                ep_size=8,
            )
        )

    def test_invalid_ep_group_topology_is_rejected(self) -> None:
        for world_size, local_world_size, ep_size in (
            (0, 8, 16),
            (16, 0, 16),
            (10, 8, 10),
        ):
            with self.subTest(
                world_size=world_size,
                local_world_size=local_world_size,
                ep_size=ep_size,
            ):
                with self.assertRaises(ValueError):
                    _requires_dedicated_ep_group(
                        world_size=world_size,
                        local_world_size=local_world_size,
                        ep_size=ep_size,
                    )

    def test_nccl_ep_accepts_padded_prefill_token_shards(self) -> None:
        for tp_size in (8, 16):
            for token_count in (1, 17, 33):
                full_input = (
                    torch.arange(token_count * 4).reshape(token_count, 4).float()
                )
                outputs = []
                for rank in range(tp_size):
                    with self.subTest(tp_size=tp_size, tokens=token_count, rank=rank):
                        layout = token_shard_layout(token_count, tp_size, rank)
                        local_input = shard_tokens(full_input, layout)
                        ids = torch.zeros((layout.local_tokens, 2), dtype=torch.int64)
                        weights = torch.ones_like(ids, dtype=torch.float32)
                        weights[layout.local_valid_tokens :] = 0
                        expected = local_input * 2
                        expert_sum = Mock(return_value=expected)
                        module = SimpleNamespace(
                            _use_nccl_ep=True, _nccl_ep_expert_sum=expert_sum
                        )
                        output = KimiK3LatentMoE._mega_expert_sum(
                            module, local_input, ids, weights, sequence_parallel=True
                        )
                        expert_sum.assert_called_once_with(local_input, ids, weights)
                        self.assertIs(output, expected)
                        outputs.append(output[: layout.local_valid_tokens])
                torch.testing.assert_close(torch.cat(outputs), full_input * 2)

    def test_nccl_ep_prefill_bounds_workspace_and_preserves_row_order(self) -> None:
        for token_count in (64, 65, 129):
            with self.subTest(tokens=token_count):
                inputs = torch.arange(token_count * 4).reshape(token_count, 4).float()
                ids = torch.arange(token_count * 2).reshape(token_count, 2)
                weights = torch.ones((token_count, 2))
                weights[-1] = 0
                expert_sum = Mock(side_effect=lambda x, ids, weights: x * 2)
                module = SimpleNamespace(
                    _use_nccl_ep=True, _nccl_ep_expert_sum=expert_sum
                )
                output = KimiK3LatentMoE._mega_expert_sum(
                    module, inputs, ids, weights, sequence_parallel=True
                )
                torch.testing.assert_close(output, inputs * 2)
                calls = expert_sum.call_args_list
                self.assertTrue(all(call.args[0].shape[0] <= 64 for call in calls))
                for column, expected in enumerate((inputs, ids, weights)):
                    torch.testing.assert_close(
                        torch.cat([call.args[column] for call in calls]), expected
                    )

    def test_nccl_ep_keeps_decode_owner_rows(self) -> None:
        inputs = torch.randn(4, 8)
        ids = torch.zeros((4, 2), dtype=torch.int64)
        weights = torch.ones((4, 2))
        expert_sum = Mock(return_value=inputs)
        module = SimpleNamespace(_use_nccl_ep=True, _nccl_ep_expert_sum=expert_sum)
        output = KimiK3LatentMoE._mega_expert_sum(
            module, inputs, ids, weights, sequence_parallel=False
        )
        expert_sum.assert_called_once_with(inputs, ids, weights)
        self.assertIs(output, inputs)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_masked_situ_packed_quant_matches_reference(self) -> None:
        torch.manual_seed(7)
        experts, capacity, hidden = 2, 8, 128
        gate_up = torch.randn(
            experts,
            capacity,
            2 * hidden,
            dtype=torch.bfloat16,
            device="cuda",
        )
        counts = torch.tensor([3, 5], dtype=torch.int32, device="cuda")
        quantized, packed_scale = situ_mul_fp8_quant_packed_masked(
            gate_up,
            counts,
            beta=1.7,
            linear_beta=2.3,
        )
        self.assertEqual(tuple(quantized.shape), (experts, capacity, hidden))
        self.assertEqual(tuple(packed_scale.shape), (experts, capacity, 1))
        for expert, count in enumerate((3, 5)):
            reference = situ_and_mul(
                gate_up[expert, :count, :hidden].contiguous(),
                gate_up[expert, :count, hidden:].contiguous(),
                beta=1.7,
                linear_beta=2.3,
            ).float()
            exponent = packed_scale[expert, :count, 0].bitwise_and(0xFF).float() - 127.0
            dequantized = quantized[expert, :count].float() * torch.exp2(
                exponent
            ).unsqueeze(-1)
            torch.testing.assert_close(
                dequantized,
                reference,
                rtol=0.08,
                atol=0.08,
            )


if __name__ == "__main__":
    unittest.main()
