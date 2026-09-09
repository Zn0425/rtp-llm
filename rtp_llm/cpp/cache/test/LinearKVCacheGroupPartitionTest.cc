#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/LinearKVCacheGroup.h"

namespace rtp_llm {
namespace test {

class LinearKVCacheGroupPartitionTest: public ::testing::TestWithParam<int> {};

TEST_P(LinearKVCacheGroupPartitionTest, ReconstructsKdaStateFromPrefillShards) {
    auto spec                = std::make_shared<LinearKVCacheSpec>();
    spec->type               = KVCacheSpecType::LinearAttention;
    spec->dtype              = DataType::TYPE_FP16;
    spec->layer_num          = 1;
    spec->local_num_k_heads  = 96;
    spec->local_num_v_heads  = 96;
    spec->local_head_num_kv  = 96;
    spec->head_k_dim         = 2;
    spec->head_v_dim         = 2;
    spec->conv_kernel_dim    = 3;
    spec->ssm_state_dtype    = DataType::TYPE_FP32;
    spec->conv_state_dtype   = DataType::TYPE_FP16;
    spec->seq_size_per_block = 4;

    const size_t block_stride = spec->block_size_bytes();
    auto         pool_config  = BlockPoolConfigHelper::createConfig(
        /*layer_num=*/1, /*block_num=*/2, block_stride, DataType::TYPE_FP16);
    auto block_pool = std::make_shared<BlockPool>(pool_config, AllocationType::HOST);
    ASSERT_TRUE(block_pool->init());

    LinearKVCacheGroup group(/*layer_ids=*/{0}, spec, block_pool, /*group_id=*/0, /*linear_step=*/2);
    ASSERT_TRUE(group.init());
    auto allocated = block_pool->malloc(1);
    ASSERT_EQ(allocated.size(), 1u);

    auto whole = block_pool->convertIndexToBuffer(/*layer_id=*/0, allocated[0]);
    ASSERT_EQ(whole.size(), 1u);
    auto* base = static_cast<char*>(whole[0].addr);

    const int    kPartitions = GetParam();
    const size_t ssm_bytes   = spec->k_block_size_bytes();
    const size_t q_bytes =
        static_cast<size_t>(spec->local_num_k_heads) * spec->head_k_dim * getTypeSize(spec->conv_state_dtype);
    const size_t k_bytes = q_bytes;
    const size_t v_bytes =
        static_cast<size_t>(spec->local_num_v_heads) * spec->head_v_dim * getTypeSize(spec->conv_state_dtype);
    const size_t history_stride = q_bytes + k_bytes + v_bytes;

    // Prefill owns 96 / TP heads; Decode owns all 96. Transfer the actual
    // source segments into the destination slices, including every Q/K/V history.
    auto source_spec = std::make_shared<LinearKVCacheSpec>(*spec);
    source_spec->local_num_k_heads /= kPartitions;
    source_spec->local_num_v_heads /= kPartitions;
    source_spec->local_head_num_kv /= kPartitions;
    auto source_config =
        BlockPoolConfigHelper::createConfig(1, 2, source_spec->block_size_bytes(), DataType::TYPE_FP16);
    auto source_pool = std::make_shared<BlockPool>(source_config, AllocationType::HOST);
    ASSERT_TRUE(source_pool->init());
    LinearKVCacheGroup source_group({0}, source_spec, source_pool, 0, 2);
    ASSERT_TRUE(source_group.init());
    auto source_allocated = source_pool->malloc(1);
    ASSERT_EQ(source_allocated.size(), 1u);
    auto source_parts         = source_group.convertIndexToBuffer(0, source_allocated[0], 1, 0);
    auto destination_segments = group.convertIndexToBuffer(0, allocated[0], 1, 0);
    ASSERT_EQ(source_parts.size(), destination_segments.size());
    std::memset(base, 0, block_stride);

    for (int partition = 0; partition < kPartitions; ++partition) {
        auto parts = group.convertIndexToBuffer(
            /*layer_id=*/0, allocated[0], kPartitions, partition);
        ASSERT_EQ(parts.size(), 7u);  // SSM + two histories * Q/K/V.
        for (size_t segment = 0; segment < parts.size(); ++segment) {
            ASSERT_EQ(source_parts[segment].size_bytes, parts[segment].size_bytes);
            const int marker = 1 + partition + segment * kPartitions;
            std::memset(source_parts[segment].addr, marker, source_parts[segment].size_bytes);
            std::memcpy(parts[segment].addr, source_parts[segment].addr, parts[segment].size_bytes);
        }
        EXPECT_EQ(parts[0].size_bytes, ssm_bytes / kPartitions);
        EXPECT_EQ(static_cast<char*>(parts[0].addr) - base, partition * static_cast<int64_t>(ssm_bytes / kPartitions));
        for (size_t history = 0; history < 2; ++history) {
            const size_t segment      = 1 + history * 3;
            const size_t history_base = ssm_bytes + history * history_stride;
            EXPECT_EQ(static_cast<char*>(parts[segment].addr) - base,
                      static_cast<int64_t>(history_base + partition * (q_bytes / kPartitions)));
            EXPECT_EQ(static_cast<char*>(parts[segment + 1].addr) - base,
                      static_cast<int64_t>(history_base + q_bytes + partition * (k_bytes / kPartitions)));
            EXPECT_EQ(static_cast<char*>(parts[segment + 2].addr) - base,
                      static_cast<int64_t>(history_base + q_bytes + k_bytes + partition * (v_bytes / kPartitions)));
        }
    }
    for (size_t segment = 0; segment < destination_segments.size(); ++segment) {
        const auto&  whole_segment = destination_segments[segment];
        const auto*  bytes         = static_cast<const unsigned char*>(whole_segment.addr);
        const size_t peer_bytes    = source_parts[segment].size_bytes;
        ASSERT_EQ(whole_segment.size_bytes, peer_bytes * kPartitions);
        for (size_t offset = 0; offset < whole_segment.size_bytes; ++offset) {
            ASSERT_EQ(bytes[offset], 1 + offset / peer_bytes + segment * kPartitions)
                << "segment=" << segment << " offset=" << offset;
        }
    }
    source_pool->requestFree(source_allocated);
    block_pool->requestFree(allocated);
}

INSTANTIATE_TEST_SUITE_P(PrefillTP, LinearKVCacheGroupPartitionTest, ::testing::Values(8, 16));

}  // namespace test
}  // namespace rtp_llm
