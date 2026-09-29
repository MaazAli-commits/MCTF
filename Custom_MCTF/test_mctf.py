import torch
import unittest
from mctf_core import mctf

class TestMCTF(unittest.TestCase):
    def setUp(self):
        self.B, self.H, self.T, self.D = 2, 4, 197, 64
        self.x = torch.randn(self.B, self.T, self.D)
        
        # Simulate an attention map (B, H, T, T) and normalize it as softmax would
        attn_logits = torch.randn(self.B, self.H, self.T, self.T)
        self.attn = torch.softmax(attn_logits, dim=-1)
        
        # Initial token sizes (all 1s)
        self.size = torch.ones(self.B, self.T, 1)

    def test_token_count_decreases(self):
        # 1. Token count actually decreases by the requested r
        r = 16
        x_merged, size_merged, _ = mctf(self.x, self.attn, self.size, r=r)
        
        expected_tokens = self.T - r
        self.assertEqual(x_merged.shape[1], expected_tokens)
        self.assertEqual(size_merged.shape[1], expected_tokens)

    def test_cls_token_never_fused(self):
        # 2. CLS token is never fused
        # Let's mock a very distinct CLS token
        cls_embedding = torch.ones(1, 1, self.D) * 999.0
        x_mock = self.x.clone()
        x_mock[:, 0:1, :] = cls_embedding
        
        r = 16
        x_merged, size_merged, _ = mctf(x_mock, self.attn, self.size, r=r, num_protected=1)
        
        # The first token of the output should still be our distinct CLS token
        cls_out = x_merged[:, 0:1, :]
        self.assertTrue(torch.allclose(cls_out, cls_embedding))
        
        # The size of the protected CLS token should remain 1.0 (no tokens merged into it)
        self.assertEqual(size_merged[0, 0, 0].item(), 1.0)

    def test_token_sizes_update_correctly(self):
        # 3. Token sizes update correctly
        r = 16
        x_merged, size_merged, _ = mctf(self.x, self.attn, self.size, r=r, pooling_type=0)
        
        # If we start with 197 tokens of size 1, the sum of all sizes should still be 197
        # regardless of how many tokens they were merged into.
        total_size_before = self.size.sum().item()
        total_size_after = size_merged.sum().item()
        
        self.assertAlmostEqual(total_size_before, total_size_after, places=4)

    def test_bidirectional_mode(self):
        # 4. Bidirectional and non-bidirectional modes behave correctly
        r = 20
        # Non-bidirectional
        x_uni, size_uni, _ = mctf(self.x, self.attn, self.size, r=r, bidirection=False)
        # Bidirectional
        x_bi, size_bi, _ = mctf(self.x, self.attn, self.size, r=r, bidirection=True)
        
        # Both should successfully reduce exactly r tokens
        self.assertEqual(x_uni.shape[1], self.T - r)
        self.assertEqual(x_bi.shape[1], self.T - r)
        
        # But their outputs should differ because bidirectional merges from both sides
        self.assertFalse(torch.allclose(x_uni, x_bi))

    def test_output_tensor_shapes(self):
        # 5. Output tensor shapes are correct
        r = 10
        x_merged, size_merged, attn_n = mctf(self.x, self.attn, self.size, r=r, one_step_ahead=False)
        
        self.assertEqual(x_merged.shape, (self.B, self.T - r, self.D))
        self.assertEqual(size_merged.shape, (self.B, self.T - r, 1))
        self.assertIsNone(attn_n)

    def test_one_step_ahead_attention(self):
        # 6. One-step-ahead attention has the expected reduced dimensions
        r = 12
        _, _, attn_n = mctf(self.x, self.attn, self.size, r=r, one_step_ahead=True)
        
        expected_T = self.T - r
        self.assertIsNotNone(attn_n)
        # Shape should be reduced on both query and key dimensions (B, H, T-r, T-r)
        self.assertEqual(attn_n.shape, (self.B, self.H, expected_T, expected_T))

if __name__ == '__main__':
    unittest.main()
