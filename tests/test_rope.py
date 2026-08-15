import torch

from rope import RotaryEmbedding


def test_rope_matches_reference_split_halves_formula_and_preserves_pass_through_dims() -> None:
    torch.manual_seed(0)
    rope = RotaryEmbedding(rope_dim=8, max_seq_len=16, base=10_000.0)
    x = torch.randn(2, 3, 5, 12)
    positions = torch.tensor([1, 2, 4, 7, 9])

    actual = rope.apply_rotary(x, positions)
    inv_freq = 1.0 / (10_000.0 ** (torch.arange(0, 8, 2).float() / 8))
    angles = torch.outer(positions.float(), inv_freq)
    angles = torch.cat((angles, angles), dim=-1)
    cos = angles.cos().view(1, 1, 5, 8)
    sin = angles.sin().view(1, 1, 5, 8)
    x_rot, x_pass = x[..., :8], x[..., 8:]
    x1, x2 = x_rot.chunk(2, dim=-1)
    expected_rot = x_rot * cos + torch.cat((-x2, x1), dim=-1) * sin
    expected = torch.cat((expected_rot, x_pass), dim=-1)

    torch.testing.assert_close(actual, expected)

