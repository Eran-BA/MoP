import torch
import pytest

from mop.models.attention_variants import EdgewiseMSA


@pytest.mark.parametrize("use_lens_bank,use_lens_bank_qk,lens_dilations,lens_qk_dilations,n_views", [
    (True, False, (1, 2), (1, 2), 3),
    (False, True, (1,), (1, 2, 3), 3),
    (True, True, (1, 2), (2, 3), 4),
    (False, False, (1,), (1,), 3),
])
def test_edgewise_lens_bank_shapes(use_lens_bank, use_lens_bank_qk, lens_dilations, lens_qk_dilations, n_views):
    torch.manual_seed(0)
    dim = 64
    heads = 4
    N = 8
    B = 2
    x = torch.randn(B, N, dim)

    msa = EdgewiseMSA(
        dim=dim,
        heads=heads,
        n_views=n_views,
        share_qkv=True,
        gate_mode="lowrank",
        gate_rank=2,
        gate_init="neutral",
        use_k3=True,
        use_lens_bank=use_lens_bank,
        lens_kernel_size=3,
        lens_dilations=lens_dilations,
        use_lens_bank_qk=use_lens_bank_qk,
        lens_qk_kernel_size=3,
        lens_qk_dilations=lens_qk_dilations,
        lens_qk_causal=True,
    )

    y = msa(x)
    assert y.shape == (B, N, dim)


def test_edgewise_lens_bank_effect_nontrivial():
    torch.manual_seed(0)
    dim = 32
    heads = 2
    N = 6
    B = 1
    x = torch.randn(B, N, dim)

    msa_no = EdgewiseMSA(
        dim=dim, heads=heads, n_views=3, share_qkv=True, use_lens_bank=False
    )
    y_no = msa_no(x)

    msa_yes = EdgewiseMSA(
        dim=dim,
        heads=heads,
        n_views=3,
        share_qkv=True,
        use_lens_bank_qk=True,
        lens_qk_kernel_size=3,
        lens_qk_dilations=(1, 2),
    )
    y_yes = msa_yes(x)

    # The lens bank path should alter outputs (not guaranteed strictly, but highly likely)
    assert not torch.allclose(y_no, y_yes), "Lens bank had no effect on outputs"


