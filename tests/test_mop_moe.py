import torch

def test_vit_mop_moe_forward():
    from mop.models import ViT_MoP
    x = torch.randn(2,3,32,32)
    m = ViT_MoP(dim=64, depth=2, heads=4, n_classes=10, n_views=3, n_kernels=2,
                use_moe=True, moe_experts=3)
    y = m(x)
    assert y.shape == (2,10)

