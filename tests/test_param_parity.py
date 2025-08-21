def _try_import():
    try:
        from mop import ViT_Baseline, ViT_MoP
    except Exception:
        from mop.models import ViT_Baseline, ViT_MoP
    return ViT_Baseline, ViT_MoP


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_param_counts_close():
    ViT_Baseline, ViT_MoP = _try_import()
    b = ViT_Baseline(dim=256, depth=6, heads=4, n_classes=10)
    m = ViT_MoP(dim=256, depth=6, heads=4, n_classes=10, n_views=5, n_kernels=3)
    pb, pm = count_params(b), count_params(m)
    assert abs(pb - pm) / max(pb, pm) < 0.02


def test_edgewise_forward_and_params():
    from experiments.cifar10_edgewise_gates import ViTEdgewise  # type: ignore

    b = _try_import()[0](dim=128, depth=2, heads=2, n_classes=10)
    e = ViTEdgewise(dim=128, depth=2, heads=2, n_classes=10, use_k3=True)
    # Forward
    import torch

    x = torch.randn(1, 3, 32, 32)
    ye = e(x)
    assert ye.shape == (1, 10)
    # Param sanity: edgewise should be within a reasonable multiple at tiny config
    pb, pe = count_params(b), count_params(e)
    assert pe <= pb * 4
