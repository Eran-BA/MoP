import torch


def _try_import():
    try:
        from mop import ViT_Baseline, ViT_MoP
    except Exception:
        from mop.models import ViT_Baseline, ViT_MoP
    return ViT_Baseline, ViT_MoP


def test_vit_shapes():
    ViT_Baseline, ViT_MoP = _try_import()
    x = torch.randn(2, 3, 32, 32)
    b = ViT_Baseline(dim=256, depth=2, heads=2, n_classes=10)
    m = ViT_MoP(dim=256, depth=2, heads=2, n_classes=10, n_views=2, n_kernels=1)
    yb = b(x)
    ym = m(x)
    assert yb.shape == (2, 10)
    assert ym.shape == (2, 10)


def test_gate_api():
    ViT_Baseline, ViT_MoP = _try_import()
    x = torch.randn(2, 3, 32, 32)
    m = ViT_MoP(dim=256, depth=2, heads=2, n_classes=10, n_views=2, n_kernels=1)
    gates, views, kernels = m.get_gate_maps(x)
    assert gates.ndim == 4 and gates.shape[1] == 1


def test_edgewise_shapes_cifar10():
    from experiments.cifar10_edgewise_gates import ViTEdgewise  # type: ignore

    x = torch.randn(2, 3, 32, 32)
    e = ViTEdgewise(dim=128, depth=2, heads=2, n_classes=10, use_k3=True)
    ye = e(x)
    assert ye.shape == (2, 10)
