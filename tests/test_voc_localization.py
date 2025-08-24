import torch


def test_vit_localizer_forward_and_iou():
    from experiments.voc_localization_vit import ViTLocalizer, bbox_iou

    # Tiny config for speed
    model = ViTLocalizer(
        dim=64, depth=2, heads=4, mlp_ratio=2.0, drop_path=0.0, patch=16, img_size=32
    )
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 4)
    # values are normalized to [0,1]
    assert torch.all((y >= 0) & (y <= 1))

    # IoU(y, y) should be 1 (approx within tolerance)
    iou = bbox_iou(y, y)
    assert torch.isfinite(iou).all()
    assert (iou >= 0).all() and (iou <= 1).all()
    assert torch.allclose(iou, torch.ones_like(iou), atol=1e-6)


