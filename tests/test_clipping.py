import torch

from residual_clipping.clipping import clip_tensor, clip_tensor_list, tensor_list_global_norm


def test_clip_tensor_reduces_norm_to_threshold():
    clipped, norm, scale = clip_tensor(torch.tensor([3.0, 4.0]), threshold=1.0)
    assert norm == 5.0
    assert abs(scale - 0.2) < 1e-12
    assert torch.allclose(clipped, torch.tensor([0.6, 0.8]))


def test_clip_tensor_list_uses_shared_norm():
    tensors = [torch.tensor([3.0, 4.0]), torch.tensor([0.0, 12.0])]
    clipped, norm, scale = clip_tensor_list(tensors, threshold=2.0)
    assert abs(norm - 13.0) < 1e-12
    assert abs(scale - (2.0 / 13.0)) < 1e-12
    assert torch.allclose(tensor_list_global_norm(clipped), torch.tensor(2.0))
