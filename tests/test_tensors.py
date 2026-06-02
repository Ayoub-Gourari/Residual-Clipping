import torch

from residual_clipping.tensors import flatten_tensors, unflatten_like


def test_flatten_and_unflatten_round_trip():
    references = [torch.ones(2, 2), torch.ones(3)]
    flat = flatten_tensors(references)
    restored = unflatten_like(flat, references)
    assert len(restored) == len(references)
    assert all(item.shape == ref.shape for item, ref in zip(restored, references))
    assert torch.equal(flatten_tensors(restored), flat)
