import torch

from residual_clipping.cifar10_pipeline import MomentumClipper


def make_parameter(*values: float) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.allclose(actual, expected, atol=1e-6), f"actual={actual}, expected={expected}"


def test_sgd_momentum_uses_ema_update():
    param = make_parameter(0.0, 0.0)
    optimizer = MomentumClipper([param], mode="sgd_momentum", beta=0.9, clip_c=None, clip_c_res=None)

    diagnostics = optimizer.step([torch.tensor([2.0, 4.0])], lr=0.5)

    expected_momentum = torch.tensor([0.2, 0.4])
    expected_param = torch.tensor([-0.1, -0.2])
    assert_close(optimizer.momentum[0], expected_momentum)
    assert_close(param.detach(), expected_param)
    assert diagnostics["grad_norm"] > 0
    assert abs(diagnostics["next_momentum_norm"] - torch.linalg.vector_norm(expected_momentum).item()) < 1e-6


def test_standard_clipping_applies_clipped_gradient_inside_ema():
    param = make_parameter(0.0, 0.0)
    optimizer = MomentumClipper([param], mode="clipped_momentum", beta=0.9, clip_c=1.0, clip_c_res=None)

    diagnostics = optimizer.step([torch.tensor([3.0, 4.0])], lr=1.0)

    expected_clipped_grad = torch.tensor([0.6, 0.8])
    expected_momentum = 0.1 * expected_clipped_grad
    expected_param = -expected_momentum
    assert_close(optimizer.momentum[0], expected_momentum)
    assert_close(param.detach(), expected_param)
    assert abs(diagnostics["clip_fraction"] - 0.2) < 1e-6
    assert diagnostics["clip_active"] == 1.0
    assert abs(diagnostics["standard_clipped_grad_norm"] - 1.0) < 1e-6


def test_residual_clipping_uses_previous_momentum_as_center():
    param = make_parameter(0.0, 0.0)
    optimizer = MomentumClipper(
        [param],
        mode="residual_clipped_momentum",
        beta=0.9,
        clip_c=None,
        clip_c_res=2.0,
    )
    optimizer.momentum = [torch.tensor([1.0, 0.0])]

    diagnostics = optimizer.step([torch.tensor([4.0, 0.0])], lr=1.0)

    expected_clipped_residual = torch.tensor([2.0, 0.0])
    expected_momentum = torch.tensor([1.2, 0.0])
    expected_param = -expected_momentum
    assert_close(optimizer.momentum[0], expected_momentum)
    assert_close(param.detach(), expected_param)
    assert abs(diagnostics["residual_clip_fraction"] - (2.0 / 3.0)) < 1e-6
    assert diagnostics["clip_active"] == 1.0
    assert abs(diagnostics["cos_grad_center"] - 1.0) < 1e-6
    assert abs(diagnostics["residual_clipped_residual_norm"] - torch.linalg.vector_norm(expected_clipped_residual).item()) < 1e-6
