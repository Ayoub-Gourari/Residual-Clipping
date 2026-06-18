import torch

from residual_clipping.adaptive_optimizers import AdaptiveAdamW


def test_adamw_first_step_matches_bias_corrected_update_with_decoupled_decay():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
    optimizer = AdaptiveAdamW(
        [param],
        mode="adamw",
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
    )

    diagnostics = optimizer.step([torch.tensor([0.5, -0.25])], lr=0.1)

    expected = torch.tensor([0.899, 2.098])
    assert torch.allclose(param.detach(), expected, atol=1e-6)
    assert optimizer.step_count == 1
    assert diagnostics["grad_norm"] > 0.0
    assert abs(diagnostics["adam_step_norm"] - 0.1 * diagnostics["adam_update_norm"]) < 1e-6


def test_adamw_rejects_unknown_mode():
    param = torch.nn.Parameter(torch.tensor([1.0]))

    try:
        AdaptiveAdamW([param], mode="future_clipped_adamw", beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)
    except ValueError as exc:
        assert "Unsupported adaptive optimizer mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown adaptive optimizer mode.")
