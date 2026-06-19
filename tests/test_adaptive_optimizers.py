import torch

from residual_clipping.adaptive_optimizers import AdaptiveAdamW


def test_adamw_first_step_matches_bias_corrected_update_with_decoupled_decay():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
    optimizer = AdaptiveAdamW(
        [param],
        optimizer_name="adamw_uncut",
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
        clip_threshold=float("inf"),
        clipping_scope="global",
        correct_bias=True,
    )

    diagnostics = optimizer.step([torch.tensor([0.5, -0.25])], lr=0.1)

    expected = torch.tensor([0.899, 2.098])
    assert torch.allclose(param.detach(), expected, atol=1e-6)
    assert optimizer.step_count == 1
    assert diagnostics["grad_global_norm"] > 0.0
    assert abs(diagnostics["adam_step_norm"] - 0.1 * diagnostics["adam_update_norm"]) < 1e-6


def test_adamw_rejects_unknown_mode():
    param = torch.nn.Parameter(torch.tensor([1.0]))

    try:
        AdaptiveAdamW(
            [param],
            optimizer_name="future_clipped_adamw",
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
            clipping_scope="local",
        )
    except ValueError as exc:
        assert "Unsupported optimizer_name" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown adaptive optimizer mode.")


def test_all_adamw_optimizer_names_produce_requested_diagnostics_without_nans():
    names = [
        "adamw_uncut",
        "adamw_clip",
        "adamw_resclip_euclidean",
        "adamw_resclip_euclidean_vclip",
        "adamw_resclip_metric",
    ]
    required_keys = {
        "optimizer_name",
        "clip_threshold",
        "grad_global_norm",
        "pseudo_grad_global_norm",
        "clipping_scale",
        "residual_global_norm",
        "metric_residual_global_norm",
        "v_pseudo_grad_global_norm",
        "v_clipping_scale",
        "update_global_norm",
        "adam_m_global_norm",
        "adam_v_global_norm",
        "learning_rate",
    }

    for name in names:
        param = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))
        optimizer = AdaptiveAdamW(
            [{"params": [param], "weight_decay": 0.01}],
            optimizer_name=name,
            beta1=0.9,
            beta2=0.99,
            eps=1e-8,
            weight_decay=0.0,
            clip_threshold=0.5,
            clipping_scope="local",
            correct_bias=False,
        )

        diagnostics = None
        for _ in range(3):
            diagnostics = optimizer.step([torch.tensor([2.0, -3.0])], lr=1e-3)
            assert torch.isfinite(param).all()

        assert diagnostics is not None
        assert required_keys.issubset(diagnostics)
        for key, value in diagnostics.items():
            if isinstance(value, float):
                assert torch.isfinite(torch.tensor(value))


def test_resclip_vclip_uses_standard_clipped_gradient_for_second_moment():
    param = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
    optimizer = AdaptiveAdamW(
        [param],
        optimizer_name="adamw_resclip_euclidean_vclip",
        beta1=0.9,
        beta2=0.0,
        eps=1e-8,
        weight_decay=0.0,
        clip_threshold=1.0,
        clipping_scope="local",
        correct_bias=False,
    )

    optimizer.step([torch.tensor([10.0])], lr=0.0)
    diagnostics = optimizer.step([torch.tensor([10.0])], lr=0.0)

    assert torch.allclose(optimizer.exp_avg_sq[0], torch.tensor([1.0]))
    assert diagnostics["pseudo_grad_global_norm"] > diagnostics["v_pseudo_grad_global_norm"]
    assert abs(diagnostics["v_pseudo_grad_global_norm"] - 1.0) < 1e-6
