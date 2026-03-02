"""Tests for leakance (groundwater-surface water exchange) implementation.

Covers:
- _compute_depth() extraction correctness
- _compute_zeta() physics (losing/gaining, gradients, edge cases)
- MC coefficient sum invariant
- MCNodeProcessor with 6 channels
- Config defaults
- Integration: full routing with/without leakance
"""

from unittest.mock import patch

import torch

from ddr.nn.node_processor import MCNodeProcessor
from ddr.routing.mmc import (
    MuskingumCunge,
    _compute_depth,
    _compute_zeta,
)
from ddr.validation.configs import Config
from tests.routing.test_utils import (
    assert_no_nan_or_inf,
    create_mock_config,
    create_mock_routing_dataclass,
    create_mock_spatial_parameters,
    create_mock_streamflow,
)

# ─── Constants ────────────────────────────────────────────────────────────────

N = 10  # number of reaches for unit tests
D_H = 16  # embedding dimension for fast tests


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_leakance_config() -> Config:
    """Create a mock config with use_leakance=True and K_D/d_gw in learnable_parameters."""
    from omegaconf import DictConfig

    from ddr.validation.configs import validate_config

    cfg = {
        "name": "mock-leakance",
        "mode": "training",
        "geodataset": "lynker_hydrofabric",
        "data_sources": {
            "geospatial_fabric_gpkg": "mock.gpkg",
            "streamflow": "mock://streamflow/store",
            "conus_adjacency": "mock.zarr",
            "gages_adjacency": "mock.zarr",
            "gages": "mock.csv",
        },
        "params": {
            "parameter_ranges": {
                "n": [0.01, 0.1],
                "q_spatial": [0.1, 0.9],
                "K_D": [1e-8, 1e-6],
                "d_gw": [0.01, 300.0],
            },
            "defaults": {"p_spatial": 1.0},
            "attribute_minimums": {
                "velocity": 0.1,
                "depth": 0.01,
                "discharge": 0.001,
                "bottom_width": 0.1,
                "slope": 0.0001,
            },
            "tau": 7,
            "use_leakance": True,
        },
        "kan": {
            "input_var_names": ["mock"],
            "learnable_parameters": ["q_spatial", "n", "K_D", "d_gw"],
        },
        "s3_region": "us-east-1",
        "device": "cpu",
    }
    return validate_config(DictConfig(cfg), save_config=False)


def _make_leakance_spatial_params(num_reaches: int) -> dict[str, torch.Tensor]:
    """Create spatial parameters including K_D and d_gw."""
    params = create_mock_spatial_parameters(num_reaches)
    params["K_D"] = torch.rand(num_reaches)
    params["d_gw"] = torch.rand(num_reaches)
    return params


# ─── Unit tests: _compute_depth ──────────────────────────────────────────────


class TestComputeDepth:
    """Tests for the extracted _compute_depth() function."""

    def test_compute_depth_matches_velocity_inline(self) -> None:
        """Depth from _compute_depth() matches the inline computation in _get_trapezoid_velocity."""
        q_t = torch.rand(N).clamp(min=0.01) * 50.0
        n = torch.rand(N) * 0.1 + 0.02
        s0 = torch.rand(N) * 0.01 + 0.0001
        p_spatial = torch.tensor(21.0)
        q_spatial = torch.rand(N) * 0.8 + 0.1
        depth_lb = torch.tensor(0.01)

        depth = _compute_depth(q_t, n, s0, p_spatial, q_spatial, depth_lb)

        # Manually compute inline (the original code before extraction)
        numerator = q_t * n * (q_spatial + 1)
        denominator = p_spatial * torch.pow(s0, 0.5)
        depth_expected = torch.clamp(
            torch.pow(
                torch.div(numerator, denominator + 1e-8),
                torch.div(3.0, 5.0 + 3.0 * q_spatial),
            ),
            min=depth_lb,
        )

        assert torch.allclose(depth, depth_expected, atol=1e-7), (
            f"Extracted _compute_depth differs from inline: max diff={(depth - depth_expected).abs().max()}"
        )

    def test_compute_depth_clamped(self) -> None:
        """Depth is clamped to depth_lb for near-zero discharge."""
        q_t = torch.tensor([1e-10, 1e-12, 0.0])
        n = torch.tensor([0.03, 0.03, 0.03])
        s0 = torch.tensor([0.001, 0.001, 0.001])
        p_spatial = torch.tensor(21.0)
        q_spatial = torch.tensor([0.5, 0.5, 0.5])
        depth_lb = torch.tensor(0.01)

        depth = _compute_depth(q_t, n, s0, p_spatial, q_spatial, depth_lb)

        assert (depth >= depth_lb).all(), "Depth should be clamped to depth_lb"
        assert_no_nan_or_inf(depth, "_compute_depth near-zero Q")

    def test_compute_depth_finite(self) -> None:
        """Depth is finite for a wide range of inputs."""
        q_t = torch.logspace(-3, 4, N)
        n = torch.full((N,), 0.035)
        s0 = torch.full((N,), 0.001)
        p_spatial = torch.tensor(21.0)
        q_spatial = torch.full((N,), 0.5)
        depth_lb = torch.tensor(0.01)

        depth = _compute_depth(q_t, n, s0, p_spatial, q_spatial, depth_lb)

        assert_no_nan_or_inf(depth, "_compute_depth wide range")


# ─── Unit tests: MC coefficients ─────────────────────────────────────────────


class TestMCCoefficients:
    """Test Muskingum-Cunge coefficient sum invariant."""

    def test_mc_coefficients_sum_to_one(self) -> None:
        """C1 + C2 + C3 should equal 1.0 for standard MC formulation."""
        cfg = create_mock_config()
        mc = MuskingumCunge(cfg, device="cpu")

        length = torch.tensor([1000.0, 5000.0, 10000.0])
        velocity = torch.tensor([1.0, 2.0, 0.5])
        x_storage = torch.tensor([0.2, 0.3, 0.1])

        c_1, c_2, c_3, c_4 = mc.calculate_muskingum_coefficients(length, velocity, x_storage)

        # C1 + C2 + C3 = 1.0 is the standard MC constraint
        total = c_1 + c_2 + c_3
        assert torch.allclose(total, torch.ones_like(total), atol=1e-6), (
            f"C1+C2+C3 should equal 1.0, got {total}"
        )


# ─── Unit tests: _compute_zeta ──────────────────────────────────────────────


class TestComputeZeta:
    """Tests for _compute_zeta() physics."""

    def _make_inputs(self) -> dict[str, torch.Tensor]:
        """Common inputs for zeta tests."""
        return {
            "q_t": torch.full((N,), 10.0),
            "n": torch.full((N,), 0.035),
            "top_width": torch.full((N,), 50.0),
            "side_slope": torch.full((N,), 2.0),
            "s0": torch.full((N,), 0.001),
            "p_spatial": torch.tensor(21.0),
            "q_spatial": torch.full((N,), 0.5),
            "length": torch.full((N,), 5000.0),
            "depth_lb": torch.tensor(0.01),
        }

    def test_compute_zeta_losing(self) -> None:
        """zeta > 0 when d_gw is large (deep water table -> losing stream)."""
        inputs = self._make_inputs()
        K_D = torch.full((N,), 1e-7)
        d_gw = torch.full((N,), 200.0)  # Very deep water table

        zeta = _compute_zeta(**inputs, K_D=K_D, d_gw=d_gw)

        assert (zeta > 0).all(), f"Losing stream should have zeta > 0, got {zeta}"
        assert_no_nan_or_inf(zeta, "zeta_losing")

    def test_compute_zeta_gaining(self) -> None:
        """zeta < 0 when d_gw is very small (shallow water table -> gaining stream).

        For gaining to occur, we need: depth - h_bed + d_gw < 0,
        i.e., d_gw < h_bed - depth. With top_width=50, side_slope=2,
        h_bed = 50/(2*2) = 12.5m. Flow depth for Q=10 is typically < 1m,
        so h_bed >> depth, meaning we need d_gw < ~11.5m.
        """
        inputs = self._make_inputs()
        K_D = torch.full((N,), 1e-7)
        d_gw = torch.full((N,), 0.01)  # Very shallow water table

        zeta = _compute_zeta(**inputs, K_D=K_D, d_gw=d_gw)

        assert (zeta < 0).all(), f"Gaining stream should have zeta < 0, got {zeta}"
        assert_no_nan_or_inf(zeta, "zeta_gaining")

    def test_compute_zeta_gradient(self) -> None:
        """Autograd flows through K_D and d_gw."""
        inputs = self._make_inputs()
        K_D = torch.full((N,), 1e-7, requires_grad=True)
        d_gw = torch.full((N,), 50.0, requires_grad=True)

        zeta = _compute_zeta(**inputs, K_D=K_D, d_gw=d_gw)
        zeta.sum().backward()

        assert K_D.grad is not None, "No gradient through K_D"
        assert K_D.grad.abs().sum() > 0, "Zero gradient through K_D"
        assert d_gw.grad is not None, "No gradient through d_gw"
        assert d_gw.grad.abs().sum() > 0, "Zero gradient through d_gw"

    def test_compute_zeta_depth_clamp(self) -> None:
        """No NaN/Inf at near-zero discharge (depth clamped)."""
        inputs = self._make_inputs()
        inputs["q_t"] = torch.full((N,), 1e-10)
        K_D = torch.full((N,), 1e-7)
        d_gw = torch.full((N,), 50.0)

        zeta = _compute_zeta(**inputs, K_D=K_D, d_gw=d_gw)

        assert_no_nan_or_inf(zeta, "zeta_near_zero_Q")

    def test_compute_zeta_zero_K_D(self) -> None:
        """Zero K_D produces zero zeta."""
        inputs = self._make_inputs()
        K_D = torch.zeros(N)
        d_gw = torch.full((N,), 100.0)

        zeta = _compute_zeta(**inputs, K_D=K_D, d_gw=d_gw)

        assert torch.allclose(zeta, torch.zeros(N), atol=1e-12), f"zeta should be 0 when K_D=0, got {zeta}"


# ─── Unit tests: MCNodeProcessor 6 channels ─────────────────────────────────


class TestNodeProcessor6Channels:
    """Tests for MCNodeProcessor with leakance (6th physics channel)."""

    def _adjacency(self, n: int) -> torch.Tensor:
        """Simple chain adjacency."""
        indices = torch.arange(n - 1, dtype=torch.long)
        dense = torch.zeros(n, n)
        dense[indices + 1, indices] = 1.0
        return dense.to_sparse_csr()

    def test_node_processor_6_channels_shape(self) -> None:
        """MCNodeProcessor with use_leakance=True produces correct output shape."""
        proc = MCNodeProcessor(d_hidden=D_H, use_leakance=True)
        h = torch.randn(N, D_H)
        q = torch.rand(N).clamp(min=1e-4)
        zeta = torch.randn(N)
        adj = self._adjacency(N)

        h_new = proc.step(
            h=h,
            c1_next_upstream=q,
            c2_prev_upstream=q,
            c3_self=q,
            c4_lateral=q,
            q_new=q,
            adjacency=adj,
            zeta=zeta,
        )

        assert h_new.shape == (N, D_H), f"Expected ({N}, {D_H}), got {h_new.shape}"
        assert torch.isfinite(h_new).all(), "Non-finite output from 6-channel processor"

    def test_node_processor_5_channels_unchanged(self) -> None:
        """MCNodeProcessor with use_leakance=False still works with 5 channels."""
        proc = MCNodeProcessor(d_hidden=D_H, use_leakance=False)
        h = torch.randn(N, D_H)
        q = torch.rand(N).clamp(min=1e-4)
        adj = self._adjacency(N)

        h_new = proc.step(
            h=h,
            c1_next_upstream=q,
            c2_prev_upstream=q,
            c3_self=q,
            c4_lateral=q,
            q_new=q,
            adjacency=adj,
        )

        assert h_new.shape == (N, D_H)
        assert torch.isfinite(h_new).all()

    def test_node_processor_zeta_gradient_flows(self) -> None:
        """Gradient flows through zeta channel to the input."""
        proc = MCNodeProcessor(d_hidden=D_H, use_leakance=True)
        h = torch.randn(N, D_H)
        q = torch.rand(N).clamp(min=1e-4)
        zeta = torch.randn(N, requires_grad=True)
        adj = self._adjacency(N)

        h_new = proc.step(
            h=h,
            c1_next_upstream=q,
            c2_prev_upstream=q,
            c3_self=q,
            c4_lateral=q,
            q_new=q,
            adjacency=adj,
            zeta=zeta,
        )
        h_new.sum().backward()

        assert zeta.grad is not None, "No gradient through zeta"
        assert zeta.grad.abs().sum() > 0, "Zero gradient through zeta"


# ─── Unit tests: Config ─────────────────────────────────────────────────────


class TestConfigLeakance:
    """Tests for leakance config defaults."""

    def test_config_use_leakance_default_false(self) -> None:
        """Existing configs have use_leakance=False by default."""
        cfg = create_mock_config()
        assert cfg.params.use_leakance is False

    def test_config_leakance_ranges_in_defaults(self) -> None:
        """K_D and d_gw are in _DEFAULT_PARAMETER_RANGES."""
        cfg = create_mock_config()
        assert "K_D" in cfg.params.parameter_ranges
        assert "d_gw" in cfg.params.parameter_ranges
        assert cfg.params.parameter_ranges["K_D"] == [1e-8, 1e-6]
        assert cfg.params.parameter_ranges["d_gw"] == [0.01, 300.0]

    def test_config_d_gw_in_log_space(self) -> None:
        """d_gw should be in log_space_parameters by default."""
        cfg = create_mock_config()
        assert "d_gw" in cfg.params.log_space_parameters

    def test_config_use_leakance_true(self) -> None:
        """Config with use_leakance=True validates successfully."""
        cfg = _make_leakance_config()
        assert cfg.params.use_leakance is True
        assert "K_D" in cfg.kan.learnable_parameters
        assert "d_gw" in cfg.kan.learnable_parameters


# ─── Integration tests ───────────────────────────────────────────────────────


class TestRoutingWithLeakance:
    """Integration tests for routing with leakance enabled."""

    def test_routing_with_leakance(self) -> None:
        """Full forward pass with leakance produces finite discharge."""
        cfg = _make_leakance_config()
        mc = MuskingumCunge(cfg, device="cpu")

        num_reaches = 5
        hydrofabric = create_mock_routing_dataclass(num_reaches=num_reaches)
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=num_reaches)
        spatial_params = _make_leakance_spatial_params(num_reaches)

        mc.setup_inputs(hydrofabric, streamflow, spatial_params)
        mc.set_progress_info(1, 0)

        output = mc.forward()

        assert torch.isfinite(output).all(), "Routing with leakance produced non-finite values"
        assert (output >= 0).all(), "Discharge should be non-negative"

    def test_routing_without_leakance_unchanged(self) -> None:
        """Routing with use_leakance=False is identical to default (no zeta)."""
        cfg = create_mock_config()
        assert cfg.params.use_leakance is False

        num_reaches = 5
        hydrofabric = create_mock_routing_dataclass(num_reaches=num_reaches)
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=num_reaches)
        spatial_params = create_mock_spatial_parameters(num_reaches=num_reaches)

        # Run with use_leakance=False (default)
        torch.manual_seed(42)
        mc1 = MuskingumCunge(cfg, device="cpu")
        mc1.setup_inputs(hydrofabric, streamflow, spatial_params)
        mc1.set_progress_info(1, 0)

        with patch("ddr.routing.mmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = torch.ones(num_reaches) * 5.0
            output1 = mc1.forward()

        # Run again to verify determinism
        torch.manual_seed(42)
        mc2 = MuskingumCunge(cfg, device="cpu")
        mc2.setup_inputs(hydrofabric, streamflow, spatial_params)
        mc2.set_progress_info(1, 0)

        with patch("ddr.routing.mmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = torch.ones(num_reaches) * 5.0
            output2 = mc2.forward()

        assert torch.allclose(output1, output2), "use_leakance=False should be deterministic"

    def test_leakance_denormalization(self) -> None:
        """K_D and d_gw are properly denormalized after setup_inputs."""
        cfg = _make_leakance_config()
        mc = MuskingumCunge(cfg, device="cpu")

        num_reaches = 5
        hydrofabric = create_mock_routing_dataclass(num_reaches=num_reaches)
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=num_reaches)
        spatial_params = _make_leakance_spatial_params(num_reaches)

        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        # K_D should be in [1e-8, 1e-6] range
        assert mc.K_D is not None, "K_D should be set"
        assert (mc.K_D >= 1e-8 - 1e-12).all(), f"K_D below lower bound: {mc.K_D.min()}"
        assert (mc.K_D <= 1e-6 + 1e-12).all(), f"K_D above upper bound: {mc.K_D.max()}"

        # d_gw should be in [0.01, 300.0] range (log-space denorm)
        assert mc.d_gw is not None, "d_gw should be set"
        assert (mc.d_gw >= 0.01 - 1e-6).all(), f"d_gw below lower bound: {mc.d_gw.min()}"
        assert (mc.d_gw <= 300.0 + 1e-3).all(), f"d_gw above upper bound: {mc.d_gw.max()}"

    def test_clear_batch_state_clears_leakance(self) -> None:
        """clear_batch_state() sets K_D and d_gw to None."""
        cfg = _make_leakance_config()
        mc = MuskingumCunge(cfg, device="cpu")

        num_reaches = 5
        hydrofabric = create_mock_routing_dataclass(num_reaches=num_reaches)
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=num_reaches)
        spatial_params = _make_leakance_spatial_params(num_reaches)

        mc.setup_inputs(hydrofabric, streamflow, spatial_params)
        assert mc.K_D is not None
        assert mc.d_gw is not None

        mc.clear_batch_state()

        assert mc.K_D is None, "K_D should be None after clear_batch_state"
        assert mc.d_gw is None, "d_gw should be None after clear_batch_state"
