import math
import statistics
import unittest
from unittest import mock

import torch

from tokengs.losses import geo_losses
from tokengs.losses.geo_losses import (
    _aggregate_parent_covariance,
    _opacity_mass_weights,
    _stable_cholesky,
    _weighted_normal_wasserstein,
    _world_covariances,
    compute_geo_losses,
)


def _identity_rotations(batch: int, slots: int, children: int, dtype=torch.float32):
    rotations = torch.zeros(batch, slots, children, 4, dtype=dtype)
    rotations[..., 0] = 1.0
    return rotations


def _structured_inputs(dtype=torch.float32):
    base = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=dtype,
    )
    centers = 0.1 * base.view(1, 1, 4, 3).repeat(2, 3, 1, 1)
    slot_offsets = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [0.1, 0.8, 0.2]]],
        dtype=dtype,
    )
    centers = centers + slot_offsets.unsqueeze(2)
    scales = torch.tensor([0.08, 0.12, 0.2], dtype=dtype).view(1, 1, 1, 3)
    scales = scales.repeat(2, 3, 4, 1)
    opacities = torch.linspace(0.1, 0.9, 24, dtype=dtype).reshape(2, 3, 4, 1)
    rotations = _identity_rotations(2, 3, 4, dtype=dtype)
    return centers, scales, opacities, rotations


def _normal_wasserstein_reference(projections, weights):
    """独立 FP64 标量参考：fsum 双端质量与解析正态积分，不调用被测函数。"""
    values = projections.detach().cpu().double().reshape(-1).tolist()
    raw_mass = weights.detach().cpu().double().reshape(-1).tolist()
    total = math.fsum(raw_mass)
    atoms = sorted((x, w / total, i) for i, (x, w) in enumerate(zip(values, raw_mass)) if w > 0)
    density = [0.0]
    normal = statistics.NormalDist()
    for boundary in range(1, len(atoms)):
        tail = min(math.fsum(w for _, w, _ in atoms[:boundary]), math.fsum(w for _, w, _ in atoms[boundary:]))
        quantile = normal.inv_cdf(tail)
        density.append(math.exp(-0.5 * quantile**2) / math.sqrt(2.0 * math.pi))
    density.append(0.0)
    terms, gradients = [], [0.0] * len(values)
    for j, (x, w, index) in enumerate(atoms):
        moment = density[j] - density[j + 1]
        terms.append(w * x**2 - 2.0 * x * moment)
        gradients[index] = 2.0 * (w * x - moment)
    return 1.0 + math.fsum(terms), torch.tensor(gradients, dtype=torch.float64).reshape(projections.shape)


def _benchmark_weighted_wasserstein():
    """手动 CUDA 微基准；不由 pytest 执行，不启动训练。"""
    def baseline(projections, weights):
        mass = weights.detach().unsqueeze(-1).expand_as(projections)
        sortable = torch.where(mass > 0, projections, torch.full_like(projections, float("inf")))
        sorted_values, order = sortable.sort(dim=2)
        sorted_mass = mass.gather(2, order)
        sorted_values = torch.where(sorted_mass > 0, sorted_values, torch.zeros_like(sorted_values))
        upper = sorted_mass.cumsum(dim=2)
        total = upper[:, :, -1:, :]
        upper = (upper / torch.where(total > 0, total, torch.ones_like(total))).clamp(0.0, 1.0)
        lower = torch.cat((torch.zeros_like(upper[:, :, :1, :]), upper[:, :, :-1, :]), dim=2)

        def density(p):
            interior = (p > 0) & (p < 1)
            q = torch.special.ndtri(torch.where(interior, p, torch.full_like(p, 0.5)))
            value = torch.exp(-0.5 * q.square()) / math.sqrt(2.0 * math.pi)
            return torch.where(interior, value, torch.zeros_like(value))

        moment = density(lower) - density(upper)
        result = 1.0 + ((upper - lower) * sorted_values.square() - 2.0 * sorted_values * moment).sum(dim=2)
        return torch.where(total.squeeze(2) > 0, result.clamp_min(0.0), torch.zeros_like(result))

    torch.manual_seed(65)
    for batch, slots in ((1, 1), (1, 4096), (2, 4096)):
        centers = torch.randn(batch, slots, 64, 3, device="cuda", requires_grad=True)
        scales = torch.full_like(centers, 0.1)
        opacities = torch.rand(batch, slots, 64, 1, device="cuda")
        weights = opacities.squeeze(-1)
        projections = torch.randn(batch, slots, 64, 8, device="cuda", requires_grad=True)
        for region in ("sw_forward", "sw_backward", "geometry_backward"):
            results = {}
            for label, function in (("old_fp32", baseline), ("stable", _weighted_normal_wasserstein)):
                def step():
                    if region == "geometry_backward":
                        centers.grad = None
                        compute_geo_losses(centers, scales, opacities)["total_geo_loss"].backward()
                    else:
                        projections.grad = None
                        result = function(projections, weights)
                        if region == "sw_backward":
                            result.mean().backward()

                with mock.patch.object(geo_losses, "_weighted_normal_wasserstein", function):
                    for _ in range(3):
                        step()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    allocated = torch.cuda.memory_allocated()
                    samples = []
                    for _ in range(5):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(5):
                            step()
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end) / 5)
                    results[label] = (statistics.median(samples), (torch.cuda.max_memory_allocated() - allocated) / 2**20)
            old, new = results["old_fp32"], results["stable"]
            print(f"B={batch} S={slots} C=64 D=8 {region}: old={old[0]:.4f}ms new={new[0]:.4f}ms "
                  f"ratio={new[0]/old[0]:.3f} per_slot={new[0]*1000/(batch*slots):.4f}us "
                  f"extra_peak_old/new={old[1]:.1f}/{new[1]:.1f}MiB", flush=True)


class TestOpacityMassSemantics(unittest.TestCase):
    def test_single_opaque_child_has_effective_children_one(self):
        centers, scales, _, rotations = _structured_inputs()
        opacities = torch.zeros(2, 3, 4, 1)
        opacities[..., 0, 0] = 1.0

        weights, active_slots = _opacity_mass_weights(opacities)
        result = compute_geo_losses(
            centers,
            scales,
            opacities,
            gs_rotations=rotations,
        )

        self.assertTrue(active_slots.all().item())
        torch.testing.assert_close(weights[..., 0], torch.ones(2, 3))
        torch.testing.assert_close(weights[..., 1:], torch.zeros(2, 3, 3))
        torch.testing.assert_close(
            result["stats"]["effective_children"], torch.tensor(1.0)
        )

    def test_zero_opacity_skips_geometry_without_uniform_mass(self):
        centers, scales, opacities, rotations = _structured_inputs()
        centers.requires_grad_()
        scales.requires_grad_()
        rotations.requires_grad_()
        opacities = torch.zeros_like(opacities, requires_grad=True)

        weights, active_slots = _opacity_mass_weights(opacities)
        result = compute_geo_losses(
            centers,
            scales,
            opacities,
            gs_rotations=rotations,
        )

        # Fixed semantics: zero-mass slots are skipped, not assigned uniform mass.
        self.assertFalse(active_slots.any().item())
        torch.testing.assert_close(weights, torch.zeros_like(weights))
        for key in ("sw_loss", "packing_loss", "inter_loss", "total_geo_loss"):
            self.assertTrue(torch.isfinite(result[key]).item(), key)
            self.assertEqual(result[key].item(), 0.0, key)
        self.assertEqual(result["stats"]["effective_children"].item(), 0.0)
        for value in result["stats"].values():
            self.assertTrue(torch.isfinite(value).item())

        result["total_geo_loss"].backward()
        for tensor in (centers, scales, rotations):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())
            torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor), rtol=0.0, atol=0.0)
        self.assertIsNone(opacities.grad)

    def test_uniform_opacity_produces_uniform_mass(self):
        centers, scales, _, rotations = _structured_inputs()
        opacities = torch.full((2, 3, 4, 1), 0.25)

        weights, active_slots = _opacity_mass_weights(opacities)
        result = compute_geo_losses(
            centers,
            scales,
            opacities,
            gs_rotations=rotations,
        )

        self.assertTrue(active_slots.all().item())
        torch.testing.assert_close(weights, torch.full_like(weights, 0.25))
        torch.testing.assert_close(
            result["stats"]["effective_children"], torch.tensor(4.0)
        )


class TestWorldCovariance(unittest.TestCase):
    def test_identity_quaternion_preserves_local_covariance(self):
        scales = torch.tensor([[[[1.0, 2.0, 3.0]]]])
        rotations = torch.tensor([[[[2.0, 0.0, 0.0, 0.0]]]])

        covariance = _world_covariances(scales, rotations)

        torch.testing.assert_close(
            covariance[0, 0, 0], torch.diag(torch.tensor([1.0, 4.0, 9.0]))
        )

    def test_anisotropic_covariance_axes_swap_after_z_rotation(self):
        scales = torch.tensor([[[[1.0, 2.0, 3.0]]]])
        half_angle = math.pi / 4.0
        rotations = torch.tensor(
            [[[[math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]]]]
        )

        covariance = _world_covariances(scales, rotations)

        torch.testing.assert_close(
            covariance[0, 0, 0],
            torch.diag(torch.tensor([4.0, 1.0, 9.0])),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_parent_covariance_uses_same_opacity_mass_weights(self):
        center_covariance = torch.zeros(1, 1, 3, 3)
        scales = torch.tensor([[[[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]]]])
        rotations = _identity_rotations(1, 1, 2)
        opacities = torch.tensor([[[[1.0], [0.0]]]])
        weights, _ = _opacity_mass_weights(opacities)

        parent_covariance = _aggregate_parent_covariance(
            center_covariance,
            scales,
            rotations,
            weights,
        )

        torch.testing.assert_close(
            parent_covariance[0, 0], torch.diag(torch.tensor([1.0, 4.0, 9.0]))
        )


class TestGeoLossNumerics(unittest.TestCase):
    def test_batch_slot_child_shapes_and_bfloat16_cpu(self):
        centers, scales, opacities, rotations = _structured_inputs(dtype=torch.bfloat16)

        result = compute_geo_losses(
            centers,
            scales,
            opacities,
            gs_rotations=rotations,
        )

        for key in ("sw_loss", "packing_loss", "inter_loss", "total_geo_loss"):
            self.assertEqual(result[key].shape, torch.Size([]))
            self.assertEqual(result[key].dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(result[key]).item())
        for value in result["stats"].values():
            self.assertEqual(value.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(value).item())

    def test_backward_is_finite_and_opacity_weights_are_detached(self):
        centers, scales, opacities, rotations = _structured_inputs()
        centers.requires_grad_()
        scales.requires_grad_()
        opacities.requires_grad_()
        rotations.requires_grad_()

        result = compute_geo_losses(
            centers,
            scales,
            opacities,
            lambda_sw=1.0,
            lambda_packing=1.0,
            lambda_inter=1.0,
            gs_rotations=rotations,
        )
        result["total_geo_loss"].backward()

        for tensor in (centers, scales, rotations):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())
        self.assertIsNone(opacities.grad)

    def test_cpu_results_are_deterministic(self):
        inputs = _structured_inputs()

        first = compute_geo_losses(*inputs[:3], gs_rotations=inputs[3])
        second = compute_geo_losses(*inputs[:3], gs_rotations=inputs[3])

        for key in ("sw_loss", "packing_loss", "inter_loss", "total_geo_loss"):
            torch.testing.assert_close(first[key], second[key], rtol=0.0, atol=0.0)
        for key in first["stats"]:
            torch.testing.assert_close(
                first["stats"][key], second["stats"][key], rtol=0.0, atol=0.0
            )


class TestWeightedWasserstein(unittest.TestCase):
    def _check_tiny_tail_mass(self, device):
        for tiny, position in ((1e-8, 1e4), (1e-12, 1e6), (1e-20, 1e8), (1e-24, 1e12)):
            for sign in (-1.0, 1.0):
                for order in ([0, 1, 2], [2, 1, 0]):
                    with self.subTest(device=device, tiny=tiny, sign=sign, order=order):
                        projections = torch.tensor([0.0, sign * position, -sign * 1e15], device=device)
                        projections = projections[order].reshape(1, 1, 3, 1).requires_grad_()
                        weights = torch.tensor([1.0, tiny, 0.0], device=device)[order].reshape(1, 1, 3).requires_grad_()
                        expected, gradient = _normal_wasserstein_reference(projections, weights)
                        loss = _weighted_normal_wasserstein(projections, weights)
                        self.assertEqual(loss.dtype, torch.float32)
                        self.assertTrue(torch.isfinite(loss).all().item())
                        torch.testing.assert_close(loss.double(), torch.full_like(loss.double(), expected), rtol=2e-7, atol=2e-7)
                        loss.sum().backward()
                        torch.testing.assert_close(projections.grad.double().cpu(), gradient, rtol=3e-7, atol=0.0)
                        tiny_index, zero_index = order.index(1), order.index(2)
                        self.assertGreater(sign * projections.grad[0, 0, tiny_index, 0].item(), 0.0)
                        self.assertEqual(projections.grad[0, 0, zero_index, 0].item(), 0.0)
                        self.assertIsNone(weights.grad)
                        # FP64 输出另外验证积分本身，而非只以 FP32 最终舍入作容差。
                        precise = _weighted_normal_wasserstein(projections.double(), weights.double())
                        torch.testing.assert_close(precise.cpu(), torch.full_like(precise.cpu(), expected), rtol=2e-14, atol=2e-14)

    def test_tiny_positive_and_negative_tail_mass_matches_fp64_reference_cpu(self):
        self._check_tiny_tail_mass("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_tiny_positive_and_negative_tail_mass_matches_fp64_reference_cuda(self):
        self._check_tiny_tail_mass("cuda")

    def test_multiple_small_atoms_on_both_tails_and_interior(self):
        for dtype in (torch.float32, torch.float64):
            projections = torch.tensor([-1e6, -1e4, -3.0, 0.0, 2.0, 1e4, 1e6, 1e15], dtype=dtype).reshape(1, 1, 8, 1)
            projections.requires_grad_()
            weights = torch.tensor([1e-12, 1e-8, 0.2, 1e-12, 0.8, 1e-8, 1e-12, 0.0], dtype=dtype).reshape(1, 1, 8)
            expected, gradient = _normal_wasserstein_reference(projections, weights)
            loss = _weighted_normal_wasserstein(projections, weights)
            tolerance = 2e-7 if dtype == torch.float32 else 2e-14
            torch.testing.assert_close(loss, torch.full_like(loss, expected), rtol=tolerance, atol=tolerance)
            loss.sum().backward()
            torch.testing.assert_close(projections.grad.double(), gradient, rtol=3e-7 if dtype == torch.float32 else 2e-14, atol=2e-15)
            self.assertEqual(projections.grad[..., -1, :].item(), 0.0)

    def test_endpoint_quantiles_are_never_evaluated_at_zero_or_one(self):
        projections = torch.tensor([0.0, 1e6, -1e15, 1e15]).reshape(1, 1, 4, 1).requires_grad_()
        original = torch.special.ndtri
        observed = []

        def inspect(probabilities):
            observed.append(probabilities.numel())
            self.assertEqual(probabilities.dtype, torch.float64)
            self.assertTrue(((probabilities > 0) & (probabilities <= 0.5)).all().item())
            return original(probabilities)

        with mock.patch.object(torch.special, "ndtri", side_effect=inspect):
            for weights in ([1.0, 1e-12, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0] * 4):
                loss = _weighted_normal_wasserstein(projections, torch.tensor(weights).reshape(1, 1, 4))
                self.assertTrue(torch.isfinite(loss).all().item())
                loss.sum().backward()
        self.assertEqual(observed, [3, 3, 3])
        self.assertTrue(torch.isfinite(projections.grad).all().item())

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "需要支持 BF16 的 CUDA")
    def test_cuda_bfloat16_tiny_mass_keeps_fp32_region_and_zero_gradients(self):
        inputs = [value.cuda().bfloat16() for value in _structured_inputs()]
        inputs[2] = torch.tensor([1.0, 1e-8, 1e-12, 0.0], device="cuda", dtype=torch.bfloat16).reshape(1, 1, 4, 1).expand(2, 3, 4, 1)
        inputs[0].requires_grad_()
        inputs[2].requires_grad_()
        reference = compute_geo_losses(*[value.float() for value in inputs[:3]], gs_rotations=inputs[3].float())
        original, observed = geo_losses._weighted_normal_wasserstein, []

        def inspect(projections, weights):
            observed.append((projections.dtype, weights.dtype, torch.is_autocast_enabled("cuda")))
            return original(projections, weights)

        with mock.patch.object(geo_losses, "_weighted_normal_wasserstein", side_effect=inspect):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = compute_geo_losses(*inputs[:3], gs_rotations=inputs[3])
                result["sw_loss"].backward()
                self.assertTrue(torch.is_autocast_enabled("cuda"))
        self.assertEqual(observed, [(torch.float32, torch.float32, False)])
        for key in ("sw_loss", "packing_loss", "inter_loss", "total_geo_loss"):
            torch.testing.assert_close(result[key], reference[key].bfloat16(), rtol=0.0, atol=0.0)
        self.assertTrue(torch.isfinite(inputs[0].grad).all().item())
        self.assertGreater(inputs[0].grad[..., 1:3, :].abs().sum().item(), 0.0)
        torch.testing.assert_close(inputs[0].grad[..., 3, :], torch.zeros_like(inputs[0].grad[..., 3, :]), rtol=0.0, atol=0.0)
        self.assertIsNone(inputs[2].grad)

    def test_nonuniform_two_atom_distribution_matches_hand_integral_and_gradient(self):
        projections = torch.tensor([[[[-1.0], [2.0]]]], requires_grad=True)
        weights = torch.tensor([[[0.25, 0.75]]], requires_grad=True)
        # Φ⁻¹(1/4)=-0.67448975，φ(Φ⁻¹(1/4))=0.31777657。
        density = math.exp(-0.5 * 0.6744897501960817**2) / math.sqrt(2.0 * math.pi)
        expected = 4.25 - 6.0 * density

        loss = _weighted_normal_wasserstein(projections, weights)
        torch.testing.assert_close(loss, torch.tensor([[[expected]]]))
        loss.sum().backward()
        expected_gradient = torch.tensor([[[[-0.5 + 2.0 * density], [3.0 - 2.0 * density]]]])
        torch.testing.assert_close(projections.grad, expected_gradient)
        self.assertIsNone(weights.grad)

    def test_uniform_two_atom_distribution_matches_exact_gaussian_integral(self):
        projections = torch.tensor([[[[-1.0], [1.0]]]])
        weights = torch.full((1, 1, 2), 0.5)
        expected = 2.0 - 4.0 / math.sqrt(2.0 * math.pi)
        torch.testing.assert_close(
            _weighted_normal_wasserstein(projections, weights),
            torch.tensor([[[expected]]]),
        )

    def test_weight_normalization_permutation_and_zero_atoms(self):
        projections = torch.tensor([[[[-1.0], [100.0], [2.0], [-100.0]]]], requires_grad=True)
        weights = torch.tensor([[[0.25, 0.0, 0.75, 0.0]]])
        loss = _weighted_normal_wasserstein(projections, weights)
        reference = _weighted_normal_wasserstein(projections[:, :, [0, 2]], weights[:, :, [0, 2]])
        for order in ([3, 2, 1, 0], [1, 0, 3, 2]):
            shuffled = _weighted_normal_wasserstein(projections[:, :, order], 7.0 * weights[:, :, order])
            torch.testing.assert_close(loss, shuffled, rtol=0.0, atol=1e-6)
        torch.testing.assert_close(loss, reference, rtol=0.0, atol=0.0)
        loss.sum().backward()
        torch.testing.assert_close(projections.grad[:, :, [1, 3]], torch.zeros(1, 1, 2, 1), rtol=0.0, atol=0.0)

    def test_single_opaque_child_is_translation_invariant_and_has_unit_sw(self):
        centers, scales, _, rotations = _structured_inputs()
        centers = centers[:1, :1].clone().requires_grad_()
        opacities = torch.tensor([[[[1.0], [0.0], [0.0], [0.0]]]], requires_grad=True)
        reference = compute_geo_losses(centers, scales[:1, :1], opacities, gs_rotations=rotations[:1, :1])
        moved = centers.detach().clone()
        moved[..., 1:, :] = torch.tensor([[1e5, -2e5, 3e5], [-4e5, 5e5, -6e5], [7e5, 8e5, 9e5]])
        moved[..., 0, :] += 7.0
        moved.requires_grad_()
        changed = compute_geo_losses(moved, scales[:1, :1], opacities, gs_rotations=rotations[:1, :1])
        torch.testing.assert_close(reference["sw_loss"], torch.tensor(1.0), rtol=0.0, atol=0.0)
        torch.testing.assert_close(reference["sw_loss"], changed["sw_loss"], rtol=0.0, atol=0.0)
        changed["sw_loss"].backward()
        torch.testing.assert_close(moved.grad, torch.zeros_like(moved), rtol=0.0, atol=0.0)
        self.assertIsNone(opacities.grad)

    def test_transparent_children_do_not_change_any_geometry_term(self):
        centers, scales, _, rotations = _structured_inputs()
        opacities = torch.tensor([0.2, 0.0, 0.3, 0.5]).reshape(1, 1, 4, 1).expand(2, 3, 4, 1)
        reference = compute_geo_losses(centers, scales, opacities, gs_rotations=rotations)
        moved = centers.clone()
        moved[..., 1, :] = torch.tensor([-1e6, 2e6, -3e6])
        moved.requires_grad_()
        changed = compute_geo_losses(moved, scales, opacities, gs_rotations=rotations)
        for key in ("sw_loss", "packing_loss", "inter_loss", "total_geo_loss"):
            torch.testing.assert_close(reference[key], changed[key], rtol=0.0, atol=0.0)
        for key in reference["stats"]:
            torch.testing.assert_close(reference["stats"][key], changed["stats"][key], rtol=0.0, atol=0.0)
        changed["total_geo_loss"].backward()
        torch.testing.assert_close(moved.grad[..., 1, :], torch.zeros_like(moved.grad[..., 1, :]), rtol=0.0, atol=0.0)
        self.assertGreater(moved.grad[..., [0, 2, 3], :].abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(moved.grad).all().item())

    def test_zero_mass_slot_does_not_change_active_slots(self):
        centers, scales, opacities, rotations = _structured_inputs()
        opacities[:, -1] = 0.0
        centers.requires_grad_()
        mixed = compute_geo_losses(centers, scales, opacities, gs_rotations=rotations)
        active = compute_geo_losses(centers[:, :2], scales[:, :2], opacities[:, :2], gs_rotations=rotations[:, :2])
        for key in ("sw_loss", "packing_loss", "inter_loss", "total_geo_loss"):
            torch.testing.assert_close(mixed[key], active[key], atol=1e-6, rtol=1e-6)
        mixed["total_geo_loss"].backward()
        torch.testing.assert_close(centers.grad[:, -1], torch.zeros_like(centers.grad[:, -1]), rtol=0.0, atol=0.0)


class TestAutocastAndDegenerateCovariance(unittest.TestCase):
    def _check_autocast(self, device, dtype, amplitude, perturbation):
        values = torch.linspace(-amplitude, amplitude, 64, device=device)
        direction = torch.tensor([1.0, 0.7, -0.3], device=device)
        centers = (values[:, None] * direction).reshape(1, 1, 64, 3).repeat(1, 3, 1, 1)
        centers[..., 1] += perturbation * torch.sin(values)
        centers[:, 1] += 0.3
        centers[:, 2] -= 0.3
        centers = centers.to(dtype).requires_grad_()
        scales = torch.full_like(centers, 0.02, requires_grad=True)
        opacities = torch.ones(1, 3, 64, 1, dtype=dtype, device=device, requires_grad=True)
        rotations = _identity_rotations(1, 3, 64, dtype=dtype).to(device).requires_grad_()
        reference = compute_geo_losses(centers, scales, opacities, gs_rotations=rotations)
        observed = []
        original = geo_losses._world_covariances

        def inspect_region(inner_scales, inner_rotations):
            observed.append((inner_scales.dtype, inner_rotations.dtype, torch.is_autocast_enabled(device)))
            value = original(inner_scales, inner_rotations)
            self.assertEqual(value.dtype, torch.float32)
            return value

        # 嵌套的两个 autocast 都开启；被测函数须自行建立完整 FP32 区域。
        with mock.patch.object(geo_losses, "_world_covariances", side_effect=inspect_region):
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    result = compute_geo_losses(centers, scales, opacities, gs_rotations=rotations)
                    self.assertTrue(torch.is_autocast_enabled(device))
                    result["total_geo_loss"].backward()
        self.assertEqual(observed, [(torch.float32, torch.float32, False)])
        for key in ("sw_loss", "packing_loss", "inter_loss", "total_geo_loss"):
            self.assertTrue(torch.isfinite(result[key]).item(), key)
            torch.testing.assert_close(result[key], reference[key], atol=0.0, rtol=0.0)
        for value in result["stats"].values():
            self.assertTrue(torch.isfinite(value).item())
        for tensor in (centers, scales, rotations):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())
        self.assertIsNone(opacities.grad)

    def test_cpu_nested_bfloat16_autocast_collinear_and_near_degenerate(self):
        for dtype in (torch.float32, torch.bfloat16):
            for amplitude, perturbation in ((1.0, 0.0), (1000.0, 0.0), (50.0, 1e-4)):
                with self.subTest(dtype=dtype, amplitude=amplitude, perturbation=perturbation):
                    self._check_autocast("cpu", dtype, amplitude, perturbation)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "需要支持 BF16 的 CUDA")
    def test_cuda_nested_bfloat16_autocast_collinear_and_near_degenerate(self):
        for dtype in (torch.float32, torch.bfloat16):
            for amplitude, perturbation in ((1.0, 0.0), (1000.0, 0.0), (50.0, 1e-4)):
                with self.subTest(dtype=dtype, amplitude=amplitude, perturbation=perturbation):
                    self._check_autocast("cuda", dtype, amplitude, perturbation)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "需要支持 BF16 的 CUDA")
    def test_simultaneous_cpu_and_cuda_autocast_restore_outer_contexts(self):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            self._check_autocast("cuda", torch.float32, 1.0, 0.0)
            self.assertTrue(torch.is_autocast_enabled("cpu"))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            self._check_autocast("cpu", torch.float32, 1.0, 0.0)
            self.assertTrue(torch.is_autocast_enabled("cuda"))

    def test_roundoff_indefinite_covariance_repair_has_finite_backward(self):
        covariance = torch.diag(torch.tensor([1.0, -1e-5, 0.0])).requires_grad_()
        cholesky = _stable_cholesky(covariance)
        self.assertTrue(torch.isfinite(cholesky).all().item())
        cholesky.square().sum().backward()
        self.assertTrue(torch.isfinite(covariance.grad).all().item())

    def test_nonfinite_covariance_and_even_inactive_inputs_are_not_hidden(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            covariance = torch.eye(3)
            covariance[0, 0] = bad
            with self.assertRaisesRegex(ValueError, "finite"):
                _stable_cholesky(covariance)
            for input_index in range(4):
                inputs = list(_structured_inputs())
                inputs[2].zero_()
                inputs[input_index].flatten()[0] = bad
                with self.assertRaisesRegex(ValueError, "finite"):
                    compute_geo_losses(*inputs[:3], gs_rotations=inputs[3])

    def test_float16_metrics_do_not_overflow_training_scalar_range(self):
        centers, scales, opacities, rotations = _structured_inputs(dtype=torch.float16)
        centers.mul_(10.0)
        scales.fill_(1e-4)
        result = compute_geo_losses(centers, scales, opacities, gs_rotations=rotations)
        self.assertGreater(result["stats"]["packing_ratio"].item(), torch.finfo(torch.float16).max)
        for key in ("sw_loss", "packing_loss", "inter_loss", "total_geo_loss"):
            self.assertEqual(result[key].dtype, torch.float32)
            self.assertTrue(torch.isfinite(result[key]).item())
        for value in result["stats"].values():
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.isfinite(value).item())


def _ffrgs_reference(positions, scales, quaternions, opacities, target_distance=2.0, k=4):
    """逐行摘自 FFRGS models/losses.py 的 _mass_weights 与 inter_slot_loss（num_queries=N）。"""
    opacity = opacities.squeeze(-1).clamp_min(1e-6)
    area = scales.prod(dim=-1).clamp_min(1e-18).pow(2.0 / 3.0)
    mass = opacity * area
    w = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    R = geo_losses._quaternion_wxyz_to_rotation_matrix(quaternions)
    B, N = positions.shape[:2]
    k = min(k, N - 1)
    eye = torch.eye(3, dtype=positions.dtype)
    mean = (w.unsqueeze(-1) * positions).sum(dim=-2)
    centered = positions - mean.unsqueeze(-2)
    cov = torch.einsum("bng,bngi,bngj->bnij", w, centered, centered)
    child_cov = torch.einsum("bngik,bngk,bngjk->bngij", R, scales.square(), R)
    cov = cov + (w[..., None, None] * child_cov).sum(dim=-3) + 1e-6 * eye
    d_euc = torch.cdist(mean, mean)
    d_euc.diagonal(dim1=-2, dim2=-1).fill_(torch.inf)
    nn_idx = d_euc.topk(k, dim=-1, largest=False).indices
    bi = torch.arange(B).view(B, 1, 1)
    diff = mean.unsqueeze(-2) - mean[bi, nn_idx]
    pair_cov = 0.5 * (cov.unsqueeze(-3) + cov[bi, nn_idx]) + 1e-6 * eye
    solved = torch.cholesky_solve(diff.unsqueeze(-1), torch.linalg.cholesky(pair_cov)).squeeze(-1)
    dist = (diff * solved).sum(dim=-1).clamp_min(1e-12).sqrt()
    log_ratio = torch.log(dist / target_distance)
    loss = torch.nn.functional.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio), beta=1.0)
    return w, loss, dist


class TestFFRGSInterForm(unittest.TestCase):
    def _two_slot_inputs(self, separation, scale=0.1):
        centers = torch.zeros(1, 2, 2, 3)
        centers[0, 1, :, 0] = separation
        scales = torch.full((1, 2, 2, 3), scale)
        opacities = torch.zeros(1, 2, 2, 1)
        opacities[:, :, 0] = 1.0
        return centers, scales, opacities, _identity_rotations(1, 2, 2)

    def test_log_smooth_l1_matches_hand_value_far_and_close(self):
        # 单个不透明 child：parent 协方差 = 0.01·I，马氏距离 = separation / 0.1。
        for separation, distance, close, far in ((0.8, 8.0, 0.0, 1.0), (0.1, 1.0, 1.0, 0.0)):
            log_ratio = math.log(distance / 2.0)
            expected = abs(log_ratio) - 0.5 if abs(log_ratio) >= 1.0 else 0.5 * log_ratio**2
            inputs = self._two_slot_inputs(separation)
            result = compute_geo_losses(*inputs[:3], gs_rotations=inputs[3])
            legacy = compute_geo_losses(*inputs[:3], gs_rotations=inputs[3], inter_form="squared")
            stats = result["stats"]
            self.assertAlmostEqual(result["inter_loss"].item(), expected, delta=1e-3)
            self.assertAlmostEqual(legacy["inter_loss"].item(), (distance - 2.0) ** 2, delta=1e-2)
            self.assertAlmostEqual(stats["inter_distance"].item(), distance, delta=1e-3 * distance)
            self.assertAlmostEqual(stats["inter_distance_median"].item(), distance, delta=1e-3 * distance)
            self.assertEqual(stats["inter_too_close"].item(), close)
            self.assertEqual(stats["inter_too_far"].item(), far)

    def test_matches_ffrgs_reference_on_random_slots(self):
        generator = torch.Generator().manual_seed(0)
        centers = torch.randn(2, 16, 8, 3, generator=generator)
        centers = centers * 0.05 + torch.randn(2, 16, 1, 3, generator=generator) * 0.3
        scales = torch.rand(2, 16, 8, 3, generator=generator) * 0.05 + 0.01
        opacities = torch.rand(2, 16, 8, 1, generator=generator) * 0.9 + 0.05
        rotations = torch.nn.functional.normalize(torch.randn(2, 16, 8, 4, generator=generator), dim=-1)

        ref_w, ref_loss, ref_dist = _ffrgs_reference(centers, scales, rotations, opacities)
        weights, active = geo_losses._opacity_area_mass_weights(opacities, scales)
        result = compute_geo_losses(centers, scales, opacities, gs_rotations=rotations)

        self.assertTrue(active.all().item())
        torch.testing.assert_close(weights, ref_w, rtol=1e-5, atol=1e-7)
        # 唯一差异是对角 jitter（FFRGS 2e-6，我们约 1e-6）。
        torch.testing.assert_close(result["inter_loss"], ref_loss, rtol=2e-3, atol=1e-5)
        torch.testing.assert_close(
            result["stats"]["inter_distance_median"], ref_dist.median(), rtol=2e-3, atol=1e-5
        )

    def test_area_mass_keeps_opacity_activity_and_zero_transparent_mass(self):
        centers, scales, opacities, rotations = _structured_inputs()
        scales = scales.clone()
        scales[..., 1, :] *= 2.0
        opacities = torch.full_like(opacities, 0.5)
        opacities[0, 0, 3] = 0.0
        weights, active = geo_losses._opacity_area_mass_weights(opacities, scales)
        _, opacity_active = _opacity_mass_weights(opacities)
        torch.testing.assert_close(active, opacity_active)
        self.assertEqual(weights[0, 0, 3].item(), 0.0)
        # 体积放大 2^3 倍 → 质量放大 (2^3)^(2/3) = 4 倍。
        torch.testing.assert_close(weights[1, 2, 1] / weights[1, 2, 0], torch.tensor(4.0))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 3))

        tiny_scales = torch.full_like(scales, 1e-7)
        _, tiny_active = geo_losses._opacity_area_mass_weights(opacities, tiny_scales)
        torch.testing.assert_close(tiny_active, opacity_active)

    def test_invalid_modes_raise(self):
        inputs = _structured_inputs()
        with self.assertRaisesRegex(ValueError, "mass_weighting"):
            compute_geo_losses(*inputs[:3], gs_rotations=inputs[3], mass_weighting="area")
        with self.assertRaisesRegex(ValueError, "inter_form"):
            compute_geo_losses(*inputs[:3], gs_rotations=inputs[3], inter_form="log")


if __name__ == "__main__":
    unittest.main()
