import unittest

import torch

from active_adaptation.learning.hierarchical.root_student_force_moe import (
    EXPERT_NAMES,
    compose_decoded_expert_actions,
    inverse_stiffness_gate_weights,
)


class RootStudentForceMoETest(unittest.TestCase):
    def test_inverse_stiffness_gate_hits_all_anchors(self):
        stiffness = torch.tensor([[200.0, 400.0, 600.0]])
        weights = inverse_stiffness_gate_weights(stiffness)

        expected = torch.tensor(
            [[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]
        )
        torch.testing.assert_close(weights, expected)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(stiffness))

    def test_inverse_stiffness_gate_interpolates_in_compliance_space(self):
        stiffness = torch.tensor([[300.0, 500.0, 600.0]])
        weights = inverse_stiffness_gate_weights(stiffness)

        torch.testing.assert_close(weights[0, 0], torch.tensor([0.0, 2.0 / 3.0, 1.0 / 3.0]))
        torch.testing.assert_close(weights[0, 1], torch.tensor([0.6, 0.4, 0.0]))
        torch.testing.assert_close(weights[0, 2], torch.tensor([1.0, 0.0, 0.0]))

    def test_axis_component_composition_uses_matching_expert_per_hand(self):
        raw_actions = {name: torch.zeros(1, 6) for name in EXPERT_NAMES}
        raw_actions["x_200"][:, [0, 3]] = torch.atanh(torch.tensor(0.8))
        raw_actions["y_400"][:, [1, 4]] = torch.atanh(torch.tensor(0.4))
        raw_actions["z_200"][:, [2, 5]] = torch.atanh(torch.tensor(-0.3))
        weights = inverse_stiffness_gate_weights(torch.tensor([[200.0, 400.0, 200.0]]))

        raw_action, decoded_action = compose_decoded_expert_actions(raw_actions, weights)

        expected = torch.tensor([[0.8, 0.4, -0.3, 0.8, 0.4, -0.3]])
        torch.testing.assert_close(decoded_action, expected)
        torch.testing.assert_close(torch.tanh(raw_action), expected)

    def test_full_residual_recovers_single_axis_expert_at_anchor(self):
        raw_actions = {name: torch.zeros(1, 6) for name in EXPERT_NAMES}
        desired = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]])
        raw_actions["x_200"] = torch.atanh(desired)
        weights = inverse_stiffness_gate_weights(torch.tensor([[200.0, 600.0, 600.0]]))

        raw_action, decoded_action = compose_decoded_expert_actions(
            raw_actions,
            weights,
            mode="full_residual",
        )

        torch.testing.assert_close(decoded_action, desired)
        torch.testing.assert_close(torch.tanh(raw_action), desired)


if __name__ == "__main__":
    unittest.main()
