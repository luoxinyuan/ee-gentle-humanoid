import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tensordict import TensorDict

from active_adaptation.learning.hierarchical.root_student_force_learned_moe import (
    ANALYTICAL_GATE_WEIGHTS_KEY,
    EXPERT_ACTIONS_KEY,
    GATE_WEIGHTS_KEY,
    REUSE_EXPERT_ACTIONS_KEY,
    AnalyticalResidualGate,
    LearnedMoEActorCore,
    RootStudentForceLearnedMoEPolicy,
    analytical_gate_kl,
    corrected_analytical_gate_weights,
)
from active_adaptation.learning.hierarchical.root_student_force_moe import (
    EXPERT_NAMES,
    inverse_stiffness_gate_weights,
)


class _ObservationSpec:
    device = torch.device("cpu")

    def keys(self, *args, **kwargs):
        return {"hl_policy", "hl_moe", "hl_priv", "hl_force_priv"}

    def zero(self):
        return TensorDict(
            {
                "hl_policy": torch.zeros(4),
                "hl_moe": torch.zeros(3),
                "hl_priv": torch.zeros(2),
                "hl_force_priv": torch.zeros(6),
            },
            batch_size=[],
        )


class _FakeExpert:
    calls = 0

    def __init__(self, name, *args, **kwargs):
        self.name = name

    def act(self, tensordict):
        type(self).calls += 1
        expert_index = EXPERT_NAMES.index(self.name)
        action = torch.full(
            (*tensordict.batch_size, 6),
            0.05 * expert_index,
        ) + tensordict["hl_policy"][..., 0:1]
        return TensorDict(
            {
                "action": action,
                "direct_priv_pred": torch.zeros(*tensordict.batch_size, 6),
            },
            batch_size=tensordict.batch_size,
        )


class RootStudentForceLearnedMoETest(unittest.TestCase):
    def test_zero_correction_recovers_analytical_weights_and_anchors(self):
        stiffness = torch.tensor(
            [[200.0, 400.0, 600.0], [300.0, 500.0, 600.0]]
        )
        analytical = inverse_stiffness_gate_weights(stiffness)
        learned = corrected_analytical_gate_weights(
            analytical, torch.zeros_like(analytical)
        )

        torch.testing.assert_close(learned, analytical)
        torch.testing.assert_close(
            learned[0],
            torch.tensor(
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
            ),
        )

    def test_correction_cannot_activate_nonadjacent_expert(self):
        analytical = inverse_stiffness_gate_weights(
            torch.tensor([[300.0, 500.0, 300.0]])
        )
        corrections = torch.tensor(
            [[[100.0, -2.0, 2.0], [-2.0, 2.0, 100.0], [100.0, -2.0, 2.0]]]
        )
        learned = corrected_analytical_gate_weights(analytical, corrections)

        self.assertEqual(learned[0, 0, 0].item(), 0.0)
        self.assertEqual(learned[0, 1, 2].item(), 0.0)
        self.assertEqual(learned[0, 2, 0].item(), 0.0)
        torch.testing.assert_close(learned.sum(dim=-1), torch.ones(1, 3))

    def test_gate_is_zero_initialized_and_prior_kl_is_zero(self):
        gate = AnalyticalResidualGate(
            45,
            6,
            [32, 16],
            init_noise_scale=0.2,
            max_logit_correction=4.0,
            layer_norm="before",
            train_action_std=True,
        )
        corrections, scale = gate(torch.randn(5, 45))
        analytical = inverse_stiffness_gate_weights(
            torch.tensor([[300.0, 500.0, 600.0]]).expand(5, -1)
        )
        learned = corrected_analytical_gate_weights(analytical, corrections)

        torch.testing.assert_close(corrections, torch.zeros_like(corrections))
        torch.testing.assert_close(scale, torch.full((5, 6), 0.2))
        torch.testing.assert_close(learned, analytical)
        torch.testing.assert_close(
            analytical_gate_kl(analytical, learned), torch.zeros(5, 3)
        )

    def test_actor_core_refreshes_rollout_and_reuses_only_when_explicit(self):
        _FakeExpert.calls = 0
        cfg = {
            "in_keys": ["hl_policy", "hl_moe"],
            "experts": {name: {} for name in EXPERT_NAMES},
            "gate_hidden_dims": [16, 16],
        }
        action_spec = SimpleNamespace(shape=(6,))

        with patch(
            "active_adaptation.learning.hierarchical.root_student_force_learned_moe.FrozenHighLevelExpert",
            _FakeExpert,
        ):
            actor = LearnedMoEActorCore(
                cfg,
                _ObservationSpec(),
                action_spec,
                reward_spec=None,
                device=torch.device("cpu"),
                env=None,
            )

        tensordict = TensorDict(
            {
                "hl_policy": torch.zeros(2, 4),
                "hl_moe": torch.tensor([[-0.5, 0.5, 1.0], [-1.0, 0.0, 1.0]]),
                "is_init": torch.zeros(2, 1, dtype=torch.bool),
            },
            batch_size=[2],
        )
        actor(tensordict)
        self.assertEqual(_FakeExpert.calls, len(EXPERT_NAMES))
        self.assertIn(EXPERT_ACTIONS_KEY, tensordict.keys())
        torch.testing.assert_close(
            tensordict[GATE_WEIGHTS_KEY],
            tensordict[ANALYTICAL_GATE_WEIGHTS_KEY],
        )
        first_actions = tensordict[EXPERT_ACTIONS_KEY].clone()

        # The cached output key survives in a rollout TensorDict, but without
        # an explicit PPO reuse marker it must be overwritten with fresh
        # expert inference for the new observation.
        tensordict["hl_policy"][..., 0] = 0.1
        actor(tensordict)
        self.assertEqual(_FakeExpert.calls, 2 * len(EXPERT_NAMES))
        self.assertFalse(torch.equal(first_actions, tensordict[EXPERT_ACTIONS_KEY]))
        refreshed_actions = tensordict[EXPERT_ACTIONS_KEY].clone()

        tensordict[REUSE_EXPERT_ACTIONS_KEY] = torch.ones(
            2, 1, dtype=torch.bool
        )
        tensordict["hl_policy"][..., 0] = 0.2
        actor(tensordict)
        self.assertEqual(_FakeExpert.calls, 2 * len(EXPERT_NAMES))
        torch.testing.assert_close(
            tensordict[EXPERT_ACTIONS_KEY], refreshed_actions
        )

    def test_stiffness_only_gate_uses_three_command_features(self):
        cfg = {
            "in_keys": ["hl_policy", "hl_moe"],
            "experts": {name: {} for name in EXPERT_NAMES},
            "gate_hidden_dims": [16, 16],
            "gate_context": "stiffness_only",
        }
        action_spec = SimpleNamespace(shape=(6,))
        with patch(
            "active_adaptation.learning.hierarchical.root_student_force_learned_moe.FrozenHighLevelExpert",
            _FakeExpert,
        ):
            actor = LearnedMoEActorCore(
                cfg,
                _ObservationSpec(),
                action_spec,
                reward_spec=None,
                device=torch.device("cpu"),
                env=None,
            )

        self.assertEqual(actor.gate.network[0].in_features, 3)

    def test_one_ppo_minibatch_updates_gate_but_not_experts(self):
        _FakeExpert.calls = 0
        cfg = {
            "in_keys": ["hl_policy", "hl_moe"],
            "experts": {name: {} for name in EXPERT_NAMES},
            "gate_hidden_dims": [16, 16],
            "critic_hidden_dims": [16, 8],
            "critic_in_keys": ["hl_policy", "hl_moe", "hl_priv", "hl_force_priv"],
            "prior_coef_start": 0.0,
            "prior_coef_end": 0.0,
        }
        action_spec = SimpleNamespace(shape=(6,))
        with patch(
            "active_adaptation.learning.hierarchical.root_student_force_learned_moe.FrozenHighLevelExpert",
            _FakeExpert,
        ):
            policy = RootStudentForceLearnedMoEPolicy(
                cfg,
                _ObservationSpec(),
                action_spec,
                reward_spec=None,
                device="cpu",
                env=None,
            )

        batch_size = 8
        tensordict = TensorDict(
            {
                "hl_policy": torch.zeros(batch_size, 4),
                "hl_moe": torch.full((batch_size, 3), -0.5),
                "hl_priv": torch.zeros(batch_size, 2),
                "hl_force_priv": torch.zeros(batch_size, 6),
                "is_init": torch.zeros(batch_size, 1, dtype=torch.bool),
            },
            batch_size=[batch_size],
        )
        policy.get_rollout_policy("train")(tensordict)
        tensordict = tensordict.detach()
        distribution = torch.distributions.Independent(
            torch.distributions.Normal(tensordict["loc"], tensordict["scale"]),
            1,
        )
        action = tensordict["loc"] + 0.1
        tensordict["action"] = action
        tensordict["sample_log_prob"] = distribution.log_prob(action)
        tensordict["adv"] = torch.ones(batch_size, 1)
        tensordict["ret"] = policy.critic(tensordict.clone())["state_value"].detach()

        actor = policy._unwrap(policy.actor)
        output_layer = actor.module[0].gate.network[-1]
        before = output_layer.weight.detach().clone()
        policy._update(tensordict)

        self.assertEqual(_FakeExpert.calls, len(EXPERT_NAMES))
        self.assertFalse(torch.equal(before, output_layer.weight.detach()))


if __name__ == "__main__":
    unittest.main()
