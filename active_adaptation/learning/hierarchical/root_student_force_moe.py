from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import hydra
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl.envs.transforms import VecNorm
from torchrl.envs.utils import ExplorationType, set_exploration_type

import active_adaptation as aa
from active_adaptation.utils.wandb import parse_checkpoint_path


EXPERT_NAMES = (
    "baseline_600",
    "x_400",
    "x_200",
    "y_400",
    "y_200",
    "z_400",
    "z_200",
)
AXIS_EXPERTS = (
    ("x_400", "x_200"),
    ("y_400", "y_200"),
    ("z_400", "z_200"),
)


def inverse_stiffness_gate_weights(
    stiffness: torch.Tensor,
    anchors: Sequence[float] = (200.0, 400.0, 600.0),
) -> torch.Tensor:
    """Return per-axis analytical MoE weights in [600, 400, 200] order.

    The interpolation is piecewise linear in compliance (1 / stiffness),
    matching the force-over-stiffness target used by the EE compliance task.
    ``stiffness`` must end in xyz and the result ends in ``xyz x 3``.
    """

    if stiffness.shape[-1] != 3:
        raise ValueError(
            "Analytical EE MoE expects stiffness with three xyz values, "
            f"got shape {tuple(stiffness.shape)}."
        )
    if len(anchors) != 3:
        raise ValueError(f"Expected [low, middle, high] stiffness anchors, got {anchors}.")

    low, middle, high = (float(value) for value in anchors)
    if not 0.0 < low < middle < high:
        raise ValueError(
            "Analytical EE MoE stiffness anchors must satisfy "
            f"0 < low < middle < high, got {anchors}."
        )
    if not torch.isfinite(stiffness).all():
        raise ValueError("Analytical EE MoE stiffness contains non-finite values.")

    stiffness = stiffness.clamp(min=low, max=high)
    compliance = stiffness.reciprocal()
    compliance_low = 1.0 / low
    compliance_middle = 1.0 / middle
    compliance_high = 1.0 / high

    lower_segment = stiffness <= middle
    weight_200_lower = (
        (compliance - compliance_middle) / (compliance_low - compliance_middle)
    ).clamp(0.0, 1.0)
    weight_400_lower = 1.0 - weight_200_lower

    weight_400_upper = (
        (compliance - compliance_high) / (compliance_middle - compliance_high)
    ).clamp(0.0, 1.0)
    weight_600_upper = 1.0 - weight_400_upper

    weight_600 = torch.where(lower_segment, torch.zeros_like(stiffness), weight_600_upper)
    weight_400 = torch.where(lower_segment, weight_400_lower, weight_400_upper)
    weight_200 = torch.where(lower_segment, weight_200_lower, torch.zeros_like(stiffness))
    return torch.stack((weight_600, weight_400, weight_200), dim=-1)


def compose_decoded_expert_actions(
    raw_actions: Mapping[str, torch.Tensor],
    gate_weights: torch.Tensor,
    *,
    mode: str = "axis_component",
    action_limit: float = 0.999,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose frozen expert actions and return ``(raw, tanh-space)`` actions."""

    missing = [name for name in EXPERT_NAMES if name not in raw_actions]
    if missing:
        raise KeyError(f"Missing analytical EE MoE expert actions: {missing}.")
    if gate_weights.shape[-2:] != (3, 3):
        raise ValueError(
            "Analytical EE MoE gate weights must end in [xyz, 600/400/200], "
            f"got shape {tuple(gate_weights.shape)}."
        )
    if not 0.0 < action_limit < 1.0:
        raise ValueError(f"action_limit must be in (0, 1), got {action_limit}.")

    decoded = {name: torch.tanh(raw_actions[name]) for name in EXPERT_NAMES}
    baseline = decoded["baseline_600"]
    if baseline.shape[-1] != 6:
        raise ValueError(
            "Analytical EE MoE requires the 6D [left xyz, right xyz] high-level action, "
            f"got action shape {tuple(baseline.shape)}."
        )
    for name, action in decoded.items():
        if action.shape != baseline.shape:
            raise ValueError(
                f"Expert {name!r} action shape {tuple(action.shape)} does not match "
                f"baseline shape {tuple(baseline.shape)}."
            )

    if mode == "axis_component":
        combined = baseline.clone()
        for axis, (middle_name, low_name) in enumerate(AXIS_EXPERTS):
            indices = torch.tensor((axis, axis + 3), device=baseline.device)
            weights = gate_weights[..., axis, :]
            axis_action = (
                weights[..., 0:1] * baseline.index_select(-1, indices)
                + weights[..., 1:2] * decoded[middle_name].index_select(-1, indices)
                + weights[..., 2:3] * decoded[low_name].index_select(-1, indices)
            )
            combined[..., indices] = axis_action
    elif mode == "full_residual":
        combined = baseline.clone()
        for axis, (middle_name, low_name) in enumerate(AXIS_EXPERTS):
            weights = gate_weights[..., axis, :]
            combined = combined + (
                weights[..., 1:2] * (decoded[middle_name] - baseline)
                + weights[..., 2:3] * (decoded[low_name] - baseline)
            )
    else:
        raise ValueError(
            "Analytical EE MoE composition mode must be 'axis_component' or "
            f"'full_residual', got {mode!r}."
        )

    combined = combined.clamp(min=-action_limit, max=action_limit)
    raw_combined = torch.atanh(combined)
    return raw_combined, combined


class FrozenHighLevelExpert:
    """Deploy-only high-level student policy with checkpoint-specific VecNorm."""

    def __init__(
        self,
        name: str,
        cfg: Any,
        observation_spec,
        action_spec,
        reward_spec,
        *,
        device: torch.device,
        env,
    ):
        self.name = name
        self.cfg = cfg
        self.device = device
        checkpoint_path = self._resolve_checkpoint_path(cfg)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

        checkpoint_cfg = state_dict.get("cfg")
        if checkpoint_cfg is None:
            raise ValueError(f"Expert {name!r} checkpoint {checkpoint_path} does not contain cfg.")
        if "policy" not in state_dict:
            raise ValueError(f"Expert {name!r} checkpoint {checkpoint_path} does not contain policy.")

        policy_state = state_dict["policy"]
        last_phase = policy_state.get("last_phase", None)
        require_student = bool(cfg.get("require_student_checkpoint", True))
        if require_student and last_phase == "train":
            raise ValueError(
                f"Expert {name!r} is a teacher/train checkpoint. Use its adapt or finetune checkpoint."
            )

        self._validate_observation_dim(name, state_dict, observation_spec)
        algo_cfg = OmegaConf.create(
            OmegaConf.to_container(checkpoint_cfg.algo, resolve=True)
        )
        OmegaConf.set_struct(algo_cfg, False)
        if last_phase in {"adapt", "finetune"}:
            algo_cfg.phase = last_phase
        else:
            algo_cfg.phase = "adapt"
        # Frozen experts are replicated inference modules inside the outer
        # MoE. They must not create their own nested DDP wrappers; only the
        # learned gate/critic (if present) participate in distributed updates.
        algo_cfg.disable_ddp = True

        policy_cls = hydra.utils.get_class(algo_cfg._target_)
        self.policy = policy_cls(
            algo_cfg,
            observation_spec,
            action_spec,
            reward_spec,
            device=device,
            env=env,
        )
        self.policy.load_state_dict(
            policy_state,
            strict=bool(cfg.get("strict", True)),
        )
        self.policy.requires_grad_(False)
        self.policy.eval()
        self.rollout_policy = self.policy.get_rollout_policy("eval")
        self.obs_norm = self._build_obs_norm(name, state_dict, observation_spec, cfg)

    @staticmethod
    def _resolve_checkpoint_path(cfg: Any) -> str:
        checkpoint_path = cfg.get("checkpoint_path", None)
        run_path = cfg.get("run_path", None)
        checkpoint_iteration = cfg.get("checkpoint_iteration", None)
        spec = None
        if checkpoint_path:
            spec = str(checkpoint_path)
        elif run_path:
            spec = f"run:{run_path}"
            if checkpoint_iteration is not None:
                spec += f":{int(checkpoint_iteration)}"
        if spec is None:
            raise ValueError(
                "Each analytical EE MoE expert needs checkpoint_path or run_path."
            )

        # All ranks share the same filesystem. Download each W&B checkpoint
        # once on rank zero, then publish its resolved local path to the other
        # workers before they construct their frozen copy of the expert.
        if aa.is_distributed():
            resolved = parse_checkpoint_path(spec) if aa.is_main_process() else None
            resolved_list = [resolved]
            dist.broadcast_object_list(resolved_list, src=0)
            dist.barrier()
            return resolved_list[0]
        return parse_checkpoint_path(spec)

    @staticmethod
    def _checkpoint_obs_dim(state_dict: Mapping[str, Any], key: str) -> int | None:
        vecnorm_state = state_dict.get("vecnorm", {})
        extra_state = vecnorm_state.get("_extra_state", {})
        value = extra_state.get(f"{key}_sum")
        return int(value.shape[-1]) if value is not None else None

    @classmethod
    def _validate_observation_dim(cls, name, state_dict, observation_spec) -> None:
        checkpoint_dim = cls._checkpoint_obs_dim(state_dict, "hl_policy")
        if checkpoint_dim is None:
            return
        current_dim = int(observation_spec["hl_policy"].shape[-1])
        if checkpoint_dim != current_dim:
            raise ValueError(
                f"Expert {name!r} expects hl_policy dim {checkpoint_dim}, but the MoE task "
                f"provides {current_dim}. Keep stiffness in the separate hl_moe observation group."
            )

    @staticmethod
    def _build_obs_norm(name, state_dict, observation_spec, cfg):
        if not bool(cfg.get("use_vecnorm", True)):
            return None
        # Match the direct checkpoint deployment path.  Some adapt
        # checkpoints contain a dummy VecNorm state (scale ~= 0.01) even
        # though the saved config explicitly disabled VecNorm.  Applying that
        # state inside the MoE would amplify raw observations by ~100x and
        # produce a policy trajectory different from direct checkpoint eval.
        checkpoint_cfg = state_dict.get("cfg")
        if checkpoint_cfg is not None and "vecnorm" in checkpoint_cfg:
            if checkpoint_cfg.get("vecnorm") is None:
                print(
                    f"[MoE] Expert {name}: checkpoint vecnorm is disabled; "
                    "using raw hl_policy observations."
                )
                return None
        if "vecnorm" not in state_dict:
            if bool(cfg.get("require_vecnorm", True)):
                raise ValueError(f"Expert {name!r} checkpoint does not contain VecNorm state.")
            return None

        checkpoint_extra_state = state_dict["vecnorm"].get("_extra_state", {})
        if "hl_policy_sum" not in checkpoint_extra_state:
            if bool(cfg.get("require_vecnorm", True)):
                raise ValueError(f"Expert {name!r} VecNorm has no hl_policy statistics.")
            return None

        vecnorm = VecNorm(["hl_policy"], decay=0.9999)
        vecnorm(observation_spec.zero())
        filtered_extra_state = {
            stat_key: value
            for stat_key, value in checkpoint_extra_state.items()
            if stat_key.startswith("hl_policy_")
        }
        vecnorm.load_state_dict({"_extra_state": filtered_extra_state})
        return vecnorm.to_observation_norm().to(observation_spec.device)

    @torch.inference_mode()
    def act(self, tensordict: TensorDictBase) -> TensorDictBase:
        input_keys = list(self.rollout_policy.in_keys)
        expert_td = tensordict.select(*input_keys, "is_init", strict=False).clone()
        if self.obs_norm is not None:
            self.obs_norm(expert_td)
        with set_exploration_type(ExplorationType.MODE):
            self.rollout_policy(expert_td)
        return expert_td


class RootStudentForceAnalyticalMoEPolicy(TensorDictModuleBase):
    """Frozen, training-free factorized MoE over seven high-level EE experts."""

    def __init__(
        self,
        cfg,
        observation_spec,
        action_spec,
        reward_spec,
        device: str = "cuda:0",
        env=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.in_keys = list(cfg.get("in_keys", ["hl_policy", "hl_moe"]))
        self.out_keys = ["action"]
        self.stiffness_key = str(cfg.get("stiffness_key", "hl_moe"))
        self.composition = str(cfg.get("composition", "axis_component"))
        self.action_limit = float(cfg.get("action_limit", 0.999))
        self.anchors = tuple(float(value) for value in cfg.get("stiffness_anchors", [200.0, 400.0, 600.0]))

        if self.stiffness_key not in observation_spec.keys(True, True):
            raise KeyError(
                f"Analytical EE MoE stiffness observation {self.stiffness_key!r} is missing."
            )

        experts_cfg = cfg.get("experts", {})
        missing = [name for name in EXPERT_NAMES if name not in experts_cfg]
        if missing:
            raise ValueError(f"Analytical EE MoE config is missing experts: {missing}.")

        self.experts = {
            name: FrozenHighLevelExpert(
                name,
                experts_cfg[name],
                observation_spec,
                action_spec,
                reward_spec,
                device=self.device,
                env=env,
            )
            for name in EXPERT_NAMES
        }
        self.last_gate_weights = None
        self.last_decoded_action = None

    def _stiffness_from_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != 3:
            raise ValueError(
                f"{self.stiffness_key!r} must contain normalized [kx, ky, kz], "
                f"got shape {tuple(observation.shape)}."
            )
        low, _, high = self.anchors
        normalized = observation.clamp(-1.0, 1.0)
        return low + 0.5 * (normalized + 1.0) * (high - low)

    @torch.inference_mode()
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        expert_outputs = {
            name: expert.act(tensordict)
            for name, expert in self.experts.items()
        }
        raw_actions = {
            name: output["action"]
            for name, output in expert_outputs.items()
        }

        stiffness = self._stiffness_from_observation(tensordict[self.stiffness_key])
        gate_weights = inverse_stiffness_gate_weights(stiffness, self.anchors)
        raw_action, decoded_action = compose_decoded_expert_actions(
            raw_actions,
            gate_weights,
            mode=self.composition,
            action_limit=self.action_limit,
        )
        tensordict["action"] = raw_action
        tensordict["moe_gate_weights"] = gate_weights.flatten(-2)

        baseline_output = expert_outputs["baseline_600"]
        if "direct_priv_pred" in baseline_output.keys():
            # Preserve existing force-estimator diagnostics using the baseline
            # expert. The composed policy itself does not train a new estimator.
            tensordict["direct_priv_pred"] = baseline_output["direct_priv_pred"]

        self.last_gate_weights = gate_weights
        self.last_decoded_action = decoded_action
        return tensordict

    def make_tensordict_primer(self):
        return None

    def get_rollout_policy(self, mode: str = "eval"):
        return self

    def broadcast_parameters(self, extra_modules=()):
        return None

    def step_schedule(self, progress: float, iter: int):
        return None

    def train_op(self, td, vecnorm):
        raise RuntimeError(
            "RootStudentForceAnalyticalMoEPolicy is frozen and evaluation-only; "
            "it has no trainable gate."
        )
