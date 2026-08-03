from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictModuleBase, TensorDictSequential as Seq
from torch.nn.parallel import DistributedDataParallel as DDP
from torchrl.modules import ProbabilisticActor

import active_adaptation as aa
from active_adaptation.learning.hierarchical.root_student_force_moe import (
    EXPERT_NAMES,
    FrozenHighLevelExpert,
    compose_decoded_expert_actions,
    inverse_stiffness_gate_weights,
)
from active_adaptation.learning.modules.distributions import IndependentNormal
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    DONE_KEY,
    REWARD_KEY,
    TERM_KEY,
    CatTensors,
    GAE,
    make_batch,
    make_mlp,
)


EXPERT_ACTIONS_KEY = "moe_expert_raw_actions"
REUSE_EXPERT_ACTIONS_KEY = "_moe_reuse_expert_raw_actions"
GATE_WEIGHTS_KEY = "moe_gate_weights"
ANALYTICAL_GATE_WEIGHTS_KEY = "moe_analytical_gate_weights"


def corrected_analytical_gate_weights(
    analytical_weights: torch.Tensor,
    corrections: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply learned logit corrections without crossing stiffness segments.

    Analytical weights are piecewise supported on adjacent stiffness anchors.
    Keeping zero-weight anchors masked preserves exact one-hot behavior at
    200/400/600 and prevents the learned gate from skipping across an anchor.
    """

    if analytical_weights.shape != corrections.shape:
        raise ValueError(
            "Analytical weights and learned corrections must have the same shape, "
            f"got {tuple(analytical_weights.shape)} and {tuple(corrections.shape)}."
        )
    if analytical_weights.shape[-2:] != (3, 3):
        raise ValueError(
            "Learned EE MoE gate tensors must end in [xyz, 600/400/200], "
            f"got {tuple(analytical_weights.shape)}."
        )
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")

    support = analytical_weights > 0.0
    logits = analytical_weights.clamp_min(eps).log() + corrections
    logits = logits.masked_fill(~support, -torch.inf)
    return torch.softmax(logits, dim=-1)


def analytical_gate_kl(
    analytical_weights: torch.Tensor,
    learned_weights: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return KL(analytical || learned), summed over anchors per xyz axis."""

    analytical = analytical_weights.clamp_min(0.0)
    learned = learned_weights.clamp_min(eps)
    terms = torch.where(
        analytical > 0.0,
        analytical * (analytical.clamp_min(eps).log() - learned.log()),
        torch.zeros_like(analytical),
    )
    return terms.sum(dim=-1)


class AnalyticalResidualGate(nn.Module):
    """Small gate initialized as an exact correction of zero."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        *,
        init_noise_scale: float,
        max_logit_correction: float,
        layer_norm: str | None,
        train_action_std: bool,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or action_dim <= 0:
            raise ValueError("Gate input and action dimensions must be positive.")
        if init_noise_scale <= 0.0:
            raise ValueError("init_noise_scale must be positive.")
        if max_logit_correction <= 0.0:
            raise ValueError("max_logit_correction must be positive.")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            layers.append(nn.Linear(previous_dim, hidden_dim))
            if layer_norm == "before":
                layers.extend((nn.LayerNorm(hidden_dim), nn.Mish()))
            elif layer_norm == "after":
                layers.extend((nn.Mish(), nn.LayerNorm(hidden_dim)))
            elif layer_norm is None:
                layers.append(nn.Mish())
            else:
                raise ValueError(
                    "layer_norm must be 'before', 'after', or null, "
                    f"got {layer_norm!r}."
                )
            previous_dim = hidden_dim

        output = nn.Linear(previous_dim, 9)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)
        self.max_logit_correction = float(max_logit_correction)

        initial_log_std = torch.full((action_dim,), math.log(init_noise_scale))
        if train_action_std:
            self.log_std = nn.Parameter(initial_log_std)
        else:
            self.register_buffer("log_std", initial_log_std)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        corrections = self.max_logit_correction * torch.tanh(self.network(context))
        scale = self.log_std.exp().expand(*context.shape[:-1], self.log_std.numel())
        return corrections.unflatten(-1, (3, 3)), scale


class LearnedMoEActorCore(TensorDictModuleBase):
    """Evaluate frozen experts, apply the learned gate, and emit Normal params."""

    def __init__(
        self,
        cfg,
        observation_spec,
        action_spec,
        reward_spec,
        *,
        device: torch.device,
        env,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.in_keys = list(cfg.get("in_keys", ["hl_policy", "hl_moe"]))
        self.out_keys = [
            "loc",
            "scale",
            EXPERT_ACTIONS_KEY,
            GATE_WEIGHTS_KEY,
            ANALYTICAL_GATE_WEIGHTS_KEY,
        ]
        self.stiffness_key = str(cfg.get("stiffness_key", "hl_moe"))
        self.composition = str(cfg.get("composition", "full_residual"))
        self.action_limit = float(cfg.get("action_limit", 0.999))
        self.anchors = tuple(
            float(value)
            for value in cfg.get("stiffness_anchors", [200.0, 400.0, 600.0])
        )
        self.prior_eps = float(cfg.get("analytical_prior_eps", 1e-6))
        self.gate_context_mode = str(
            cfg.get("gate_context", "stiffness_and_expert_actions")
        )
        self.action_dim = int(action_spec.shape[-1])
        if self.action_dim != 6:
            raise ValueError(
                "Learned EE MoE requires the 6D [left xyz, right xyz] high-level action, "
                f"got {self.action_dim}."
            )
        if self.stiffness_key not in observation_spec.keys(True, True):
            raise KeyError(
                f"Learned EE MoE stiffness observation {self.stiffness_key!r} is missing."
            )

        experts_cfg = cfg.get("experts", {})
        missing = [name for name in EXPERT_NAMES if name not in experts_cfg]
        if missing:
            raise ValueError(f"Learned EE MoE config is missing experts: {missing}.")
        self.experts = {
            name: FrozenHighLevelExpert(
                name,
                experts_cfg[name],
                observation_spec,
                action_spec,
                reward_spec,
                device=device,
                env=env,
            )
            for name in EXPERT_NAMES
        }

        if self.gate_context_mode == "stiffness_only":
            gate_input_dim = 3
        elif self.gate_context_mode == "stiffness_and_expert_actions":
            gate_input_dim = 3 + len(EXPERT_NAMES) * self.action_dim
        else:
            raise ValueError(
                "gate_context must be 'stiffness_only' or "
                f"'stiffness_and_expert_actions', got {self.gate_context_mode!r}."
            )
        self.gate = AnalyticalResidualGate(
            gate_input_dim,
            self.action_dim,
            cfg.get("gate_hidden_dims", [128, 128]),
            init_noise_scale=float(cfg.get("init_noise_scale", 0.2)),
            max_logit_correction=float(cfg.get("max_logit_correction", 4.0)),
            layer_norm=cfg.get("layer_norm", "before"),
            train_action_std=bool(cfg.get("train_action_std", True)),
        ).to(device)

    def _stiffness_from_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != 3:
            raise ValueError(
                f"{self.stiffness_key!r} must contain normalized [kx, ky, kz], "
                f"got shape {tuple(observation.shape)}."
            )
        low, _, high = self.anchors
        normalized = observation.clamp(-1.0, 1.0)
        return low + 0.5 * (normalized + 1.0) * (high - low)

    @staticmethod
    def _unpack_expert_actions(flat_actions: torch.Tensor) -> dict[str, torch.Tensor]:
        if flat_actions.shape[-1] != len(EXPERT_NAMES) * 6:
            raise ValueError(
                f"Cached learned-MoE expert actions must have {len(EXPERT_NAMES) * 6} "
                f"values, got shape {tuple(flat_actions.shape)}."
            )
        chunks = flat_actions.split(6, dim=-1)
        return dict(zip(EXPERT_NAMES, chunks))

    def _evaluate_experts(
        self, tensordict: TensorDict
    ) -> tuple[dict[str, torch.Tensor], Mapping[str, TensorDict]]:
        expert_outputs = {
            name: expert.act(tensordict)
            for name, expert in self.experts.items()
        }
        # Expert inference uses inference-mode tensors. Clone after inference so
        # the gate can safely save the bounded action context for backprop.
        raw_actions = {
            name: output[ACTION_KEY].detach().clone()
            for name, output in expert_outputs.items()
        }
        return raw_actions, expert_outputs

    def forward(self, tensordict: TensorDict) -> TensorDict:
        # TorchRL carries unknown root TensorDict keys across environment
        # steps. Therefore, the mere presence of EXPERT_ACTIONS_KEY does not
        # mean that it belongs to the current observation. Rollout/eval always
        # refreshes all experts; only PPO update minibatches explicitly opt in
        # to reusing the actions saved for that same transition.
        reuse_flag = tensordict.get(REUSE_EXPERT_ACTIONS_KEY, None)
        reuse_expert_actions = (
            reuse_flag is not None and bool(reuse_flag.all().item())
        )
        if reuse_expert_actions:
            if EXPERT_ACTIONS_KEY not in tensordict.keys():
                raise RuntimeError(
                    "PPO requested cached learned-MoE expert actions, but the "
                    f"minibatch has no {EXPERT_ACTIONS_KEY!r}."
                )
            raw_actions = self._unpack_expert_actions(tensordict[EXPERT_ACTIONS_KEY])
            expert_outputs = None
        else:
            raw_actions, expert_outputs = self._evaluate_experts(tensordict)
            tensordict[EXPERT_ACTIONS_KEY] = torch.cat(
                [raw_actions[name] for name in EXPERT_NAMES], dim=-1
            )

        normalized_stiffness = tensordict[self.stiffness_key].clamp(-1.0, 1.0)
        if self.gate_context_mode == "stiffness_only":
            gate_context = normalized_stiffness
        else:
            decoded_actions = [torch.tanh(raw_actions[name]) for name in EXPERT_NAMES]
            gate_context = torch.cat([normalized_stiffness, *decoded_actions], dim=-1)
        corrections, scale = self.gate(gate_context)

        stiffness = self._stiffness_from_observation(normalized_stiffness)
        analytical_weights = inverse_stiffness_gate_weights(stiffness, self.anchors)
        gate_weights = corrected_analytical_gate_weights(
            analytical_weights,
            corrections,
            eps=self.prior_eps,
        )
        raw_action, _ = compose_decoded_expert_actions(
            raw_actions,
            gate_weights,
            mode=self.composition,
            action_limit=self.action_limit,
        )

        tensordict["loc"] = raw_action
        tensordict["scale"] = scale
        tensordict[GATE_WEIGHTS_KEY] = gate_weights.flatten(-2)
        tensordict[ANALYTICAL_GATE_WEIGHTS_KEY] = analytical_weights.flatten(-2)

        if expert_outputs is not None:
            baseline_output = expert_outputs["baseline_600"]
            if "direct_priv_pred" in baseline_output.keys():
                tensordict["direct_priv_pred"] = baseline_output["direct_priv_pred"]
        return tensordict


class RootStudentForceLearnedMoEPolicy(TensorDictModuleBase):
    """PPO-trained gate over seven frozen deployable high-level experts."""

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
        self.out_keys = [ACTION_KEY]
        self.entropy_coef = float(cfg.get("entropy_coef_start", 0.001))
        self.prior_coef = float(cfg.get("prior_coef_start", 0.1))
        self.clip_param = float(cfg.get("clip_param", 0.2))
        self.gae = GAE(0.99, 0.95)
        self.num_minibatches = int(cfg.get("num_minibatches", 8))
        self.progress = 0.0
        self.current_lr = float(cfg.get("lr", 1e-4))
        self.num_updates = 0

        actor_core = LearnedMoEActorCore(
            cfg,
            observation_spec,
            action_spec,
            reward_spec,
            device=self.device,
            env=env,
        )
        self.actor = ProbabilisticActor(
            module=actor_core,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True,
        ).to(self.device)

        critic_in_keys = list(
            cfg.get(
                "critic_in_keys",
                ["hl_policy", "hl_moe", "hl_priv", "hl_force_priv"],
            )
        )
        self.critic = Seq(
            CatTensors(critic_in_keys, "_critic_inp", del_keys=False, sort=False),
            Mod(
                nn.Sequential(
                    make_mlp(
                        cfg.get("critic_hidden_dims", [512, 256]),
                        norm=cfg.get("layer_norm", "before"),
                    ),
                    nn.LazyLinear(1),
                ),
                ["_critic_inp"],
                ["state_value"],
            ),
        ).to(self.device)

        fake_td = observation_spec.zero().to(self.device)
        fake_td["is_init"] = torch.ones(
            *fake_td.batch_size, 1, dtype=torch.bool, device=self.device
        )
        self.critic(fake_td)

        self.world_size = 1
        if aa.is_distributed():
            self.world_size = aa.get_world_size()
            ddp_kwargs = dict(
                device_ids=[aa.get_local_rank()],
                output_device=aa.get_local_rank(),
                broadcast_buffers=True,
                find_unused_parameters=False,
            )
            self.actor = DDP(self.actor, **ddp_kwargs)
            self.critic = DDP(self.critic, **ddp_kwargs)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.current_lr)
        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(cfg.get("critic_lr", 3e-4)),
        )

    @staticmethod
    def _unwrap(module):
        return module.module if isinstance(module, DDP) else module

    def make_tensordict_primer(self):
        return None

    def get_rollout_policy(self, mode: str = "train"):
        return Seq(self.actor)

    def broadcast_parameters(self, extra_modules=()):
        if not aa.is_distributed():
            return
        for module in (self.actor, self.critic, *extra_modules):
            for parameter in module.parameters():
                dist.broadcast(parameter, src=0)
            for buffer in module.buffers():
                dist.broadcast(buffer, src=0)

    def step_schedule(self, progress: float, iter: int):
        self.progress = progress
        entropy_start = float(self.cfg.get("entropy_coef_start", 0.001))
        entropy_end = float(self.cfg.get("entropy_coef_end", 0.0001))
        prior_start = float(self.cfg.get("prior_coef_start", 0.1))
        prior_end = float(self.cfg.get("prior_coef_end", 0.005))
        self.entropy_coef = self._geometric_schedule(entropy_start, entropy_end, progress)
        self.prior_coef = self._geometric_schedule(prior_start, prior_end, progress)

    @staticmethod
    def _geometric_schedule(start: float, end: float, progress: float) -> float:
        if start <= 0.0:
            return 0.0
        if end <= 0.0:
            return start * max(0.0, 1.0 - progress)
        return start * (end / start) ** progress

    def _do_lr_schedule(self, kl: float):
        if not bool(self.cfg.get("adaptive_lr", True)):
            return
        if self.progress < 0.1:
            return
        desired_kl = float(self.cfg.get("desired_kl", 0.01))
        new_lr = self.current_lr
        if kl > desired_kl * 2.0:
            new_lr = max(1e-5, new_lr / 1.1)
        elif 0.0 < kl < desired_kl / 2.0:
            new_lr = min(5e-4, new_lr * 1.1)
        self.current_lr = new_lr
        for group in self.opt_actor.param_groups:
            group["lr"] = self.current_lr

    def train_op(self, td: TensorDict, vecnorm):
        info = self._ppo_update(td)
        self.num_updates += 1
        return info

    @torch.no_grad()
    def _compute_advantage(self, td: TensorDict):
        if "state_value" not in td.keys(True, True):
            self.critic(td.view(-1))
        if ("next", "state_value") not in td.keys(True, True):
            self.critic(td["next"].view(-1))

        rewards = td[REWARD_KEY].sum(dim=-1, keepdim=True)
        adv, ret = self.gae(
            rewards,
            td[TERM_KEY],
            td[DONE_KEY],
            td["state_value"],
            td["next", "state_value"],
        )
        td["adv"] = adv
        td["ret"] = ret

        valid = ~td["is_init"]
        mean = td["adv"][valid].mean()
        std = td["adv"][valid].std().clamp_min(1e-5)
        td["adv"][valid] = (td["adv"][valid] - mean) / std

    def _ppo_update(self, td: TensorDict):
        infos = []
        self._compute_advantage(td)
        for _ in range(int(self.cfg.get("ppo_epochs", 5))):
            for minibatch in make_batch(td, self.num_minibatches):
                infos.append(TensorDict(self._update(minibatch), []))

        info = {key: value.mean().item() for key, value in torch.stack(infos).items()}
        self._do_lr_schedule(info["actor/kl"])
        info["lr"] = self.current_lr
        info["gate/prior_coef"] = self.prior_coef
        return info

    def _update(self, minibatch: TensorDict):
        loc_old = minibatch["loc"].clone()
        scale_old = minibatch["scale"].clone()
        action_old = minibatch[ACTION_KEY].clone()
        logp_old = minibatch["sample_log_prob"].clone()
        valid = ~minibatch["is_init"]

        minibatch = minibatch.exclude("next", "sample_log_prob", ACTION_KEY)
        minibatch[REUSE_EXPERT_ACTIONS_KEY] = torch.ones(
            (*minibatch.batch_size, 1),
            dtype=torch.bool,
            device=minibatch.device,
        )
        self.actor(minibatch)
        values = self.critic(minibatch)["state_value"]

        distribution = IndependentNormal(minibatch["loc"], minibatch["scale"])
        logp = distribution.log_prob(action_old)
        entropy = distribution.entropy().mean()
        ratio = torch.exp(logp - logp_old).unsqueeze(-1)
        surrogate_1 = minibatch["adv"] * ratio
        surrogate_2 = minibatch["adv"] * ratio.clamp(
            1.0 - self.clip_param, 1.0 + self.clip_param
        )
        policy_loss = -torch.mean(torch.min(surrogate_1, surrogate_2) * valid)
        entropy_loss = -self.entropy_coef * entropy
        value_loss = F.mse_loss(values, minibatch["ret"], reduction="none")
        value_loss = (value_loss * valid).mean()

        learned_weights = minibatch[GATE_WEIGHTS_KEY].unflatten(-1, (3, 3))
        analytical_weights = minibatch[ANALYTICAL_GATE_WEIGHTS_KEY].unflatten(
            -1, (3, 3)
        )
        prior_kl_xyz = analytical_gate_kl(analytical_weights, learned_weights)
        valid_scalar = valid.squeeze(-1).to(prior_kl_xyz.dtype)
        prior_loss = (
            (prior_kl_xyz.mean(dim=-1) * valid_scalar).sum()
            / valid_scalar.sum().clamp_min(1.0)
        )

        loss = policy_loss + entropy_loss + value_loss + self.prior_coef * prior_loss
        self.opt_actor.zero_grad()
        self.opt_critic.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_actor.step()
        self.opt_critic.step()

        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            kl = torch.sum(
                torch.log(minibatch["scale"]) - torch.log(scale_old)
                + (scale_old.square() + (loc_old - minibatch["loc"]).square())
                / (2.0 * minibatch["scale"].square())
                - 0.5,
                dim=-1,
            ).mean()
            gate_delta = (learned_weights - analytical_weights).abs()

        return {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/grad_norm": actor_grad_norm.detach(),
            "actor/clamp_ratio": clip_fraction.detach(),
            "actor/kl": kl.detach(),
            "critic/value_loss": value_loss.detach(),
            "critic/grad_norm": critic_grad_norm.detach(),
            "gate/prior_loss": prior_loss.detach(),
            "gate/weight_delta_mean": gate_delta.mean().detach(),
            "gate/weight_delta_max": gate_delta.max().detach(),
            "gate/action_std": minibatch["scale"].mean().detach(),
        }

    def state_dict(self):
        actor = self._unwrap(self.actor)
        critic = self._unwrap(self.critic)
        return OrderedDict(
            actor=actor.state_dict(),
            critic=critic.state_dict(),
            last_phase="gate_train",
            _meta={
                "current_lr": self.current_lr,
                "entropy_coef": self.entropy_coef,
                "prior_coef": self.prior_coef,
                "progress": self.progress,
                "num_updates": self.num_updates,
            },
        )

    def load_state_dict(self, state_dict: Mapping, strict: bool = True):
        actor = self._unwrap(self.actor)
        critic = self._unwrap(self.critic)
        actor.load_state_dict(state_dict.get("actor", {}), strict=strict)
        critic.load_state_dict(state_dict.get("critic", {}), strict=strict)

        meta = state_dict.get("_meta", {})
        self.current_lr = float(meta.get("current_lr", self.current_lr))
        self.entropy_coef = float(meta.get("entropy_coef", self.entropy_coef))
        self.prior_coef = float(meta.get("prior_coef", self.prior_coef))
        self.progress = float(meta.get("progress", self.progress))
        self.num_updates = int(meta.get("num_updates", self.num_updates))
        for group in self.opt_actor.param_groups:
            group["lr"] = self.current_lr
