import torch

from gymkit.spec import Problem


def reference(logp_new, logp_old, logp_ref, rewards, mask, group_size, clip_eps, beta):
    n = rewards.shape[0]
    adv = torch.empty_like(rewards)
    for g in range(0, n, group_size):
        r = rewards[g:g + group_size]
        adv[g:g + group_size] = (r - r.mean()) / (r.std() + 1e-4)
    row_losses = []
    for i in range(n):
        m = mask[i]
        new, old, ref = logp_new[i][m], logp_old[i][m], logp_ref[i][m]
        ratio = torch.exp(new - old)
        surrogate = torch.minimum(ratio * adv[i], torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[i])
        kl = torch.exp(ref - new) - (ref - new) - 1
        row_losses.append((-(surrogate - beta * kl)).mean())
    return torch.stack(row_losses).mean(), adv


def _case(prompts, g, t, seed, name, clip_eps=0.2, beta=0.04, drift=0.1, tie_group=False, binary=True):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n = prompts * g
    old = -torch.rand(n, t, generator=generator) * 5.0
    new = old + drift * torch.randn(n, t, generator=generator)
    ref = old + 0.1 * torch.randn(n, t, generator=generator)
    rewards = (torch.rand(n, generator=generator) < 0.5).float() if binary else torch.randn(n, generator=generator)
    if tie_group:
        rewards[:g] = 1.0  # every completion of the first prompt scored the same
    lengths = torch.randint(1, t + 1, (n,), generator=generator)
    mask = torch.arange(t)[None, :] < lengths[:, None]
    junk = -30.0 * torch.rand(n, t, generator=generator)
    new, old, ref = (torch.where(mask, x, junk) for x in (new, old, ref))
    return {
        "_name": name,
        "logp_new": new.to("cuda"),
        "logp_old": old.to("cuda"),
        "logp_ref": ref.to("cuda"),
        "rewards": rewards.to("cuda"),
        "mask": mask.to("cuda"),
        "group_size": g,
        "clip_eps": clip_eps,
        "beta": beta,
    }


def functional_cases():
    cuda = {"device": "cuda"}
    zeros = torch.zeros(2, 2, **cuda)
    example = {
        "_name": "example",
        "logp_new": zeros, "logp_old": zeros.clone(), "logp_ref": zeros.clone(),
        "rewards": torch.tensor([1.0, 0.0], **cuda),
        "mask": torch.tensor([[True, True], [True, False]], **cuda),
        "group_size": 2, "clip_eps": 0.2, "beta": 0.0,
    }
    return [
        example,
        _case(2, 4, 6, 1, "identical rewards in a group: advantage 0", tie_group=True),
        _case(3, 4, 8, 2, "ratios far outside the clip range", drift=1.0),
        _case(3, 4, 8, 3, "beta = 0", beta=0.0),
        _case(2, 2, 5, 4, "G = 2, continuous rewards", binary=False),
        _case(8, 8, 64, 5, "random 8 prompts x 8, T=64", drift=0.3),
    ]


def performance_case(seed: int = 0):
    return _case(64, 8, 2048, seed, "perf")


def build() -> Problem:
    return Problem(
        slug="grpo_loss",
        title="GRPO loss",
        difficulty="hard",
        summary="Group-relative advantages, the clipped policy-ratio objective and a KL penalty, over masked completion tokens.",
        params=["logp_new", "logp_old", "logp_ref", "rewards", "mask", "group_size", "clip_eps", "beta"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        atol=1e-5,
        rtol=1e-4,
    )
