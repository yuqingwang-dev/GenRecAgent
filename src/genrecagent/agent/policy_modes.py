"""Policy names only; importable without Torch/NumPy or model initialization."""

GUARDED_POLICIES = ("guarded_prompt", "guarded_sft")
COMPACT_POLICIES = ("compact_prompt", "compact_sft")
SFT_POLICIES = ("sft", "guarded_sft", "compact_sft")
POLICY_CHOICES = ("fixed", "prompt", "sft", *GUARDED_POLICIES, *COMPACT_POLICIES)
