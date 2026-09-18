"""Branch registry (plan §6). build_branch("a".."e") -> nn.Module whose
forward(mono_chunk_at_branch_sr) -> per-chunk score/logit (B,)."""
from __future__ import annotations


def build_branch(name: str, pretrained: bool = True):
    name = name.lower()
    if name == "a":
        from models.branch_a_ssl import BranchA
        return BranchA(pretrained=pretrained)
    if name == "b":
        from models.branch_b_mert import BranchB
        return BranchB(pretrained=pretrained)
    if name == "c":
        from models.branch_c_physics import BranchC
        return BranchC()
    if name == "d":
        from models.branch_d_longcontext import BranchD
        return BranchD(pretrained=pretrained)
    if name == "e":
        from models.branch_e_anomaly import BranchE
        return BranchE()
    raise ValueError(f"Unknown branch: {name!r} (expected a-e)")
