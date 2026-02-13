"""
GptOss MoE specific utilities and processing functions.

This module contains specialized logic for handling GptOss architecture,
which uses a batched expert implementation different from standard MoE models.

GptOss Architecture:
-------------------
- Uses batched expert weights stored in single tensors: (num_experts, ...)
- Router returns (router_scores, router_indices) tuple
- MLP forward returns (routed_out, router_scores) tuple
- Experts are computed via batch matrix multiplication for efficiency

This differs from loop-based MoE (e.g., Qwen3, Mixtral) where experts are
stored as separate nn.Module instances in a ModuleList, and from Llama4's
fused MoE implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def compute_gptoss_activations(
    module: nn.Module,
    flat_input: torch.Tensor,
    output: Tuple[torch.Tensor, torch.Tensor],
    num_experts: int,
    top_k: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute expert activations for GptOss MLP layer.
    
    GptOss uses a batched expert implementation where all expert weights
    are stored in single tensors with shape (num_experts, ...).
    
    Args:
        module: GptOssMLP module instance
        flat_input: Flattened input tensor (total_tokens, hidden_dim)
        output: Tuple of (routed_out, router_scores) from GptOssMLP forward
        num_experts: Number of experts in the layer
        top_k: Number of experts activated per token
        device: Device to run computations on
        
    Returns:
        Tuple of:
            - activations: (num_experts, total_tokens, hidden_dim) - unweighted expert outputs
            - router_logits: (total_tokens, num_experts) - router logit scores
            - selected_experts: (total_tokens, top_k) - indices of selected experts
    """
    # GptOss specific: output = (routed_out, router_scores)
    # router_scores is sparse: (total_tokens, num_experts) with only top-k non-zero
    _, router_scores = output
    router_scores_flat = router_scores.view(-1, num_experts)
    
    # Get router logits and selected experts by calling router directly
    router_output = module.router(flat_input)
    if isinstance(router_output, tuple):
        _, selected_experts = router_output  # (total_tokens, top_k)
    else:
        # Fallback: extract from router_scores
        selected_experts = (router_scores_flat > 0).nonzero(as_tuple=False)[:, 1].view(-1, top_k)
    
    # Compute router logits for later use (needed for pruning metrics)
    router_logits = F.linear(flat_input, module.router.weight, module.router.bias)
    
    # Compute activations for all experts in batch
    # This is more efficient than looping over experts individually
    
    # Expand flat_input to (num_experts, num_tokens, hidden_dim)
    expert_inputs = flat_input.unsqueeze(0).expand(num_experts, -1, -1)
    
    # gate_up_proj: (num_experts, hidden_size, 2 * expert_dim)
    # Batch matmul: (num_experts, num_tokens, hidden_size) @ (num_experts, hidden_size, 2 * expert_dim)
    gate_up = torch.bmm(
        expert_inputs, 
        module.experts.gate_up_proj
    ) + module.experts.gate_up_proj_bias.unsqueeze(1)
    
    # gate_up: (num_experts, num_tokens, 2 * expert_dim)
    gate, up = gate_up.chunk(2, dim=-1)
    intermediate = F.silu(gate) * up
    
    # down_proj: (num_experts, expert_dim, hidden_size)
    expert_out = torch.bmm(
        intermediate, 
        module.experts.down_proj
    ) + module.experts.down_proj_bias.unsqueeze(1)
    
    # expert_out: (num_experts, num_tokens, hidden_size)
    activations = expert_out
    
    return activations, router_logits, selected_experts


def is_gptoss_module(module: nn.Module) -> bool:
    """Check if a module is a GptOssMLP module."""
    return module.__class__.__name__ == "GptOssMLP"


def prune_gptoss_experts(
    moe: nn.Module,
    model_attrs: dict,
    retained_expert_indices: list[int],
) -> None:
    """
    Physical pruning for Batched Experts (GptOss).
    """
    experts = getattr(moe, model_attrs["experts"])
    router = getattr(moe, model_attrs["router"])
    
    print(f"   ✂️ Pruning GptOss Layer: keeping {len(retained_expert_indices)}/{experts.num_experts} experts")

    # === Helper: force deep copy and break memory contiguity ===
    # Prevents safetensors stride metadata issues on save
    def shrink_tensor(tensor, indices, dim=0):
        if tensor is None: return None
        # .clone().contiguous() is key to avoid corruption
        return tensor.index_select(dim, indices.to(tensor.device)).clone().contiguous()

    indices_tensor = torch.tensor(retained_expert_indices, device=experts.gate_up_proj.device)

    # 1. Prune expert weights
    experts.gate_up_proj.data = shrink_tensor(experts.gate_up_proj.data, indices_tensor)
    experts.down_proj.data = shrink_tensor(experts.down_proj.data, indices_tensor)
    
    # 2. Prune expert bias (previous fix logic)
    if hasattr(experts, 'gate_up_proj_bias'):
        experts.gate_up_proj_bias.data = shrink_tensor(experts.gate_up_proj_bias.data, indices_tensor)
    if hasattr(experts, 'down_proj_bias'):
        experts.down_proj_bias.data = shrink_tensor(experts.down_proj_bias.data, indices_tensor)

    # 3. Update expert count attribute
    experts.num_experts = len(retained_expert_indices)

    # 4. Prune router (row slicing)
    # Note: we physically cut router rows. This shrinks the softmax denominator and requires fine-tuning.
    router.weight.data = shrink_tensor(router.weight.data, indices_tensor, dim=0)
    if hasattr(router, "bias") and router.bias is not None:
        router.bias.data = shrink_tensor(router.bias.data, indices_tensor, dim=0)
    
    router.out_features = len(retained_expert_indices)
    router.num_experts = len(retained_expert_indices)