import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

try:
    import masked_attention_cuda
    CUDA_AVAILABLE = True
    print("CUDA extension loaded successfully")
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA extension not available, using PyTorch implementation")


class MaskedMultiHeadAttention(nn.Module):
    """
    Optimized Multi-Head Attention with masking support.
    Uses custom CUDA kernels when available for better performance.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        use_cuda_kernel: bool = True
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_cuda_kernel = use_cuda_kernel and CUDA_AVAILABLE
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of masked multi-head attention.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_dim]
            attention_mask: Boolean mask of shape [batch_size, seq_len] where True means keep
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Create default mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)
        
        if self.use_cuda_kernel and hidden_states.is_cuda:
            return self._forward_cuda(hidden_states, attention_mask, return_attention_weights)
        else:
            return self._forward_pytorch(hidden_states, attention_mask, return_attention_weights)
    
    def _forward_cuda(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention_weights: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass using custom CUDA kernels."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Step 1: Compute Q, K, V projections with masking
        qkv = masked_attention_cuda.masked_qkv_projection(
            hidden_states,
            self.q_proj.weight,
            self.k_proj.weight,
            self.v_proj.weight,
            self.q_proj.bias if self.q_proj.bias is not None else torch.zeros(hidden_dim, device=hidden_states.device),
            self.k_proj.bias if self.k_proj.bias is not None else torch.zeros(hidden_dim, device=hidden_states.device),
            self.v_proj.bias if self.v_proj.bias is not None else torch.zeros(hidden_dim, device=hidden_states.device),
            attention_mask
        )
        
        query, key, value = qkv[0], qkv[1], qkv[2]
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Step 2: Compute masked attention
        attention_output = masked_attention_cuda.masked_attention(query, key, value, attention_mask)
        
        # Step 3: Combine heads
        combined_output = masked_attention_cuda.combine_heads(attention_output, attention_mask)
        
        # Final projection
        output = self.out_proj(combined_output)
        
        # Apply mask to final output
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(output)
        output = output * mask_expanded.float()
        
        return output, None  # CUDA kernel doesn't return attention weights currently
    
    def _forward_pytorch(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention_weights: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fallback PyTorch implementation."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Apply mask to input
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
        masked_hidden_states = hidden_states * mask_expanded.float()
        
        # Compute Q, K, V
        query = self.q_proj(masked_hidden_states)
        key = self.k_proj(masked_hidden_states)
        value = self.v_proj(masked_hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            # Create 4D mask: [batch_size, 1, seq_len, seq_len]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.expand(batch_size, 1, seq_len, seq_len)
            
            # Create bidirectional mask (query_mask AND key_mask)
            query_mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq_len, 1]
            key_mask = attention_mask.unsqueeze(1).unsqueeze(1)     # [batch, 1, 1, seq_len]
            bidirectional_mask = query_mask * key_mask              # [batch, 1, seq_len, seq_len]
            
            # Apply mask to attention scores
            attention_scores = attention_scores.masked_fill(~bidirectional_mask, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, 0.0)  # Handle NaN from all -inf rows
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Compute attention output
        attention_output = torch.matmul(attention_weights, value)
        
        # Reshape and combine heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, hidden_dim)
        
        # Final projection
        output = self.out_proj(attention_output)
        
        # Apply mask to final output
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(output)
        output = output * mask_expanded.float()
        
        if return_attention_weights:
            return output, attention_weights
        else:
            return output, None


class OptimizedViTAttention(nn.Module):
    """
    Vision Transformer attention layer with masking optimization.
    Compatible with Hugging Face transformers.
    """
    
    def __init__(self, config):
        super().__init__()
        self.attention = MaskedMultiHeadAttention(
            hidden_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            use_cuda_kernel=True
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ):
        # Convert head_mask to attention_mask format if needed
        if attention_mask is None and head_mask is not None:
            # This is a simplified conversion - you might need to adapt based on your use case
            attention_mask = head_mask.squeeze() if head_mask.dim() > 2 else head_mask
        
        output, attention_weights = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            return_attention_weights=output_attentions
        )
        
        if output_attentions:
            return (output, attention_weights)
        else:
            return (output,)