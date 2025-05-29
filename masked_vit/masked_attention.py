"""
MaskedAttention - Drop-in replacement for HuggingFace ViT attention with CUDA optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings

try:
    import masked_attention_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn("CUDA extension not available, using PyTorch fallback")


class MaskedAttention(nn.Module):
    """
    Optimized multi-head attention with masking support for Vision Transformers.
    
    This class is designed as a drop-in replacement for HuggingFace's ViTAttention
    but with optimized CUDA kernels that efficiently handle masked tokens.
    
    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
        bias (bool): Whether to use bias in linear layers
        
    Example:
        >>> # Replace HuggingFace ViTAttention
        >>> from masked_vit import MaskedAttention
        >>> attention = MaskedAttention(d_model=768, num_heads=12)
        >>> 
        >>> # Use with mask
        >>> input_tensor = torch.randn(2, 197, 768).cuda()  # [batch, seq_len, d_model]
        >>> mask = torch.ones(2, 197, dtype=torch.bool).cuda()  # [batch, seq_len]
        >>> mask[0, 100:] = False  # Mask some tokens
        >>> 
        >>> output = attention(input_tensor, mask=mask)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 12,
        dropout: float = 0.0,
        bias: bool = True,
        use_cuda_kernel: bool = True
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.use_cuda_kernel = use_cuda_kernel and CUDA_AVAILABLE
        
        # QKV projection (combined for efficiency)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        
        # Output projection
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # Scale factor
        self.scale = self.head_dim ** -0.5
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of masked attention.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, d_model]
            mask: Boolean mask [batch_size, seq_len] where True means keep token
            head_mask: Head mask (not used in current implementation)
            output_attentions: Whether to output attention weights
            
        Returns:
            Tuple of (output_tensor, attention_weights)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Validate inputs
        assert d_model == self.d_model, f"Input d_model {d_model} != expected {self.d_model}"
        
        if mask is not None:
            assert mask.shape == (batch_size, seq_len), f"Mask shape {mask.shape} != expected ({batch_size}, {seq_len})"
            assert mask.dtype == torch.bool, f"Mask must be boolean, got {mask.dtype}"
        else:
            # Create default mask (all tokens valid)
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)
        
        # Use CUDA kernel if available and conditions are met
        if (self.use_cuda_kernel and 
            hidden_states.is_cuda and 
            not output_attentions and 
            head_mask is None):
            return self._forward_cuda(hidden_states, mask)
        else:
            return self._forward_pytorch(hidden_states, mask, output_attentions)
    
    def _forward_cuda(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass using optimized CUDA kernel."""
        try:
            output = masked_attention_cuda.masked_multihead_attention_forward(
                hidden_states,
                mask,
                self.qkv.weight,
                self.qkv.bias if self.qkv.bias is not None else torch.zeros(3 * self.d_model, device=hidden_states.device, dtype=hidden_states.dtype),
                self.proj.weight,
                self.proj.bias if self.proj.bias is not None else torch.zeros(self.d_model, device=hidden_states.device, dtype=hidden_states.dtype),
                self.num_heads,
                self.dropout if self.training else 0.0,
                self.training
            )
            return output, None
        except Exception as e:
            warnings.warn(f"CUDA kernel failed: {e}. Falling back to PyTorch implementation.")
            return self._forward_pytorch(hidden_states, mask, False)
    
    def _forward_pytorch(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fallback PyTorch implementation."""
        batch_size, seq_len, d_model = hidden_states.shape
        
        # QKV projection
        qkv = self.qkv(hidden_states)  # [batch_size, seq_len, 3 * d_model]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv.unbind(0)
        
        # Apply masking to Q, K, V
        mask_expanded = mask.unsqueeze(1).unsqueeze(3)  # [batch_size, 1, seq_len, 1]
        q = q.masked_fill(~mask_expanded, 0)
        k = k.masked_fill(~mask_expanded, 0)
        v = v.masked_fill(~mask_expanded, 0)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Create attention mask [batch_size, seq_len, seq_len]
        attention_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # [batch_size, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
        
        # Apply attention mask
        scores = scores.masked_fill(~attention_mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape and project
        context = context.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        output = self.proj(context)
        
        # Apply output masking
        mask_expanded = mask.unsqueeze(2)  # [batch_size, seq_len, 1]
        output = output.masked_fill(~mask_expanded, 0)
        
        return output, attn_weights if output_attentions else None
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, num_heads={self.num_heads}, dropout={self.dropout}, cuda_kernel={self.use_cuda_kernel}'


class MaskedViTAttention(MaskedAttention):
    """
    Alias for MaskedAttention with ViT-specific defaults.
    Direct replacement for transformers.models.vit.modeling_vit.ViTAttention
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize from HuggingFace config or kwargs."""
        if config is not None:
            # Extract parameters from HuggingFace config
            d_model = getattr(config, 'hidden_size', 768)
            num_heads = getattr(config, 'num_attention_heads', 12)
            dropout = getattr(config, 'attention_probs_dropout_prob', 0.0)
        else:
            # Use provided kwargs
            d_model = kwargs.get('d_model', 768)
            num_heads = kwargs.get('num_heads', 12)
            dropout = kwargs.get('dropout', 0.0)
        
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            use_cuda_kernel=kwargs.get('use_cuda_kernel', True)
        )


def replace_vit_attention(model, use_cuda_kernel: bool = True):
    """
    Replace all ViT attention modules in a model with MaskedAttention.
    
    Args:
        model: HuggingFace ViT model
        use_cuda_kernel: Whether to use CUDA optimization
        
    Returns:
        Modified model with replaced attention layers
    """
    def replace_attention_recursive(module):
        for name, child in module.named_children():
            if 'attention' in name.lower() and hasattr(child, 'query'):
                # This is likely a ViT attention module
                try:
                    # Extract config if available
                    config = getattr(model, 'config', None)
                    new_attention = MaskedViTAttention(config, use_cuda_kernel=use_cuda_kernel)
                    setattr(module, name, new_attention)
                    print(f"Replaced {name} with MaskedAttention")
                except Exception as e:
                    print(f"Failed to replace {name}: {e}")
            else:
                replace_attention_recursive(child)
    
    replace_attention_recursive(model)
    return model