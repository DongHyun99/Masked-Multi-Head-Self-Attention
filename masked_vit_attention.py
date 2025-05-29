"""
Masked Vision Transformer Multi-Head Self-Attention Module
Integrates with HuggingFace transformers library with CUDA acceleration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

# Import CUDA extension
try:
    import masked_attention_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA extension not available. Falling back to PyTorch implementation.")


class MaskedViTAttention(nn.Module):
    """
    Masked Multi-Head Self-Attention for Vision Transformer with CUDA acceleration
    
    Args:
        config: Configuration object with the following attributes:
            - hidden_size: Hidden dimension size
            - num_attention_heads: Number of attention heads
            - attention_probs_dropout_prob: Dropout probability for attention
            - hidden_dropout_prob: Dropout probability for hidden layers
    """
    
    def __init__(self, config, use_cuda_kernel=True):
        super().__init__()
        
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.use_cuda_kernel = use_cuda_kernel and CUDA_AVAILABLE
        
        # Linear transformations for Q, K, V
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Output projection
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Scale factor
        self.scale = math.sqrt(self.attention_head_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for masked multi-head self-attention
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Mask tensor of shape [batch_size, seq_length] with True for valid tokens
            head_mask: Optional head mask (not used in CUDA kernel)
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (attention_output, attention_probs)
        """
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Convert attention mask to boolean if needed
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()
        else:
            attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bool, device=hidden_states.device)
        
        # Use CUDA kernel if available and enabled
        if self.use_cuda_kernel and hidden_states.is_cuda:
            return self._cuda_forward(hidden_states, attention_mask, output_attentions)
        else:
            return self._pytorch_forward(hidden_states, attention_mask, head_mask, output_attentions)
    
    def _cuda_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass using CUDA kernel"""
        
        # Get weight and bias tensors
        query_weight = self.query.weight.t().contiguous()  # [hidden_size, hidden_size]
        key_weight = self.key.weight.t().contiguous()
        value_weight = self.value.weight.t().contiguous()
        dense_weight = self.dense.weight.t().contiguous()
        
        query_bias = self.query.bias
        key_bias = self.key.bias
        value_bias = self.value.bias
        dense_bias = self.dense.bias
        
        # Call CUDA kernel
        attention_output = masked_attention_cuda.masked_multi_head_attention(
            hidden_states,
            query_weight,
            key_weight,
            value_weight,
            dense_weight,
            query_bias,
            key_bias,
            value_bias,
            dense_bias,
            attention_mask,
            self.num_attention_heads
        )
        
        # Apply output dropout
        attention_output = self.output_dropout(attention_output)
        
        # Return attention weights as None for CUDA kernel (not computed for efficiency)
        attention_probs = None if not output_attentions else torch.zeros(
            hidden_states.shape[0], self.num_attention_heads, 
            hidden_states.shape[1], hidden_states.shape[1],
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        
        return attention_output, attention_probs
    
    def _pytorch_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fallback PyTorch implementation"""
        
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Apply mask to input (zero out masked tokens)
        masked_hidden_states = hidden_states.clone()
        masked_hidden_states[~attention_mask] = 0
        
        # Compute Q, K, V
        mixed_query_layer = self.query(masked_hidden_states)
        mixed_key_layer = self.key(masked_hidden_states)
        mixed_value_layer = self.value(masked_hidden_states)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            # Create extended mask for all heads
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            extended_attention_mask = extended_attention_mask.expand(
                batch_size, self.num_attention_heads, seq_length, seq_length
            )
            
            # Mask out invalid positions
            attention_scores = attention_scores.masked_fill(~extended_attention_mask, -1e9)
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply head mask if provided
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        # Compute context
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        # Final projection
        attention_output = self.dense(context_layer)
        attention_output = self.output_dropout(attention_output)
        
        # Add residual connection and apply mask
        attention_output = attention_output + hidden_states
        attention_output[~attention_mask] = hidden_states[~attention_mask]  # Preserve original for masked tokens
        
        return attention_output, attention_probs if output_attentions else None
    
    def transpose_for_scores(self, x):
        """Transpose tensor for multi-head attention computation"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)


class MaskedViTSelfAttention(nn.Module):
    """
    Wrapper for HuggingFace compatibility
    """
    
    def __init__(self, config, use_cuda_kernel=True):
        super().__init__()
        self.attention = MaskedViTAttention(config, use_cuda_kernel)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        return self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )


# Utility functions for integration
def replace_vit_attention_with_masked(model, use_cuda_kernel=True):
    """
    Replace standard ViT attention modules with masked versions
    
    Args:
        model: HuggingFace ViT model
        use_cuda_kernel: Whether to use CUDA kernel acceleration
    """
    
    def replace_attention_recursive(module, name=""):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Check if this is an attention module we want to replace
            if hasattr(child_module, '__class__') and 'Attention' in child_module.__class__.__name__:
                if hasattr(child_module, 'config') or hasattr(child_module, 'attention'):
                    # Get config from the module or its parent
                    config = getattr(child_module, 'config', None)
                    if config is None and hasattr(module, 'config'):
                        config = module.config
                    
                    if config is not None:
                        # Replace with our masked attention
                        new_attention = MaskedViTSelfAttention(config, use_cuda_kernel)
                        setattr(module, child_name, new_attention)
                        print(f"Replaced {full_name} with MaskedViTSelfAttention")
            else:
                # Recursively process child modules
                replace_attention_recursive(child_module, full_name)
    
    replace_attention_recursive(model)
    return model


def create_random_mask(batch_size, seq_length, mask_ratio=0.15, device='cuda'):
    """
    Create random boolean mask for testing
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        mask_ratio: Ratio of tokens to mask (False)
        device: Device to create mask on
        
    Returns:
        Boolean mask tensor with True for valid tokens
    """
    mask = torch.rand(batch_size, seq_length, device=device) > mask_ratio
    # Ensure at least one token per sequence is valid
    for i in range(batch_size):
        if not mask[i].any():
            mask[i, 0] = True
    return mask