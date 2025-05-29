"""
Performance-optimized version with adaptive backend selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

try:
    import masked_attention_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

class MaskedAttention(nn.Module):
    """
    Performance-optimized masked attention with adaptive backend selection
    """
    
    def __init__(self, config, use_cuda_kernel=True):
        super().__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.use_cuda_kernel = use_cuda_kernel and CUDA_AVAILABLE
        
        # Use single linear layer for QKV projection (more efficient)
        self.qkv = nn.Linear(config.hidden_size, 3 * self.all_head_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.scale = math.sqrt(self.attention_head_size)
        
        # Performance thresholds for backend selection
        self.cublas_threshold_seq = 128
        self.cublas_threshold_dim = 512
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bool, device=hidden_states.device)
        elif attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
        
        # Adaptive backend selection based on tensor size
        use_optimized_path = (
            seq_length >= self.cublas_threshold_seq and 
            hidden_size >= self.cublas_threshold_dim and
            hidden_states.is_cuda
        )
        
        if use_optimized_path:
            return self._optimized_forward(hidden_states, attention_mask, output_attentions)
        elif self.use_cuda_kernel and hidden_states.is_cuda:
            return self._cuda_forward(hidden_states, attention_mask, output_attentions)
        else:
            return self._pytorch_forward(hidden_states, attention_mask, head_mask, output_attentions)
    
    def _optimized_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Highly optimized path using cuBLAS and minimal custom kernels"""
        
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Mask input efficiently
        if attention_mask is not None:
            masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1).float()
        else:
            masked_hidden_states = hidden_states
        
        # Single QKV projection (much more efficient than 3 separate projections)
        qkv = self.qkv(masked_hidden_states)
        qkv = qkv.view(batch_size, seq_length, 3, self.num_attention_heads, self.attention_head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        
        query, key, value = qkv[0], qkv[1], qkv[2]
        
        # Use Flash Attention if available (PyTorch 2.0+)
        if hasattr(F, 'scaled_dot_product_attention'):
            # Convert mask to the format expected by Flash Attention
            if attention_mask is not None:
                # Create causal mask format: [B, H, N, N]
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
                attn_mask = attn_mask.expand(batch_size, self.num_attention_heads, seq_length, seq_length)
                attn_mask = ~attn_mask  # Flash attention uses True for masked positions
            else:
                attn_mask = None
            
            # Use Flash Attention (highly optimized)
            try:
                context_layer = F.scaled_dot_product_attention(
                    query, key, value,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False
                )
                attention_probs = None  # Flash attention doesn't return attention weights
            except:
                # Fallback to standard attention if Flash attention fails
                return self._standard_attention_forward(query, key, value, attention_mask, output_attentions)
        else:
            return self._standard_attention_forward(query, key, value, attention_mask, output_attentions)
        
        # Reshape output
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        # Final projection
        attention_output = self.dense(context_layer)
        attention_output = self.output_dropout(attention_output)
        
        # Residual connection with proper masking
        attention_output = attention_output + hidden_states
        if attention_mask is not None:
            # Preserve original values for masked positions
            attention_output = torch.where(
                attention_mask.unsqueeze(-1), 
                attention_output, 
                hidden_states
            )
        
        return attention_output, attention_probs
    
    def _standard_attention_forward(self, query, key, value, attention_mask, output_attentions):
        """Standard attention computation with optimizations"""
        
        batch_size, seq_length = query.shape[0], query.shape[2]
        
        # Compute attention scores using optimized matmul
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        
        # Apply mask efficiently
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.expand(
                batch_size, self.num_attention_heads, seq_length, seq_length
            )
            attention_scores = attention_scores.masked_fill(~extended_attention_mask, -1e9)
        
        # Softmax with improved numerical stability
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value)
        
        return context_layer, attention_probs if output_attentions else None
    
    def _cuda_forward(self, hidden_states, attention_mask, output_attentions):
        """CUDA kernel path (for smaller tensors where custom kernels might be beneficial)"""
        
        # Get weights in the format expected by CUDA kernel
        qkv_weight = self.qkv.weight.view(3, self.all_head_size, -1)
        query_weight = qkv_weight[0].t().contiguous()
        key_weight = qkv_weight[1].t().contiguous()
        value_weight = qkv_weight[2].t().contiguous()
        dense_weight = self.dense.weight.t().contiguous()
        
        qkv_bias = self.qkv.bias.view(3, self.all_head_size)
        query_bias = qkv_bias[0]
        key_bias = qkv_bias[1]
        value_bias = qkv_bias[2]
        dense_bias = self.dense.bias
        
        # Call optimized CUDA kernel
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
        
        attention_output = self.output_dropout(attention_output)
        return attention_output, None
    
    def _pytorch_forward(self, hidden_states, attention_mask, head_mask, output_attentions):
        """Fallback PyTorch implementation"""
        
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Apply mask to input
        if attention_mask is not None:
            masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1).float()
        else:
            masked_hidden_states = hidden_states
        
        # QKV projection
        qkv = self.qkv(masked_hidden_states)
        qkv = qkv.view(batch_size, seq_length, 3, self.num_attention_heads, self.attention_head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        query, key, value = qkv[0], qkv[1], qkv[2]
        
        # Standard attention computation
        context_layer, attention_probs = self._standard_attention_forward(
            query, key, value, attention_mask, output_attentions
        )
        
        # Reshape and project
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        attention_output = self.dense(context_layer)
        attention_output = self.output_dropout(attention_output)
        
        # Residual connection with masking
        attention_output = attention_output + hidden_states
        if attention_mask is not None:
            attention_output = torch.where(
                attention_mask.unsqueeze(-1),
                attention_output,
                hidden_states
            )
        
        return attention_output, attention_probs


def benchmark_attention_backends():
    """Benchmark different attention backends"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA not available for benchmarking")
        return
    
    class BenchConfig:
        hidden_size = 768
        num_attention_heads = 12
        attention_probs_dropout_prob = 0.0
        hidden_dropout_prob = 0.0
    
    config = BenchConfig()
    
    # Test different sizes
    test_sizes = [
        (2, 64, 768),    # Small
        (4, 128, 768),   # Medium  
        (8, 197, 768),   # ViT-Base
        (16, 384, 768),  # Large
    ]
    
    for batch_size, seq_len, hidden_size in test_sizes:
        print(f"\nTesting size: B={batch_size}, N={seq_len}, D={hidden_size}")
        
        # Create test data
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
        mask = torch.rand(batch_size, seq_len, device=device) > 0.15
        
        # Test different backends
        backends = [
            ("Optimized (Flash/cuBLAS)", MaskedAttention(config, False)),
            ("CUDA Kernel", MaskedAttention(config, True)),
        ]
        
        for name, model in backends:
            model = model.to(device).eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(hidden_states, mask)
            
            torch.cuda.synchronize()
            
            # Benchmark
            import time
            num_iterations = 100
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(hidden_states, mask)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            print(f"  {name}: {elapsed:.4f}s ({elapsed/num_iterations*1000:.2f}ms/iter)")


if __name__ == "__main__":
    benchmark_attention_backends()