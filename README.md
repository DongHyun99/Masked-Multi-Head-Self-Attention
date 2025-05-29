# Masked-Multi-Head-Self-Attention-CUDA

This is a CUDA kernel-based masked multi-head self-attention implementation code.

## Useage

``` bash
git clone https://github.com/DongHyun99/masked-attention-cuda.git
cd masked-attention-cuda

chmod +x install.sh
./install.sh
```

## Test performance

### whole testing
```bash
python test_masked_attention.py
```

### separate testing

```bash
# default
python -c "from test_masked_attention import test_basic_functionality; test_basic_functionality()"

# comparison
python -c "from test_masked_attention import test_performance_comparison; test_performance_comparison()"

# extensive comparison
python -c "from test_masked_attention import run_comprehensive_performance_test; run_comprehensive_performance_test()"
```