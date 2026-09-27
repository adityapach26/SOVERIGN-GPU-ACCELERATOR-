import re

with open('src/cuda/kkt.cu', 'r') as f:
    content = f.read()

# Change d_M_orig_vals_ to d_M_vals_ in iterative refinement
content = content.replace('d_M_orig_vals_', 'd_M_vals_')

with open('src/cuda/kkt.cu', 'w') as f:
    f.write(content)
