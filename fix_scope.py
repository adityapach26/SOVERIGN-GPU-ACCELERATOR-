import re

with open('src/cuda/kkt.cu', 'r') as f:
    content = f.read()

# Define the block to remove
block_to_remove = r'''struct abs_max_functor \{
    __host__ __device__
    Float operator\(\)\(const Float& x\) const \{
        return x < 0\.0 \? -x : x;
    \}
\};

static Float compute_norm_kkt\(Index n, const Float\* d_vec\) \{
    thrust::device_ptr<const Float> ptr\(d_vec\);
    return thrust::transform_reduce\(thrust::device, ptr, ptr \+ n, abs_max_functor\(\), Float\(0\.0\), thrust::maximum<Float>\(\)\);
\}

'''

# Remove it
content = re.sub(block_to_remove, '', content)

# Define the block to insert
block_to_insert = '''
struct abs_max_functor {
    __host__ __device__
    Float operator()(const Float& x) const {
        return x < 0.0 ? -x : x;
    }
};

static Float compute_norm_kkt(Index n, const Float* d_vec) {
    thrust::device_ptr<const Float> ptr(d_vec);
    return thrust::transform_reduce(thrust::device, ptr, ptr + n, abs_max_functor(), Float(0.0), thrust::maximum<Float>());
}
'''

# Find insertion point
insertion_point = r"namespace sankhya \{\nnamespace gpu \{"
replace_str = "namespace sankhya {\nnamespace gpu {" + block_to_insert

content = re.sub(insertion_point, replace_str, content)

with open('src/cuda/kkt.cu', 'w') as f:
    f.write(content)
