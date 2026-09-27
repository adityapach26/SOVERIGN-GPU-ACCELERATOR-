import re

with open('src/cuda/kkt.cu', 'r') as f:
    content = f.read()

# I will add thrust headers and the compute_norm_kkt function
headers = '''#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <cmath>

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

# Find the first namespace sankhya declaration and insert headers before it
match_str = r"namespace sankhya \{"
replace_str = headers + "\nnamespace sankhya {"
new_content = re.sub(match_str, replace_str, content, count=1)

with open('src/cuda/kkt.cu', 'w') as f:
    f.write(new_content)
