import re

with open('src/cuda/kkt.cu', 'r') as f:
    content = f.read()

match_str = r"        // Apply diagonal regularization: M_delta = A Theta A\^T \+ delta\*I \(ADR Phase 33.1\)\n        M_values\[p\] = sum \+ \(j == i \? kIPMNormalEquationRegularization : Float\(0.0\)\);"
replace_str = r"        M_values[p] = sum;"

new_content = re.sub(match_str, replace_str, content)
with open('src/cuda/kkt.cu', 'w') as f:
    f.write(new_content)
