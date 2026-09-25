import numpy as np

H = np.array([[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]])
c = np.array([-8.0, -12.0, -8.0])
A = np.array([[1.0, 1.0, 1.0]])
b = np.array([6.0])
lb = np.array([0.0, 0.0, 0.0])
ub = np.array([1e15, 1e15, 1e15])

x = np.array([2.0, 2.0, 2.0])
y = np.array([0.0])
zl = np.array([1.0, 1.0, 1.0])
zu = np.array([0.0, 0.0, 0.0])

for iter in range(200):
    rp = A @ x - b
    rd = H @ x + c - A.T @ y - zl + zu
    mu = (np.sum((x - lb) * zl) + np.sum(np.maximum(ub - x, 1e-12) * zu)) / (len(x) * 2)
    
    if np.linalg.norm(rp, np.inf) < 1e-6 and np.linalg.norm(rd, np.inf) < 1e-6 and mu < 1e-6:
        print(f"Converged in {iter} iterations. x = {x}, obj = {0.5 * x.T @ H @ x + c.T @ x}")
        break
        
    sigma = 0.1
    gap_l = x - lb
    gap_u = np.maximum(ub - x, 1e-12)
    comp_term = (sigma * mu - zl * gap_l) / gap_l - (sigma * mu - zu * gap_u) / gap_u
    rhs_x = -rd + comp_term
    
    D = zl / gap_l + zu / gap_u
    W = H + np.diag(D)
    Winv = np.linalg.inv(W)
    
    rhs_y = -rp - A @ Winv @ rhs_x
    
    S = A @ Winv @ A.T
    dy = np.linalg.solve(S, rhs_y)
    
    dx = Winv @ (A.T @ dy + rhs_x)
    dzl = (sigma * mu - zl * gap_l - zl * dx) / gap_l
    dzu = (sigma * mu - zu * gap_u + zu * dx) / gap_u
    
    alpha_p = 1.0
    alpha_d = 1.0
    for i in range(len(x)):
        if dx[i] < 0:
            alpha_p = min(alpha_p, -gap_l[i] / dx[i])
        if dx[i] > 0 and ub[i] < 1e15:
            alpha_p = min(alpha_p, gap_u[i] / dx[i])
        if dzl[i] < 0:
            alpha_d = min(alpha_d, -zl[i] / dzl[i])
        if dzu[i] < 0:
            alpha_d = min(alpha_d, -zu[i] / dzu[i])
            
    alpha_p = min(1.0, 0.995 * alpha_p)
    alpha_d = min(1.0, 0.995 * alpha_d)
    
    x += alpha_p * dx
    zl += alpha_d * dzl
    zu += alpha_d * dzu
    y += alpha_d * dy
else:
    print("Iteration limit reached")
