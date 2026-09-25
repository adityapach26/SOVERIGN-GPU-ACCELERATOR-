# ADR-QP-001: Phase 16A.1 QP Mathematical Contract and Scope Lock

## Context
This ADR defines the scope, representation, and integration contract for Quadratic Programming (QP) capabilities in SANKHYA / VAJRA-OPT. 

### Source Traceability
- **[A] SOURCE-SPECIFIED**: The solver scope includes LP, MILP, and QP from mathematical first principles.
- **[C] ENGINEERING INFERENCE**: 
  - Convex QP will be implemented using the existing primal-dual interior-point/KKT spine.
  - The solver will reuse project-owned KKT assembly/factorization interfaces.
  - Nonconvex QP is unsupported and will be explicitly rejected rather than heuristically modified.
  - A persistent GPU residency path orchestrating existing GPU KKT code will be used.
- **[D] SOURCE GAP**: 
  - Complete QP algorithm not uniquely specified.
  - Hessian storage policy not uniquely specified.
  - Nonconvex-QP handling not source-specified.
  - Exact numerical termination defaults not source-specified.
  - Exact QPLIB coverage not source-specified.

## Decisions

### 1. Canonical QP Mathematical Form
**[C] ENGINEERING INFERENCE**: The canonical convex QP is represented as:
```
minimize    1/2 x^T H x + c^T x
subject to  A x = b
            l <= x <= u
```
- `H`: Hessian matrix (symmetric positive semidefinite).
- `c`: Linear objective vector (reuses `obj` in `core::Model`).
- `A`: Equality constraint matrix.
- `b`: Right-hand side vector (`rhs` in `core::Model`).
- `l`, `u`: Variable bounds.

### 2. Gradient and Hessian Convention
**[C] ENGINEERING INFERENCE**:
- `∇f(x) = Hx + c`
- `∇²f(x) = H`
- `H` must be structurally symmetric. The implementation expects a `CSCMatrix` containing only the lower triangular elements (including diagonal).

### 3. Convexity Boundary
**[C] ENGINEERING INFERENCE**: 
- Supported: `H` is symmetric positive semidefinite (PSD).
- Unsupported: `H` is indefinite or negative definite (nonconvex).

### 4. Indefinite / Nonconvex Behavior
**[C] ENGINEERING INFERENCE**: 
Unsupported indefinite/nonconvex Hessians will fail explicitly by throwing an exception (or returning an equivalent unsupported status code). We will NOT silently regularize, fall back to an external solver, or downgrade to an LP.

### 5. Numerical Termination Contract
**[C] ENGINEERING INFERENCE**: 
The solver termination tolerances reuse `math::kDefaultFeasibilityTol` and established norms:
- Primal Feasibility: `||Ax - b||_{\infty} <= epsilon_{feas}`
- Dual Feasibility: Complementary slackness violation `<= epsilon_{dual}`
- Stationarity: `||Hx + c - A^T y - z||_{\infty} <= epsilon_{grad}`
- Default iteration limit: 200

### 6. CPU Reference Path
**[C] ENGINEERING INFERENCE**: 
The CPU reference path uses the **normal-equations reduction** of the QP Newton system:
1. Form the SPD matrix `W = H + D` where `D` is the positive barrier diagonal.
2. Solve `W z = v` for right-hand sides via Cholesky factorization (SPD).
3. Form the Schur complement `S = A W^{-1} A^T` (SPD).
4. Solve `S Δy = rhs` via Cholesky factorization (SPD, reusing existing `GPUKKTCholeskySolver` architecture).
5. Back-substitute for `Δx`.

Both `W` and `S` are SPD (because `H` is PSD and `D > 0` for interior points), so only Cholesky factorization is required — no LDLT or indefinite factorizer is needed.

No external solvers (Gurobi, OSQP, HiGHS, etc.) will be wrapped or invoked.

### 7. GPU Acceleration Path
**[C] ENGINEERING INFERENCE**: 
GPU acceleration will map the extended KKT system into the existing `WorkingBasisState` and `HBFManager` structures, performing sparse LU/Cholesky updates using project-owned GPU kernels. 

### 8. Interaction with Existing IPM/KKT Infrastructure
**[C] ENGINEERING INFERENCE**: 
The QP solver will interface with `src/core/problem.hpp` (for A, b, c), `src/numerics/factorization.hpp`, and `src/gpu/hbf.cuh`. These components will NOT be modified during 16A.1.

### 9. API Representation
**[C] ENGINEERING INFERENCE**:
A minimal `QPModel` aggregates the linear `core::Model` and the Hessian `core::CSCMatrix`. `QPSolver` exposes a `solve(const QPModel&)` interface returning standard `Solution` objects.

### 10. QPLIB Validation Scope
**[C] ENGINEERING INFERENCE**:
QPLIB validation is deferred to future milestones (Step 16A.2 or later). Step 16A.1 only commits to the structural/contractual capability to express these problems.

### 11. Source Gap Register
- Complete QP algorithm not uniquely specified.
- Hessian storage policy not uniquely specified.
- Nonconvex-QP handling not source-specified.
- Exact numerical defaults not source-specified.
- Exact QPLIB coverage not source-specified.

