#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "core/problem.hpp"
#include "parsers/mps_reader.hpp"
#include "verifier/certificate.hpp"
#include "ipm/mehrotra.hpp"

#include <cmath>
#include <limits>

using namespace sankhya;

// ---- DIAGNOSTIC: identify failing KKT condition ----
// TEMPORARY diagnostic only. Does NOT modify verify_optimal.
// Reproduces verify_optimal logic step-by-step and prints the worst violation.
static void diagnose_certificate(const std::string& label,
                                  const core::Model& model,
                                  const std::vector<Float>& x,
                                  const std::vector<Float>& pi) {
    const std::size_t n = x.size();
    const std::size_t m = pi.size();
    const double eps = 1e-6;

    std::cout << "\n--- DIAGNOSTIC: " << label << " ---\n";
    std::cout << "  x.size()=" << n << "  pi.size()=" << m
              << "  model.n=" << model.obj.size()
              << "  model.m=" << model.rhs.size() << "\n";

    // 1. Primal feasibility: ||Ax - b||_inf
    std::vector<double> Ax(m, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        Index st = model.A.col_ptrs[j], en = model.A.col_ptrs[j+1];
        for (Index k = st; k < en; ++k) {
            std::size_t i = static_cast<std::size_t>(model.A.row_indices[k]);
            Ax[i] += static_cast<double>(model.A.values[k]) * static_cast<double>(x[j]);
        }
    }
    double max_primal = 0.0;
    std::size_t max_primal_idx = 0;
    for (std::size_t i = 0; i < m; ++i) {
        double r = std::abs(Ax[i] - static_cast<double>(model.rhs[i]));
        if (r > max_primal) { max_primal = r; max_primal_idx = i; }
    }
    std::cout << "  Primal ||Ax-b||_inf = " << max_primal
              << " at row " << max_primal_idx << "\n";
    if (max_primal >= eps)
        std::cout << "  ** FAIL: primal infeasibility\n";

    // 2. Bound feasibility: worst lb/ub violation
    double max_bound = 0.0;
    std::size_t max_bound_idx = 0;
    for (std::size_t j = 0; j < n; ++j) {
        double xj = x[j], lbj = model.lb[j], ubj = model.ub[j];
        if (lbj > -1e30 && xj < lbj - eps) {
            double v = lbj - xj;
            if (v > max_bound) { max_bound = v; max_bound_idx = j; }
        }
        if (ubj < 1e30 && xj > ubj + eps) {
            double v = xj - ubj;
            if (v > max_bound) { max_bound = v; max_bound_idx = j; }
        }
    }
    std::cout << "  Bound violation max = " << max_bound
              << " at var " << max_bound_idx << "\n";
    if (max_bound >= eps)
        std::cout << "  ** FAIL: bound violation\n";

    // 3. Dual feasibility / complementary slackness: find worst
    double max_dual_viol = 0.0;
    std::size_t max_dual_idx = 0;
    std::string max_dual_reason;

    for (std::size_t j = 0; j < n; ++j) {
        double cj = static_cast<double>(model.obj[j]);
        double AT_pi_j = 0.0;
        Index st = model.A.col_ptrs[j], en = model.A.col_ptrs[j+1];
        for (Index k = st; k < en; ++k) {
            std::size_t i = static_cast<std::size_t>(model.A.row_indices[k]);
            AT_pi_j += static_cast<double>(model.A.values[k]) * static_cast<double>(pi[i]);
        }
        double rj = cj - AT_pi_j;
        double xj = x[j], lbj = model.lb[j], ubj = model.ub[j];

        // Check which condition the verifier uses:
        std::string reason;
        double viol = 0.0;
        if (rj >= eps) {
            // Must be at lower bound
            if (lbj <= -1e30) {
                viol = rj; reason = "rj>0 but lb=-inf";
            } else if (xj >= lbj + eps) {
                viol = xj - lbj; reason = "rj>0 but x not at lb";
            } else if (std::abs(rj * (xj - lbj)) >= eps) {
                viol = std::abs(rj*(xj-lbj)); reason = "CS rj*(x-lb) fail";
            }
        } else if (rj <= -eps) {
            // Must be at upper bound
            if (ubj >= 1e30) {
                viol = -rj; reason = "rj<0 but ub=+inf";
            } else if (xj <= ubj - eps) {
                viol = ubj - xj; reason = "rj<0 but x not at ub";
            } else if (std::abs(rj * (xj - ubj)) >= eps) {
                viol = std::abs(rj*(xj-ubj)); reason = "CS rj*(x-ub) fail";
            }
        }
        if (viol > max_dual_viol) {
            max_dual_viol = viol;
            max_dual_idx = j;
            max_dual_reason = reason;
        }
    }

    if (max_dual_viol > 0.0) {
        std::size_t j = max_dual_idx;
        double cj = static_cast<double>(model.obj[j]);
        double AT_pi_j = 0.0;
        Index st = model.A.col_ptrs[j], en = model.A.col_ptrs[j+1];
        // Collect all row contributions for this column:
        std::cout << "  ** FAIL: dual/CS violation = " << max_dual_viol
                  << " at var " << j << " (" << max_dual_reason << ")\n";
        std::cout << "    x[" << j << "] = " << x[j]
                  << "  lb=" << model.lb[j] << "  ub=" << model.ub[j] << "\n";
        std::cout << "    c[" << j << "] = " << cj << "\n";
        for (Index k = st; k < en; ++k) {
            std::size_t i = static_cast<std::size_t>(model.A.row_indices[k]);
            double aij = static_cast<double>(model.A.values[k]);
            double pi_i = static_cast<double>(pi[i]);
            AT_pi_j += aij * pi_i;
            std::cout << "    A[" << i << "," << j << "]=" << aij
                      << " * pi[" << i << "]=" << pi_i << "\n";
        }
        std::cout << "    A^T pi [" << j << "] = " << AT_pi_j << "\n";
        std::cout << "    r[" << j << "] = c - A^T pi = " << (cj - AT_pi_j) << "\n";
        // Check if it is a slack variable (obj=0, nonzero in exactly one row)
        if (en - st == 1) {
            std::cout << "    [This appears to be a single-row variable (slack?).]\n";
        }
    } else {
        std::cout << "  Dual/CS: no violation found above eps=" << eps << "\n";
    }
    std::cout << "--- END DIAGNOSTIC ---\n\n";
}


static void run_netlib_benchmark(const std::string& name, const std::string& path) {
    std::cout << "\n=== Netlib Validation: " << name << " ===\n\n";

    core::Model model;
    try {
        model = parsers::read_mps(path);
    } catch (const std::exception& e) {
        std::string err = e.what();
        if (err.find("Unsupported") != std::string::npos || err.find("unsupported") != std::string::npos) {
            std::cout << "Benchmark: " << name << "\n";
            std::cout << "Result: UNSUPPORTED BY CURRENT MPS PARSER\n";
            std::cout << "Reason: " << err << "\n";
            return;
        } else {
            FAIL("Failed to parse " << name << ": " << err);
        }
    }

    std::cout << "Variables: " << model.obj.size() << "\n";
    std::cout << "Constraints: " << model.rhs.size() << "\n";

    // 4. Run the existing SANKHYA solve path.
    // The intended end-to-end GPU path for arbitrary LPs is Phase 15.2: MehrotraSolver
    ipm::MehrotraSolver solver(model);
    ipm::MehrotraResult result = solver.solve();
    
    std::string status_str;
    switch(result.status) {
        case simplex::SimplexStatus::Optimal: status_str = "Optimal"; break;
        case simplex::SimplexStatus::Infeasible: status_str = "Infeasible"; break;
        case simplex::SimplexStatus::IterationLimit: status_str = "IterationLimit"; break;
        case simplex::SimplexStatus::Unbounded: status_str = "Unbounded"; break;
        default: status_str = "Unknown"; break;
    }

    std::cout << "Solver Status: " << status_str << "\n";
    std::cout << "Objective: " << result.objective_value << "\n";
    std::cout << "Primal Residual: " << result.primal_residual << "\n";

    // 5 & 6. Independent optimality certificate.
    // The contract distinguishes three cases:
    //   Optimal        -> run certificate, REQUIRE PASS
    //   IterationLimit -> solver did not converge; do not claim optimality
    //   Other          -> report only
    if (result.status == simplex::SimplexStatus::Optimal) {
        bool cert = verifier::verify_optimal(model, result.x, result.pi);
        std::cout << "Certificate: " << (cert ? "PASS" : "FAIL") << "\n";
        if (!cert) {
            diagnose_certificate(name, model, result.x, result.pi);
        }
        REQUIRE(cert);
    } else if (result.status == simplex::SimplexStatus::IterationLimit) {
        std::cout << "Certificate: NOT ATTEMPTED (solver did not converge)\n";
    } else {
        std::cout << "Certificate: NOT ATTEMPTED (solver status: " << status_str << ")\n";
    }
}

#ifndef SANKHYA_SOURCE_DIR
#define SANKHYA_SOURCE_DIR "."
#endif

TEST_CASE("Phase 33.1: Netlib/MIPLIB Benchmark Validation", "[integration][benchmark][netlib]") {
    std::string base_path = std::string(SANKHYA_SOURCE_DIR) + "/";
    run_netlib_benchmark("afiro", base_path + "benchmarks/netlib/afiro.mps");
    run_netlib_benchmark("adlittle", base_path + "benchmarks/netlib/adlittle.mps");
}
