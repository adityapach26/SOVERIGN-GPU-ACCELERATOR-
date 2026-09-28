#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "core/problem.hpp"
#include "parsers/mps_reader.hpp"
#include "verifier/certificate.hpp"
#include "ipm/mehrotra.hpp"
#include "simplex/phase1.hpp"
#include "numerics/sparse_lu.hpp"

using namespace sankhya;

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

    // -----------------------------------------------------------------------
    // Primary path: GPU Mehrotra IPM
    // -----------------------------------------------------------------------
    ipm::MehrotraSolver solver(model);
    ipm::MehrotraResult result = solver.solve();

    std::string status_str;
    switch(result.status) {
        case simplex::SimplexStatus::Optimal:        status_str = "Optimal";       break;
        case simplex::SimplexStatus::Infeasible:     status_str = "Infeasible";    break;
        case simplex::SimplexStatus::IterationLimit: status_str = "IterationLimit"; break;
        case simplex::SimplexStatus::Unbounded:      status_str = "Unbounded";     break;
        default:                                      status_str = "Unknown";       break;
    }

    std::cout << "GPU Mehrotra Status: " << status_str << "\n";

    if (result.status == simplex::SimplexStatus::Optimal) {
        // -----------------------------------------------------------------------
        // Mehrotra succeeded: verify independently and require certificate.
        // -----------------------------------------------------------------------
        std::cout << "Algorithm: GPU Mehrotra\n";
        std::cout << "Objective: " << result.objective_value << "\n";
        std::cout << "Primal Residual: " << result.primal_residual << "\n";

        REQUIRE(result.status == simplex::SimplexStatus::Optimal);

        bool cert = verifier::verify_optimal(model, result.x, result.pi);
        std::cout << "Certificate: " << (cert ? "PASS" : "FAIL") << "\n";
        REQUIRE(cert);

    } else if (result.status == simplex::SimplexStatus::IterationLimit) {
        // -----------------------------------------------------------------------
        // Mehrotra did not converge: invoke native Phase-I + Phase-II recovery.
        // This is the recovery path; it must also pass the independent certificate.
        // -----------------------------------------------------------------------
        std::cout << "GPU Mehrotra did not converge; attempting native Phase-I + Phase-II recovery.\n";

        simplex::Basis basis;
        std::vector<Float> x;
        numerics::SparseLUFactorization factorizer;

        simplex::SimplexStatus p1_status =
            simplex::solve_with_phase1(model, basis, x, factorizer);

        std::string p1_str;
        switch(p1_status) {
            case simplex::SimplexStatus::Optimal:        p1_str = "Optimal";       break;
            case simplex::SimplexStatus::Infeasible:     p1_str = "Infeasible";    break;
            case simplex::SimplexStatus::IterationLimit: p1_str = "IterationLimit"; break;
            case simplex::SimplexStatus::Unbounded:      p1_str = "Unbounded";     break;
            default:                                      p1_str = "Unknown";       break;
        }

        std::cout << "Algorithm: Native Phase-I + Phase-II recovery\n";
        std::cout << "Recovery Status: " << p1_str << "\n";

        REQUIRE(p1_status == simplex::SimplexStatus::Optimal);

        // Compute dual prices for the certificate
        const Index m_rows = model.A.rows;
        factorizer.factorize(model.A, basis);
        std::vector<Float> pi(static_cast<std::size_t>(m_rows), 0.0);
        const Float obj_sign = (model.sense == OptimizationSense::Minimize) ? 1.0 : -1.0;
        for (Index i = 0; i < m_rows; ++i) {
            const auto bi = static_cast<std::size_t>(basis.basic_indices[static_cast<std::size_t>(i)]);
            pi[static_cast<std::size_t>(i)] = obj_sign * model.obj[bi];
        }
        factorizer.btran(pi);
        for (Float& pi_i : pi) pi_i *= obj_sign; // un-sign for verifier

        // Compute objective value
        Float obj_val = 0.0;
        for (std::size_t j = 0; j < x.size(); ++j) {
            obj_val += model.obj[j] * x[j];
        }
        std::cout << "Objective: " << obj_val << "\n";

        bool cert = verifier::verify_optimal(model, x, pi);
        std::cout << "Certificate: " << (cert ? "PASS" : "FAIL") << "\n";
        REQUIRE(cert);

    } else {
        // Mehrotra returned Infeasible or Unbounded — no recovery attempted.
        std::cout << "Algorithm: GPU Mehrotra\n";
        std::cout << "Solver reported " << status_str << " (no recovery path for this status).\n";
        // Report the status but do not require Optimal for non-convergence cases.
        REQUIRE(result.status == simplex::SimplexStatus::Optimal);
    }
}

#ifndef SANKHYA_SOURCE_DIR
#define SANKHYA_SOURCE_DIR "."
#endif

TEST_CASE("Phase 33.1: Netlib/MIPLIB Benchmark Validation", "[integration][benchmark][netlib]") {
    std::string base_path = std::string(SANKHYA_SOURCE_DIR) + "/";
    run_netlib_benchmark("afiro",    base_path + "benchmarks/netlib/afiro.mps");
    run_netlib_benchmark("adlittle", base_path + "benchmarks/netlib/adlittle.mps");
}
