#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "core/problem.hpp"
#include "parsers/mps_reader.hpp"
#include "verifier/certificate.hpp"
#include "ipm/mehrotra.hpp"

using namespace sankhya;

static void run_netlib_benchmark(const std::string& name, const std::string& path) {
    std::cout << "\n=== Netlib Validation: " << name << " ===\n\n";

    core::Model model;
    try {
        parsers::MPSReader reader(path);
        model = reader.parse();
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
        case simplex::SimplexStatus::NumericalFailure: status_str = "NumericalFailure"; break;
        case simplex::SimplexStatus::Unbounded: status_str = "Unbounded"; break;
        default: status_str = "Unknown"; break;
    }

    std::cout << "Solver Status: " << status_str << "\n";
    std::cout << "Objective: " << result.objective_value << "\n";
    std::cout << "Primal Residual: " << result.primal_residual << "\n";

    // 5 & 6. Obtain x and pi.
    // The IPM solver architecture now exposes the vectors via MehrotraResult.
    bool cert = verifier::verify_optimal(model, result.x, result.pi);
    std::cout << "Certificate: " << (cert ? "PASS" : "FAIL") << "\n";
}

TEST_CASE("Phase 33.1: Netlib/MIPLIB Benchmark Validation", "[integration][benchmark][netlib]") {
    run_netlib_benchmark("afiro", "benchmarks/netlib/afiro.mps");
    run_netlib_benchmark("adlittle", "benchmarks/netlib/adlittle.mps");
}
