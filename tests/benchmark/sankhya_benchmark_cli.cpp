#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "core/problem.hpp"
#include "parsers/mps_reader.hpp"
#include "verifier/certificate.hpp"
#include "ipm/mehrotra.hpp"
#include "simplex/phase1.hpp"
#include "numerics/sparse_lu.hpp"

using namespace sankhya;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: sankhya_benchmark_cli <mps_file>\n";
        return 1;
    }

    std::string path = argv[1];
    
    core::Model model;
    try {
        model = parsers::read_mps(path);
    } catch (const std::exception& e) {
        std::cout << "{\"error\": \"" << e.what() << "\"}\n";
        return 0;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    
    ipm::MehrotraSolver solver(model);
    ipm::MehrotraResult result = solver.solve();

    std::string status_str;
    Float obj_val = 0.0;
    Float prim_res = 0.0;
    bool cert_pass = false;
    Index iterations = result.iterations;

    if (result.status == simplex::SimplexStatus::Optimal) {
        status_str = "Optimal";
        obj_val = result.objective_value;
        prim_res = result.primal_residual;
        cert_pass = verifier::verify_optimal(model, result.x, result.pi);
    } else if (result.status == simplex::SimplexStatus::IterationLimit) {
        status_str = "NumericalFailure";
    } else {
        status_str = "Failed";
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> solve_time_ms = end_time - start_time;

    // JSON output
    std::cout << "{\n";
    std::cout << "  \"solver\": \"SANKHYA\",\n";
    std::cout << "  \"status\": \"" << status_str << "\",\n";
    std::cout << "  \"solve_time_ms\": " << solve_time_ms.count() << ",\n";
    std::cout << "  \"iterations\": " << iterations << ",\n";
    std::cout << "  \"objective\": " << obj_val << ",\n";
    std::cout << "  \"primal_residual\": " << prim_res << ",\n";
    std::cout << "  \"variables\": " << model.obj.size() << ",\n";
    std::cout << "  \"constraints\": " << model.rhs.size() << ",\n";
    std::cout << "  \"nnz\": " << model.A.values.size() << ",\n";
    std::cout << "  \"certificate_pass\": " << (cert_pass ? "true" : "false") << ",\n";
    std::cout << "  \"fallback_triggered\": \"NO\"\n";
    std::cout << "}\n";

    return 0;
}

