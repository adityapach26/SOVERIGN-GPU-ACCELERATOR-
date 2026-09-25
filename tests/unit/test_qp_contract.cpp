#include <catch2/catch_test_macros.hpp>
#include "qp/qp_model.hpp"
#include "qp/qp_solver.hpp"
#include "core/problem.hpp"

using namespace sankhya;
using namespace sankhya::qp;

TEST_CASE("Phase 16A.1: QP Mathematical Contract", "[qp][contract]") {
    QPModel qp;
    
    // Setup a 2-variable problem
    qp.lp_part.A.cols = 2;
    qp.lp_part.A.rows = 1;
    qp.lp_part.sense = OptimizationSense::Minimize;
    qp.lp_part.lb = {0.0, 0.0};
    qp.lp_part.ub = {1.0, 1.0};
    
    // A. Canonical model construction & E. Convexity boundary
    SECTION("Valid PSD Convex QP is accepted") {
        qp.H.rows = 2;
        qp.H.cols = 2;
        qp.convexity = QPConvexityStatus::ConvexPSD;
        
        REQUIRE_NOTHROW(qp.validate());
    }

    // B. Objective convention
    SECTION("Maximization QP is rejected in canonical form") {
        qp.H.rows = 2;
        qp.H.cols = 2;
        qp.lp_part.sense = OptimizationSense::Maximize;
        qp.convexity = QPConvexityStatus::ConvexPSD;
        
        REQUIRE_THROWS_AS(qp.validate(), std::invalid_argument);
    }

    // C. Dimension validation
    SECTION("Hessian dimension mismatch is rejected") {
        qp.H.rows = 3; // Mismatch
        qp.H.cols = 3;
        qp.convexity = QPConvexityStatus::ConvexPSD;
        
        REQUIRE_THROWS_AS(qp.validate(), std::invalid_argument);
    }

    // D. Bound validation
    SECTION("Invalid bounds are rejected") {
        qp.H.rows = 2;
        qp.H.cols = 2;
        qp.lp_part.lb = {2.0, 0.0};
        qp.lp_part.ub = {1.0, 1.0}; // lb > ub for var 0
        qp.convexity = QPConvexityStatus::ConvexPSD;
        
        REQUIRE_THROWS_AS(qp.validate(), std::invalid_argument);
    }

    // F. Indefinite Hessian & H. Solver contract
    SECTION("Nonconvex / Indefinite QP is explicitly marked unsupported") {
        qp.H.rows = 2;
        qp.H.cols = 2;
        qp.convexity = QPConvexityStatus::Nonconvex;
        
        // Model validation throws
        REQUIRE_THROWS_AS(qp.validate(), std::invalid_argument);

        // Solver returns Unsupported status rather than silently downgrading
        QPSolver solver;
        std::vector<Float> x, y, z;
        QPSolverStatus status = solver.solve(qp, x, y, z);
        
        REQUIRE(status == QPSolverStatus::UnsupportedNonconvex);
    }

    // G. No silent fallback
    SECTION("Unknown convexity defaults to unsupported") {
        qp.H.rows = 2;
        qp.H.cols = 2;
        qp.convexity = QPConvexityStatus::Unknown; // Default
        
        QPSolver solver;
        std::vector<Float> x, y, z;
        QPSolverStatus status = solver.solve(qp, x, y, z);
        
        // Cannot solve unless explicitly validated as ConvexPSD
        REQUIRE(status == QPSolverStatus::UnsupportedNonconvex);
    }
}

