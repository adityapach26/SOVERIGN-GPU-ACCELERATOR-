#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>

#include "core/problem.hpp"
#include "milp/branching.hpp"
#include "milp/tree.hpp"
#include "simplex/primal.hpp"
#include "numerics/sparse_lu.hpp"
#include "mrpl/mccormick.hpp"

using namespace sankhya;

// Helper to evaluate a model using the existing node-solving path (CPU version)
Float evaluate_model(const core::Model& model) {
    // We use primal_simplex_phase2 to solve it
    // First, we need an initial basic feasible solution.
    // For our simple test (w <= 4y etc.), the origin (0,0,0) with slacks is feasible.
    // Let's ensure the model is simple enough that slacks are a valid basis.
    
    // We will build a small factorizer and basis
    simplex::Basis basis;
    // Basic variables are the slacks (one for each constraint)
    const Index m = static_cast<Index>(model.rhs.size());
    const Index n = static_cast<Index>(model.vtype.size());
    for (Index i = 0; i < m; ++i) {
        basis.basic_indices.push_back(n - m + i); // Assuming slacks are at the end
    }
    
    std::vector<Float> x(n, 0.0);
    // Initialize x
    for (Index i = 0; i < m; ++i) {
        x[static_cast<std::size_t>(n - m + i)] = model.rhs[static_cast<std::size_t>(i)]; // x_slack = rhs
    }

    numerics::SparseLUFactorization factorizer;
    simplex::SimplexStatus status = simplex::primal_simplex_phase2(model, basis, x, factorizer);
    
    if (status != simplex::SimplexStatus::Optimal) {
        return math::kInfinity;
    }

    Float obj = 0.0;
    for (Index i = 0; i < n; ++i) {
        obj += model.obj[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
    }
    return obj;
}

// Helper to build the parent model
core::Model build_parent_model() {
    core::Model m;
    
    // Variables: x (0), y (1), w (2), s1 (3), s2 (4), s3 (5), s4 (6)
    // objective: minimize -w (maximize w)
    m.add_variable(0.0, 0.0, 10.0); // x
    m.add_variable(0.0, 0.0, 10.0); // y
    m.add_variable(-1.0, 0.0, 100.0); // w
    
    // Slacks for the 4 inequalities (<= forms)
    m.add_variable(0.0, 0.0, math::kInfinity);
    m.add_variable(0.0, 0.0, math::kInfinity);
    m.add_variable(0.0, 0.0, math::kInfinity);
    m.add_variable(0.0, 0.0, math::kInfinity);

    // Initial McCormick for [0,10] x [0,10]
    auto cuts = mrpl::mccormick_envelopes(0.0, 10.0, 0.0, 10.0);
    
    for (int i = 0; i < 4; ++i) {
        const auto& cut = cuts[static_cast<std::size_t>(i)];
        std::vector<Index> cols = {0, 1, 2, 3 + i};
        std::vector<Float> vals = {cut.coefficients[0], cut.coefficients[1], cut.coefficients[2], 1.0};
        m.add_constraint(cols, vals, cut.rhs);
    }
    
    core::BilinearTerm term;
    term.x_col = 0;
    term.y_col = 1;
    term.w_col = 2;
    term.mccormick_rows = {0, 1, 2, 3};
    m.bilinear_terms.push_back(term);
    
    m.finalize();
    return m;
}

TEST_CASE("Phase 26.1: Spatial branching domain split", "[milp][branching]") {
    core::Model parent_model = build_parent_model();
    
    milp::MILPNode parent_node;
    parent_node.hbf_id = 0;
    parent_node.parent_bound = -100.0;
    parent_node.depth = 0;
    
    milp::NodeQueue queue;
    
    // Split on x (var 0) at value 4.0
    milp::branch_spatial(parent_node, parent_model, parent_model.bilinear_terms[0], 0, 4.0, queue);
    
    // Should enqueue 2 children
    REQUIRE(!queue.empty());
    
    milp::MILPNode child1 = queue.pop();
    milp::MILPNode child2 = queue.pop();
    REQUIRE(queue.empty());
    
    // One is [0, 4], the other is [4, 10].
    // Since queue is max-heap by parent_bound, and both inherit -100, order is arbitrary, 
    // so we check both.
    bool found_left = false;
    bool found_right = false;
    
    auto check_child = [&](const milp::MILPNode& node) {
        REQUIRE(node.depth == 1);
        REQUIRE(node.spatial_model != nullptr);
        
        Float lb = node.spatial_model->lb[0];
        Float ub = node.spatial_model->ub[0];
        
        if (lb == 0.0 && ub == Catch::Approx(4.0)) {
            found_left = true;
        } else if (lb == Catch::Approx(4.0) && ub == 10.0) {
            found_right = true;
        }
        
        // y bounds should remain [0, 10]
        REQUIRE(node.spatial_model->lb[1] == 0.0);
        REQUIRE(node.spatial_model->ub[1] == Catch::Approx(10.0));
    };
    
    check_child(child1);
    check_child(child2);
    
    REQUIRE(found_left);
    REQUIRE(found_right);
}

TEST_CASE("Phase 26.1: Spatial branching boundary rejection", "[milp][branching]") {
    core::Model parent_model = build_parent_model();
    milp::MILPNode parent_node;
    milp::NodeQueue queue;
    
    // Attempt to split exactly at the upper bound (10.0)
    milp::branch_spatial(parent_node, parent_model, parent_model.bilinear_terms[0], 0, 10.0, queue);
    
    // Should reject the split and enqueue nothing
    REQUIRE(queue.empty());
}

TEST_CASE("Phase 26.1: Relaxation tightening", "[milp][branching]") {
    core::Model parent_model = build_parent_model();
    
    Float parent_bound = evaluate_model(parent_model);
    REQUIRE(parent_bound == Catch::Approx(-100.0)); // w max is 100 -> min -w is -100
    
    milp::MILPNode parent_node;
    parent_node.hbf_id = 0;
    parent_node.parent_bound = parent_bound;
    parent_node.depth = 0;
    
    milp::NodeQueue queue;
    milp::branch_spatial(parent_node, parent_model, parent_model.bilinear_terms[0], 0, 4.0, queue);
    
    milp::MILPNode child1 = queue.pop();
    milp::MILPNode child2 = queue.pop();
    
    // One child has x in [0,4], so max w is 40 -> min -w is -40
    // The other has x in [4,10], so max w is 100 -> min -w is -100
    Float bound1 = evaluate_model(*child1.spatial_model);
    Float bound2 = evaluate_model(*child2.spatial_model);
    
    bool has_tightened = (bound1 == Catch::Approx(-40.0) || bound2 == Catch::Approx(-40.0));
    bool has_original = (bound1 == Catch::Approx(-100.0) || bound2 == Catch::Approx(-100.0));
    
    REQUIRE(has_tightened);
    REQUIRE(has_original);
}
