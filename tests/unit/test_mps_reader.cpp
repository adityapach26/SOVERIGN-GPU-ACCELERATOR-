#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "parsers/mps_reader.hpp"
#include <fstream>
#include <cstdio>

using namespace sankhya;

TEST_CASE("Phase 29.1: MPS Reader - Basic constraints and bounds", "[parsers][mps]") {
    const char* filename = "test_fixture.mps";
    {
        std::ofstream out(filename);
        out << "NAME          TESTPROB\n";
        out << "ROWS\n";
        out << " N  COST\n";
        out << " L  LIM1\n";
        out << " G  LIM2\n";
        out << " E  MYEQN\n";
        out << "COLUMNS\n";
        out << "    XONE      COST                 1   LIM1                 1\n";
        out << "    XONE      LIM2                 1\n";
        out << "    YTWO      COST                 4   LIM1                 1\n";
        out << "    YTWO      MYEQN               -1\n";
        out << "    ZTHREE    COST                 9   LIM2                 1\n";
        out << "    ZTHREE    MYEQN                1\n";
        out << "RHS\n";
        out << "    RHS1      LIM1                 5   LIM2                10\n";
        out << "    RHS1      MYEQN                7\n";
        out << "BOUNDS\n";
        out << " UP BND1      XONE                 4\n";
        out << " LO BND1      YTWO                -1\n";
        out << " UP BND1      YTWO                 1\n";
        out << "ENDATA\n";
    }

    core::Model m = parsers::read_mps(filename);
    std::remove(filename);

    REQUIRE(m.obj.size() == 5); // 3 vars + 2 slacks
    REQUIRE(m.vtype.size() == 5);
    REQUIRE(m.rhs.size() == 3);

    // Obj coeffs
    REQUIRE(m.obj[0] == Catch::Approx(1.0));
    REQUIRE(m.obj[1] == Catch::Approx(4.0));
    REQUIRE(m.obj[2] == Catch::Approx(9.0));

    // RHS
    REQUIRE(m.rhs[0] == Catch::Approx(5.0));
    REQUIRE(m.rhs[1] == Catch::Approx(10.0));
    REQUIRE(m.rhs[2] == Catch::Approx(7.0));

    // Bounds
    REQUIRE(m.lb[0] == Catch::Approx(0.0)); // Default
    REQUIRE(m.ub[0] == Catch::Approx(4.0)); // UP

    REQUIRE(m.lb[1] == Catch::Approx(-1.0)); // LO
    REQUIRE(m.ub[1] == Catch::Approx(1.0)); // UP

    REQUIRE(m.lb[2] == Catch::Approx(0.0)); // Default
    REQUIRE(m.ub[2] == math::kInfinity);

    // Check slacks
    REQUIRE(m.obj[3] == Catch::Approx(0.0));
    REQUIRE(m.obj[4] == Catch::Approx(0.0));
}

TEST_CASE("Phase 29.1: MPS Reader - Integer markers", "[parsers][mps]") {
    const char* filename = "test_int_fixture.mps";
    {
        std::ofstream out(filename);
        out << "NAME INTPROB\n";
        out << "ROWS\n";
        out << " N  OBJ\n";
        out << " L  C1\n";
        out << "COLUMNS\n";
        out << "    MARK0000  'MARKER'                 'INTORG'\n";
        out << "    X1        OBJ                  1   C1                   1\n";
        out << "    MARK0001  'MARKER'                 'INTEND'\n";
        out << "    X2        OBJ                  2   C1                   2\n";
        out << "RHS\n";
        out << "    RHS1      C1                   5\n";
        out << "BOUNDS\n";
        out << "ENDATA\n";
    }

    core::Model m = parsers::read_mps(filename);
    std::remove(filename);

    // X1 (integer), X2 (continuous), S1 (continuous)
    REQUIRE(m.vtype.size() == 3);
    REQUIRE(m.vtype[0] == VariableType::Integer);
    REQUIRE(m.vtype[1] == VariableType::Continuous);
    REQUIRE(m.vtype[2] == VariableType::Continuous);
}
