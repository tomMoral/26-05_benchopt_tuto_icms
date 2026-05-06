import pytest


def check_test_solver_run(benchmark, solver):
    if solver.name.lower() == "mysolver":
        pytest.skip("mysolver is not implemented yet")
