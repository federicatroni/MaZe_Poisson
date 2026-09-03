import ctypes
from pathlib import Path

import numpy as np
import pytest

from maze_poisson.constants import a0
from maze_poisson.myio.input import MDVariables


def _load_kernel():
    candidates = [
        Path("library/build/libmaze_poisson.so"),
        Path("library/build/libmaze_poisson.dylib"),
        Path("maze_poisson/libmaze_poisson.so"),
        Path("maze_poisson/libmaze_poisson.dylib"),
    ]
    library_path = next((path for path in candidates if path.exists()), None)
    if library_path is None:
        pytest.skip("maze_poisson shared library has not been built")
    library = ctypes.CDLL(str(library_path.resolve()))
    kernel = library.compute_ribar_window_pairwise_correction
    float_array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    kernel.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        float_array,
        float_array,
        float_array,
        float_array,
    ]
    kernel.restype = ctypes.c_double
    return kernel


def _evaluate(kernel, distance, charges=(1.0, -1.0)):
    a0_ang = 0.529177210903
    positions = np.array(
        [[0.0, 0.0, 0.0], [distance / a0_ang, 0.0, 0.0]],
        dtype=np.float64,
    ).ravel()
    forces = np.zeros(6, dtype=np.float64)
    energy = kernel(
        2,
        100.0 / a0_ang,
        3.0 / a0_ang,
        23.0,
        80.0,
        np.asarray(charges, dtype=np.float64),
        positions,
        np.array([1.4, 2.0], dtype=np.float64) / a0_ang,
        forces,
    )
    return energy, forces


@pytest.mark.parametrize("distance", [3.0, 4.2, 5.5])
def test_ribar_force_is_energy_derivative_and_obeys_newton_third_law(distance):
    kernel = _load_kernel()
    a0_ang = 0.529177210903
    step = 1.0e-6
    energy_plus, _ = _evaluate(kernel, distance + step * a0_ang)
    energy_minus, _ = _evaluate(kernel, distance - step * a0_ang)
    _, forces = _evaluate(kernel, distance)
    numerical_derivative = (energy_plus - energy_minus) / (2.0 * step)

    assert forces[0] == pytest.approx(numerical_derivative, abs=1.0e-10)
    assert forces[0] == pytest.approx(-forces[3], abs=1.0e-14)


def test_ribar_window_is_pair_specific_and_zero_beyond_contact_plus_delta():
    kernel = _load_kernel()
    energy_inside, _ = _evaluate(kernel, 6.3)  # Na-Cl contact 3.4 A, upper edge 6.4 A
    energy_outside, forces_outside = _evaluate(kernel, 6.5)

    assert energy_inside != pytest.approx(0.0)
    assert energy_outside == pytest.approx(0.0, abs=1.0e-15)
    assert forces_outside == pytest.approx(np.zeros(6), abs=1.0e-15)


def test_ribar_window_boundaries_are_continuous_in_energy():
    kernel = _load_kernel()
    contact = 3.4
    upper = contact + 3.0
    step = 1.0e-7

    energy_contact, _ = _evaluate(kernel, contact)
    energy_contact_outside, _ = _evaluate(kernel, contact + step)
    energy_upper_inside, _ = _evaluate(kernel, upper - step)
    energy_upper, forces_upper = _evaluate(kernel, upper)

    assert energy_contact_outside == pytest.approx(energy_contact, abs=1.0e-8)
    assert energy_upper_inside == pytest.approx(0.0, abs=1.0e-8)
    assert energy_upper == pytest.approx(0.0, abs=1.0e-15)
    assert forces_upper == pytest.approx(np.zeros(6), abs=1.0e-15)


def test_ribar_window_strengthens_both_signs_at_contact():
    kernel = _load_kernel()
    unlike_energy, unlike_forces = _evaluate(kernel, 3.0, charges=(1.0, -1.0))
    like_energy, like_forces = _evaluate(kernel, 3.0, charges=(1.0, 1.0))

    assert unlike_energy < 0.0
    assert like_energy > 0.0
    assert unlike_forces[0] == pytest.approx(-like_forces[0])


def test_ribar_yaml_units_and_model_validation():
    mdv = MDVariables.from_dict({
        "N_steps": 10,
        "T": 298.15,
        "dt_fs": 2.0,
        "electrostatic_model": "ribar_window_pairwise",
        "ribar_window_ang": 3.0,
    })

    assert mdv.electrostatic_model == "RIBAR_WINDOW_PAIRWISE"
    assert mdv.ribar_window * a0 == pytest.approx(3.0)

    with pytest.raises(ValueError, match="requires poisson_boltzmann=false"):
        MDVariables.from_dict({
            "N_steps": 10,
            "T": 298.15,
            "dt_fs": 2.0,
            "poisson_boltzmann": True,
            "electrostatic_model": "RIBAR_WINDOW_PAIRWISE",
        })
