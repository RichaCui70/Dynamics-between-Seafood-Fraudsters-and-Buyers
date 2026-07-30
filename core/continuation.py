"""Pseudo-arclength continuation of fixed points for the discrete-time map."""

import numpy as np

from core.System import DynamicalSystem
from core.constants import DEFAULT_INIT_STATE

_STATE_VARS = ('S', 'E', 'F', 'FP')
_STABILITY_MARGIN = 1e-6
_CLOSE_TO_ZERO = 1e-14
_CLOSE_TO_ONE = 1.0 - 1e-14


def _clamp_state(state_vector):
    """Clamp a length-4 state into the physically valid open box."""
    return np.array([
        max(float(state_vector[0]), _CLOSE_TO_ZERO),
        max(float(state_vector[1]), _CLOSE_TO_ZERO),
        min(max(float(state_vector[2]), _CLOSE_TO_ZERO), _CLOSE_TO_ONE),
        min(max(float(state_vector[3]), _CLOSE_TO_ZERO), _CLOSE_TO_ONE),
    ], dtype=np.float64)


def _build_system(params_at, mu, initial_state, equation_form):
    """Build a DynamicalSystem at parameter value mu."""
    state = {key: np.float128(value) for key, value in initial_state.items()}
    return DynamicalSystem(params_at(float(mu)), state, equation_form)


def _map_residual(params_at, state_vector, mu, initial_state, equation_form):
    """Return G(x; μ) − x for the clamped state."""
    clamped = _clamp_state(state_vector)
    system = _build_system(params_at, mu, initial_state, equation_form)
    return system._evaluate_map_vector(clamped) - clamped


def _jacobian_state(params_at, state_vector, mu, initial_state, equation_form):
    """Return ∂G/∂x at (x, μ)."""
    clamped = _clamp_state(state_vector)
    system = _build_system(params_at, mu, initial_state, equation_form)
    state_dict = {key: float(value) for key, value in zip(_STATE_VARS, clamped)}
    return np.asarray(system.jacobian(state=state_dict), dtype=np.float64)


def _jacobian_parameter(params_at, state_vector, mu, initial_state, equation_form,
                        param_scale):
    """Return ∂G/∂μ via central finite differences in the continuation parameter."""
    clamped = _clamp_state(state_vector)
    step = max(1e-6, 1e-6 * abs(param_scale), 1e-6 * max(abs(mu), 1.0))
    map_forward = _build_system(
        params_at, mu + step, initial_state, equation_form,
    )._evaluate_map_vector(clamped)
    map_backward = _build_system(
        params_at, mu - step, initial_state, equation_form,
    )._evaluate_map_vector(clamped)
    return (map_forward - map_backward) / (2.0 * step)


def _scaling_matrix(state_vector, mu, param_range_width):
    """Diagonal scaling so state and parameter contribute comparably to arclength."""
    scales = np.array([
        max(1.0, abs(float(state_vector[0]))),
        max(1.0, abs(float(state_vector[1]))),
        max(1.0, abs(float(state_vector[2]))),
        max(1.0, abs(float(state_vector[3]))),
        max(1e-6, abs(float(param_range_width))),
    ], dtype=np.float64)
    return np.diag(scales)


def _null_space_tangent(jacobian_x, jacobian_mu, scaling, previous_tangent=None,
                        preferred_mu_sign=1.0):
    """
    Compute a unit tangent in scaled coordinates via SVD of the augmented Jacobian.

    Args:
        jacobian_x: 4×4 state Jacobian ∂G/∂x.
        jacobian_mu: length-4 parameter Jacobian ∂G/∂μ.
        scaling: 5×5 diagonal scaling matrix D.
        previous_tangent: previous raw (unscaled) tangent for sign continuity.
        preferred_mu_sign: sign of dμ preferred on the first step.

    Returns:
        length-5 raw (unscaled) tangent vector [dx; dμ].
    """
    identity = np.eye(4, dtype=np.float64)
    augmented = np.column_stack([jacobian_x - identity, jacobian_mu])
    # Work in scaled coordinates: A_scaled = A @ D so singular vectors are fair.
    scaled_augmented = augmented @ scaling
    _, _, right_singular_vectors = np.linalg.svd(scaled_augmented, full_matrices=True)
    scaled_tangent = right_singular_vectors[-1, :].astype(np.float64)
    raw_tangent = scaling @ scaled_tangent
    norm = np.linalg.norm(raw_tangent)
    if not np.isfinite(norm) or norm < 1e-18:
        return None
    raw_tangent = raw_tangent / norm

    if previous_tangent is not None:
        if float(np.dot(raw_tangent, previous_tangent)) < 0.0:
            raw_tangent = -raw_tangent
    elif preferred_mu_sign != 0.0 and np.sign(raw_tangent[4]) != np.sign(preferred_mu_sign):
        raw_tangent = -raw_tangent
    return raw_tangent


def _classify_stability(jacobian_x):
    """Return (is_stable, spectral_radius) from the state Jacobian."""
    if not np.all(np.isfinite(jacobian_x)):
        return False, np.inf
    eigenvalues = np.linalg.eig(jacobian_x)[0]
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    is_stable = spectral_radius < 1.0 - _STABILITY_MARGIN
    return is_stable, spectral_radius


def _newton_corrector(params_at, predicted_state, predicted_mu, previous_state,
                      previous_mu, step_size, tangent, initial_state, equation_form,
                      param_range_width, tol=1e-9, max_iterations=25):
    """
    Newton corrector for the pseudo-arclength augmented system.

    Solves [G(x;μ)−x ; N(x,μ)] = 0 where N is the scaled arclength constraint
    relative to the previous accepted point.

    Returns:
        (converged, state, mu, iterations) or (False, None, None, iterations).
    """
    state = _clamp_state(predicted_state)
    mu = float(predicted_mu)
    scaling = _scaling_matrix(previous_state, previous_mu, param_range_width)
    inverse_scaling = np.diag(1.0 / np.diag(scaling))
    previous_augmented = np.append(previous_state, previous_mu)
    scaled_tangent = inverse_scaling @ tangent
    scaled_tangent_norm = np.linalg.norm(scaled_tangent)
    if scaled_tangent_norm < 1e-18:
        return False, None, None, 0
    scaled_tangent = scaled_tangent / scaled_tangent_norm

    for iteration in range(max_iterations):
        residual_map = _map_residual(
            params_at, state, mu, initial_state, equation_form,
        )
        if not np.all(np.isfinite(residual_map)):
            return False, None, None, iteration + 1

        current_augmented = np.append(state, mu)
        scaled_delta = inverse_scaling @ (current_augmented - previous_augmented)
        arclength_residual = float(np.dot(scaled_tangent, scaled_delta) - step_size)
        residual = np.append(residual_map, arclength_residual)
        if float(np.linalg.norm(residual)) < tol:
            return True, state, mu, iteration + 1

        jacobian_x = _jacobian_state(
            params_at, state, mu, initial_state, equation_form,
        )
        jacobian_mu = _jacobian_parameter(
            params_at, state, mu, initial_state, equation_form, param_range_width,
        )
        if not np.all(np.isfinite(jacobian_x)) or not np.all(np.isfinite(jacobian_mu)):
            return False, None, None, iteration + 1

        identity = np.eye(4, dtype=np.float64)
        jacobian_augmented = np.zeros((5, 5), dtype=np.float64)
        jacobian_augmented[:4, :4] = jacobian_x - identity
        jacobian_augmented[:4, 4] = jacobian_mu
        # ∂N/∂(x,μ) in raw coordinates = scaled_tangent^T @ D^{-1}
        jacobian_augmented[4, :] = scaled_tangent @ inverse_scaling

        try:
            delta = np.linalg.solve(jacobian_augmented, -residual)
        except np.linalg.LinAlgError:
            return False, None, None, iteration + 1
        if not np.all(np.isfinite(delta)):
            return False, None, None, iteration + 1

        state = _clamp_state(state + delta[:4])
        mu = float(mu + delta[4])

    residual_map = _map_residual(params_at, state, mu, initial_state, equation_form)
    if np.all(np.isfinite(residual_map)) and float(np.linalg.norm(residual_map)) < tol:
        return True, state, mu, max_iterations
    return False, None, None, max_iterations


def _seed_fixed_point(params_at, param_range, initial_state, equation_form,
                      warmup_steps, tol):
    """
    Find an initial fixed point by trying mid/min/max of the parameter range.

    Returns:
        (state_vector, mu) or (None, None) if no seed converges.
    """
    range_min, range_max = float(param_range[0]), float(param_range[1])
    candidates = [
        0.5 * (range_min + range_max),
        range_min,
        range_max,
    ]
    for mu in candidates:
        system = _build_system(params_at, mu, initial_state, equation_form)
        result = system.find_fixed_point(warmup_steps=warmup_steps, tol=tol)
        if not result['converged']:
            continue
        fixed_point = result['fixed_point']
        state_vector = np.array(
            [float(fixed_point[key]) for key in _STATE_VARS], dtype=np.float64,
        )
        if np.all(np.isfinite(state_vector)):
            return _clamp_state(state_vector), float(mu)
    return None, None


def _append_point(branch_lists, state, mu, is_stable, spectral_radius):
    """Append one accepted continuation point to the branch lists."""
    branch_lists['param_values'].append(float(mu))
    branch_lists['S'].append(float(state[0]))
    branch_lists['E'].append(float(state[1]))
    branch_lists['F'].append(float(state[2]))
    branch_lists['FP'].append(float(state[3]))
    branch_lists['stable'].append(bool(is_stable))
    branch_lists['spectral_radius'].append(float(spectral_radius))


def _trace_direction(params_at, start_state, start_mu, start_tangent, direction_sign,
                     param_range, initial_state, equation_form, num_points,
                     param_range_width):
    """
    Trace the branch in one signed arclength direction until a range edge or failure.

    Args:
        direction_sign: +1 for forward (increasing arclength along tangent), −1 reverse.

    Returns:
        (points_list_of_dicts, terminated_early)
    """
    range_min, range_max = float(param_range[0]), float(param_range[1])
    points = []
    state = start_state.copy()
    mu = float(start_mu)
    tangent = start_tangent.copy()
    if direction_sign < 0:
        tangent = -tangent

    # Target arclength step so ~num_points covers the parameter interval.
    initial_step = abs(param_range_width) / max(num_points, 1)
    if abs(tangent[4]) > 1e-12:
        initial_step = abs(param_range_width / (tangent[4] * max(num_points, 1)))
    step_size = max(initial_step, 1e-4 * abs(param_range_width))
    min_step = max(1e-8 * abs(param_range_width), 1e-10)
    max_step = max(0.25 * abs(param_range_width), step_size)
    max_steps = 8 * max(num_points, 1)
    terminated_early = False

    for _ in range(max_steps):
        # Stop if already outside / on the far edge in the mu direction of travel.
        traveling_up = tangent[4] > 0
        if traveling_up and mu >= range_max - 1e-12:
            break
        if (not traveling_up) and mu <= range_min - 1e-12:
            break

        accepted = False
        for _retry in range(8):
            predicted_state = state + step_size * tangent[:4]
            predicted_mu = mu + step_size * tangent[4]
            # Soft clip predicted mu into a padded range before correcting.
            predicted_mu = float(np.clip(
                predicted_mu, range_min - 0.05 * abs(param_range_width),
                range_max + 0.05 * abs(param_range_width),
            ))

            converged, corrected_state, corrected_mu, iterations = _newton_corrector(
                params_at, predicted_state, predicted_mu, state, mu, step_size,
                tangent, initial_state, equation_form, param_range_width,
            )
            if converged and corrected_state is not None:
                jacobian_x = _jacobian_state(
                    params_at, corrected_state, corrected_mu,
                    initial_state, equation_form,
                )
                jacobian_mu = _jacobian_parameter(
                    params_at, corrected_state, corrected_mu,
                    initial_state, equation_form, param_range_width,
                )
                new_tangent = _null_space_tangent(
                    jacobian_x, jacobian_mu,
                    _scaling_matrix(corrected_state, corrected_mu, param_range_width),
                    previous_tangent=tangent,
                )
                if new_tangent is None:
                    step_size *= 0.5
                    if step_size < min_step:
                        break
                    continue

                state = corrected_state
                mu = float(corrected_mu)
                tangent = new_tangent
                is_stable, spectral_radius = _classify_stability(jacobian_x)
                points.append({
                    'state': state.copy(),
                    'mu': mu,
                    'stable': is_stable,
                    'spectral_radius': spectral_radius,
                })
                if iterations <= 4:
                    step_size = min(step_size * 1.5, max_step)
                elif iterations >= 12:
                    step_size = max(step_size * 0.7, min_step)
                accepted = True
                break

            step_size *= 0.5
            if step_size < min_step:
                break

        if not accepted:
            terminated_early = True
            break

        # Land exactly on the range boundary when we overshoot.
        if mu < range_min or mu > range_max:
            target_mu = range_min if mu < range_min else range_max
            boundary_state, boundary_mu, ok = _correct_to_fixed_mu(
                params_at, state, target_mu, initial_state, equation_form,
            )
            if ok:
                jacobian_x = _jacobian_state(
                    params_at, boundary_state, boundary_mu,
                    initial_state, equation_form,
                )
                is_stable, spectral_radius = _classify_stability(jacobian_x)
                # Replace the overshooting point with the boundary landing.
                points[-1] = {
                    'state': boundary_state,
                    'mu': boundary_mu,
                    'stable': is_stable,
                    'spectral_radius': spectral_radius,
                }
            else:
                points.pop()
                terminated_early = True
            break

    else:
        terminated_early = True

    return points, terminated_early


def _correct_to_fixed_mu(params_at, state_guess, target_mu, initial_state,
                         equation_form, tol=1e-9, max_iterations=30):
    """
    Newton-correct a fixed point at a pinned parameter value μ = target_mu.

    Returns:
        (state, mu, success)
    """
    state = _clamp_state(state_guess)
    mu = float(target_mu)
    for _ in range(max_iterations):
        residual = _map_residual(params_at, state, mu, initial_state, equation_form)
        if not np.all(np.isfinite(residual)):
            return state, mu, False
        if float(np.linalg.norm(residual)) < tol:
            return state, mu, True
        jacobian_x = _jacobian_state(
            params_at, state, mu, initial_state, equation_form,
        )
        try:
            delta = np.linalg.solve(jacobian_x - np.eye(4), -residual)
        except np.linalg.LinAlgError:
            return state, mu, False
        if not np.all(np.isfinite(delta)):
            return state, mu, False
        state = _clamp_state(state + delta)
    residual = _map_residual(params_at, state, mu, initial_state, equation_form)
    success = np.all(np.isfinite(residual)) and float(np.linalg.norm(residual)) < tol
    return state, mu, success


def continue_fixed_point(
    params_at,
    param_range,
    initial_state_defaults=None,
    num_points=150,
    equation_form="dimensionalized",
    warmup_steps=500,
    tol=1e-10,
):
    """
    Trace a fixed-point branch of G(x; μ) = x by pseudo-arclength continuation.

    Seeds a fixed point near the middle (then ends) of ``param_range``, computes a
    scaled null-space tangent of the augmented Jacobian [∂G/∂x − I | ∂G/∂μ], and
    advances with a predictor–corrector that stays well-posed through folds.

    Args:
        params_at: Callable ``mu -> dict`` returning the full parameter dict at μ.
        param_range: ``(min, max)`` continuation interval for μ.
        initial_state_defaults: Optional initial state for orbit warm-start seeding.
            Defaults to ``DEFAULT_INIT_STATE``.
        num_points: Approximate number of accepted points per direction across the
            interval (used to set the initial arclength step).
        equation_form: Passed to ``DynamicalSystem`` (``"dimensionalized"`` by default).
        warmup_steps: Orbit warm-start length for the initial ``find_fixed_point`` seed.
        tol: Residual tolerance for seeding and corrector convergence.

    Returns:
        Dict with arrays ``param_values``, ``S``, ``E``, ``F``, ``FP``, ``stable``,
        ``spectral_radius`` in trace order (not re-sorted), plus diagnostics
        ``seed_found``, ``forward_terminated_early``, ``backward_terminated_early``.

    Notes:
        Follows a single connected branch from one seed. Coexisting / disconnected
        fixed points are not discovered.
    """
    initial_state = dict(
        DEFAULT_INIT_STATE if initial_state_defaults is None else initial_state_defaults
    )
    range_min, range_max = float(param_range[0]), float(param_range[1])
    if range_max < range_min:
        range_min, range_max = range_max, range_min
    param_range = (range_min, range_max)
    param_range_width = max(range_max - range_min, 1e-12)

    empty = {
        'param_values': np.array([], dtype=np.float64),
        'S': np.array([], dtype=np.float64),
        'E': np.array([], dtype=np.float64),
        'F': np.array([], dtype=np.float64),
        'FP': np.array([], dtype=np.float64),
        'stable': np.array([], dtype=bool),
        'spectral_radius': np.array([], dtype=np.float64),
        'seed_found': False,
        'forward_terminated_early': False,
        'backward_terminated_early': False,
    }

    seed_state, seed_mu = _seed_fixed_point(
        params_at, param_range, initial_state, equation_form, warmup_steps, tol,
    )
    if seed_state is None:
        return empty

    jacobian_x = _jacobian_state(
        params_at, seed_state, seed_mu, initial_state, equation_form,
    )
    jacobian_mu = _jacobian_parameter(
        params_at, seed_state, seed_mu, initial_state, equation_form,
        param_range_width,
    )
    tangent = _null_space_tangent(
        jacobian_x, jacobian_mu,
        _scaling_matrix(seed_state, seed_mu, param_range_width),
        preferred_mu_sign=1.0,
    )
    if tangent is None:
        return empty

    is_stable, spectral_radius = _classify_stability(jacobian_x)

    forward_points, forward_terminated_early = _trace_direction(
        params_at, seed_state, seed_mu, tangent, +1, param_range,
        initial_state, equation_form, num_points, param_range_width,
    )
    backward_points, backward_terminated_early = _trace_direction(
        params_at, seed_state, seed_mu, tangent, -1, param_range,
        initial_state, equation_form, num_points, param_range_width,
    )

    branch = {
        'param_values': [],
        'S': [],
        'E': [],
        'F': [],
        'FP': [],
        'stable': [],
        'spectral_radius': [],
    }
    # Trace order: reverse of backward branch, then seed, then forward.
    for point in reversed(backward_points):
        _append_point(
            branch, point['state'], point['mu'],
            point['stable'], point['spectral_radius'],
        )
    _append_point(branch, seed_state, seed_mu, is_stable, spectral_radius)
    for point in forward_points:
        _append_point(
            branch, point['state'], point['mu'],
            point['stable'], point['spectral_radius'],
        )

    return {
        'param_values': np.asarray(branch['param_values'], dtype=np.float64),
        'S': np.asarray(branch['S'], dtype=np.float64),
        'E': np.asarray(branch['E'], dtype=np.float64),
        'F': np.asarray(branch['F'], dtype=np.float64),
        'FP': np.asarray(branch['FP'], dtype=np.float64),
        'stable': np.asarray(branch['stable'], dtype=bool),
        'spectral_radius': np.asarray(branch['spectral_radius'], dtype=np.float64),
        'seed_found': True,
        'forward_terminated_early': bool(forward_terminated_early),
        'backward_terminated_early': bool(backward_terminated_early),
    }
