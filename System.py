import numpy as np
import warnings
from scipy.optimize import least_squares

CLOSE_TO_ZERO = np.finfo(np.float128).eps
CLOSE_TO_ONE = 1 - np.finfo(np.float128).epsneg
POSITIVE_INF = np.inf
STATE_KEYS = ('S', 'E', 'F', 'FP')
DEFAULT_PARAMS = {
    'gamma_m': 10.0,
    'gamma_p': 1.0,
    'gamma_s': 1.0,
    'gamma_e': 0.225,
    'gamma_f': 1.0,
    'gamma_fp': 10.0,
    
    'e_d': 1.0,
    'e_sw': 0.95,
    'e_sm': 1.0,
    
    'K': 1.0,
    'F_threshold': 0.5,
    'r': 0.225,
    
    'q0': 0.07,
    'q1': 0.15,
    'pw0': 1.0,
    'pw1': 0.81,
    'c0': 0.9,
    'c1': 0.153
}
class DynamicalSystem():
    # CONSTRUCTOR
    def __init__(self, params, state, type="nondimensionalized"):
        '''
        Args:
            params (dict): A dictionary of system parameters containing:
                Expected keys:
                * gamma_m (float128): [Add description for gamma_m].
                * gamma_p (float128): [Add description for gamma_p].
                * gamma_f (float128): [Add description for gamma_f].
                * gamma_s (float128): [Add description for gamma_s].
                * gamma_e (float128): [Add description for gamma_e].
                * gamma_fp (float128): [Add description for gamma_fp].
                * e_d (float128): [Add description for e_d].
                * e_sw (float128): [Add description for e_sw].
                * e_sm (float128): [Add description for e_sm].
                * K (float128): Carrying capacity of the seafood population.
                * F_threshold (float128): Threshold limit for fraud.
                * q0 (float128): Catchability coefficient when no fraudsters are present.
                * q1 (float128): Catchability coefficient when fraudsters are present.
                * r (float128): Intrinsic growth rate.
                * pw0 (float128): [Add description for pw0].
                * c0 (float128): [Add description for c0].
                * pw1 (float128): [Add description for pw1].
                * c1 (float128): [Add description for c1].
            state: A dictionary of the initial system state containing:
                Expected keys:
                * S (float128): Seafood biomass.
                * E (float128): Fishing effort.
                * F (float128): Current level of fraud.
                * FP (float128): Public perception of fraud.
            type: A string indicating the type of system to be initialized.
                Expected values:
                * "nondimensionalized": Initializes the system in nondimensionalized form.
                * "dimensionalized": Initializes the system in dimensionalized form.
        '''
        self._params = {}
        self._state = {}
        self._type = type
        
        self.params = params if params is not None else DEFAULT_PARAMS
        self.state = state
    
    # STATE VARIABLES (nondimensionalized)
    def seafood_state_nondim(self):
        S = self.state['S']
        E = self.state['E']
        catchability = self.catchability_nondim()
        gamma_s = self.nondim_params['gamma_s']
        
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        S_next = np.clip(
            [S * np.exp(gamma_s * (1 - S - E * catchability))],
            np.finfo(np.float128).eps,
            POSITIVE_INF
        )[0]

        return S_next
    def effort_state_nondim(self): 
        S = self.state['S']
        E = self.state['E']
        F = self.state['F']
        e_sw = self.nondim_params['e_sw']
        gamma_p = self.nondim_params['gamma_p']
        gamma_e = self.nondim_params['gamma_e']
        mu = self.nondim_params['mu']
        wholesale_price = self.wholesale_price_nondim()
        cost = self.cost_nondim()
        catchability = self.catchability_nondim()

        denom_e = (gamma_p * E * S * catchability) ** e_sw
        term = (mu * S * catchability * wholesale_price) / denom_e
                
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        E_next = np.clip(
            [E * np.exp(gamma_e * (term - cost))],
            CLOSE_TO_ZERO,
            POSITIVE_INF
        )[0]
        
        return E_next
    def fraudster_state_nondim(self):
        S = self.state['S']
        E = self.state['E']
        F = self.state['F']
        FP = self.state['FP']
        e_sw = self.nondim_params['e_sw']
        e_sm = self.nondim_params['e_sm']
        e_d = self.nondim_params['e_d']
        gamma_f = self.nondim_params['gamma_f']
        gamma_m = self.nondim_params['gamma_m']
        gamma_p = self.nondim_params['gamma_p']
        wholesale_price = self.wholesale_price_nondim()
        catchability = self.catchability_nondim()
        
        '''
            1.0 and 0.0 are known fixed points.
            Better to simply return what we know instead of
            calculating the result and risking numerical imprecisions. 
        '''
        if F == 1.0:
            return 1.0
        if F == 0.0:
            return 0.0
        
        with warnings.catch_warnings(record=True) as recorded_warnings:
            denom_market = (E * S * catchability)**(e_sm/2)
            price_market = gamma_m * ((1 - FP)**(e_d/2) / denom_market)
            
            denom_wholesale = (gamma_p * E * S * catchability)**(e_sw)
            price_wholesale = wholesale_price / denom_wholesale
            
            delta = gamma_f * (price_market - price_wholesale)
            
            '''
            Artifically clip between 0 and 1 (noninclusive).
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
            '''
            F_next = np.clip(
                [(F * np.exp(delta)) / (1 + F * (np.exp(delta) - 1))],
                CLOSE_TO_ZERO,
                CLOSE_TO_ONE
            )[0]
        if recorded_warnings:
            print(f"Captured {len(recorded_warnings)} warning(s):")
            print(f"S: {S}")
            print(f"E: {E}")
            print(f"F: {F}")
            print(f"Fp: {FP}")
            print(f"market price: {price_market}")
            print(f"wholesale price: {price_wholesale}")
            print(self.params)
            for w in recorded_warnings:
                print(f"- Message: {w.message}, Category: {w.category.__name__}")
        return F_next
    def p_fraudster_state_nondim(self): 
        F = self.state['F']
        FP = self.state['FP']
        F_threshold = self.nondim_params['F_threshold']
        gamma_fp = self.nondim_params['gamma_fp']
        
        exp_delta_fp = np.exp(gamma_fp * (F - F_threshold))

        '''
            Artifically clip between 0 and 1 (noninclusive).
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        FP_next = np.clip(
            [(FP * exp_delta_fp) / (1 + FP * (exp_delta_fp - 1))],
            CLOSE_TO_ZERO,
            CLOSE_TO_ONE
        )[0]
        
        return FP_next
    
    # VARIABLES (nondimensionalized)
    def wholesale_price_nondim(self):
        F = self.state['F']
        pw = self.nondim_params['pw']
        return F * (pw - 1) + 1
    def cost_nondim(self):
        F = self.state['F']
        c = self.nondim_params['c']
        return F * (c - 1) + 1
    def catchability_nondim(self):
        F = self.state['F']
        q = self.nondim_params['q']
        return F * (q - 1) + 1
    
    # STATE VARIABLES (dimensionful)
    def seafood_state_dimful(self):
        S = self.state['S']
        E = self.state['E']
        r = self.params['r']
        K = self.params['K']
        gamma_s = self.params['gamma_s']
        
        q = self.catchability()
        
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        S_next = np.clip(
            [S * np.exp(gamma_s * r * (1 - S / K) - q * E)],
            np.finfo(np.float128).eps,
            POSITIVE_INF
        )[0]

        return S_next
    def effort_state_dimful(self):
        S = self.state['S']
        E = self.state['E']
        gamma_e = self.params['gamma_e']
                
        revenue = self.revenue_per_unit_effort()
        cost = self.cost_per_unit_effort()
                
        E_next = np.clip(
            [E * np.exp(gamma_e * (revenue - cost))],
            CLOSE_TO_ZERO,
            POSITIVE_INF
        )[0]
        
        return E_next
    def fraudster_state_dimful(self):
        F = self.state['F']
        
        if F == 1.0:
            return 1.0
        if F == 0.0:
            return 0.0
        
        gamma_f = self.params['gamma_f']
        
        pm = self.market_price()
        pw = self.wholesale_price()
        delta = gamma_f * (pm - pw)
        
        return np.clip([F * np.exp(delta) / (1 + F * (np.exp(delta) - 1))], CLOSE_TO_ZERO, CLOSE_TO_ONE)[0]
    def p_fraudster_state_dimful(self):
        F = self.state['F']
        FP = self.state['FP']
        
        if FP == 1.0:
            return 1.0
        if FP == 0.0:
            return 0.0
        
        F_threshold = self.params['F_threshold']
        gamma_fp = self.params['gamma_fp']
        exp_delta_fp = np.exp(gamma_fp * (F - F_threshold))
        
        return np.clip([FP * exp_delta_fp / (1 + FP * (exp_delta_fp - 1))], CLOSE_TO_ZERO, CLOSE_TO_ONE)[0]
    
    # VARIABLES (dimensionful)
    def catchability(self):
        q0 = self.params['q0']
        q1 = self.params['q1']
        F = self.state['F']
        return q0 + (q1 - q0) * F
    def harvest(self):
        S = self.state['S']
        E = self.state['E']
        q = self.catchability()
        
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        return np.clip([q * E * S], np.finfo(type(S)).eps, np.inf)[0]
    def demand(self, **kwargs):
        FP = self.state['FP'] if 'FP' not in kwargs else kwargs['FP']
        e_sm = self.params['e_sm'] if 'e_sm' not in kwargs else kwargs['e_sm']
        e_d = self.params['e_d'] if 'e_d' not in kwargs else kwargs['e_d']
        
        H = self.harvest() if 'H' not in kwargs else kwargs['H']
        
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        return np.clip([np.sqrt((FP)**e_d * H**e_sm)], np.finfo(type(FP)).eps, np.inf)[0]
    def market_price(self):
        FP = self.state['FP']
        gamma_m = self.params['gamma_m']
        e_d = self.params['e_d']
        e_sm = self.params['e_sm']
        H = self.harvest()
        
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        return np.clip([np.sqrt((1-FP)**e_d / H**e_sm) * gamma_m], CLOSE_TO_ZERO, POSITIVE_INF)[0]
    def wholesale_price(self):
        F = self.state['F']
        pw0 = self.params['pw0']
        pw1 = self.params['pw1']
        e_sw = self.params['e_sw']
        gamma_p = self.params['gamma_p']
        H = self.harvest()
        
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        return np.clip([(F*(pw1 - pw0) + pw0) / ((gamma_p * H)**e_sw)], np.finfo(type(F)).eps, np.inf)[0]
    def revenue_per_unit_effort(self):
        q = self.catchability()
        Pw = self.wholesale_price()
        S = self.state['S']
        
        return q * Pw * S
    def cost_per_unit_effort(self):
        F = self.state['F']
        c0 = self.params['c0']
        c1 = self.params['c1']   
        return F * (c1 - c0) + c0
      
    def system_map(self) -> dict:        
        '''
        Get system's values for the next time step.
        Only computes the state update matching self.type to avoid
        calling nondim/dimful functions with incompatible params.
        '''
        market_price = self.market_price()
        wholesale_price = self.wholesale_price()
        catchability = self.catchability()
        revenue = self.revenue_per_unit_effort()
        cost = self.cost_per_unit_effort()
        harvest = self.harvest()
        demand = self.demand()
        
        if self.type == "dimensionalized":
            S_next = self.seafood_state_dimful()
            E_next = self.effort_state_dimful()
            F_next = self.fraudster_state_dimful()
            FP_next = self.p_fraudster_state_dimful()
        else:
            S_next = self.seafood_state_nondim()
            E_next = self.effort_state_nondim()
            F_next = self.fraudster_state_nondim()
            FP_next = self.p_fraudster_state_nondim()
        
        return {
            'S': S_next,
            'E': E_next,
            'F': F_next,
            'FP': FP_next,
            'market_price': market_price,
            'wholesale_price': wholesale_price,
            'catchability': catchability,
            'revenue_per_unit_effort': revenue,
            'cost_per_unit_effort': cost,
            'harvest': harvest,
            'demand': demand,
        }
    
    def time_series_plot(self, time, title="", x_label="", y_label="", ax=None) -> dict:
        seafood = np.array(self.state['S'], dtype=np.float128)
        effort = np.array(self.state['E'], dtype=np.float128)
        fraudsters = np.array(self.state['F'], dtype=np.float128)
        p_fraudsters = np.array(self.state['FP'], dtype=np.float128)
        harvest_arr = np.array(self.harvest(), dtype=np.float128)
        market_price_arr = np.array(self.market_price(), dtype=np.float128)
        wholesale_price_arr = np.array(self.wholesale_price(), dtype=np.float128)
        revenue_arr = np.array(self.revenue_per_unit_effort(), dtype=np.float128)
        cost_arr = np.array(self.cost_per_unit_effort(), dtype=np.float128)
        
        for _ in range(time):
            result = self.system_map()
            self.state = {
                'S': result['S'], 'E': result['E'],
                'F': result['F'], 'FP': result['FP'],
            }
            
            seafood = np.append(seafood, result['S'])
            effort = np.append(effort, result['E'])
            fraudsters = np.append(fraudsters, result['F'])
            p_fraudsters = np.append(p_fraudsters, result['FP'])
            market_price_arr = np.append(market_price_arr, result['market_price'])
            wholesale_price_arr = np.append(wholesale_price_arr, result['wholesale_price'])
            harvest_arr = np.append(harvest_arr, result['harvest'])
            revenue_arr = np.append(revenue_arr, result['revenue_per_unit_effort'])
            cost_arr = np.append(cost_arr, result['cost_per_unit_effort'])
        
        return {
            'Seafood': seafood,
            'Effort': effort,
            'Fraudsters': fraudsters,
            'Perception of Fraud': p_fraudsters,
            'Market Price': market_price_arr,
            'Wholesale Price': wholesale_price_arr,
            'Harvest': harvest_arr,
            'Revenue per Effort': revenue_arr,
            'Cost per Effort': cost_arr,
        }

    def _evaluate_map_vec(self, state_vec):
        '''
        Evaluate the 4D map G(x) at an arbitrary state vector without
        permanently mutating self.state.

        Clamps inputs to the physically valid domain before evaluation
        so that finite-difference perturbations in the Jacobian cannot
        push variables into regions that produce NaN/Inf (e.g. F < 0
        causing negative wholesale prices, or H → 0 blowing up market
        price).

        Args:
            state_vec: length-4 array-like [S, E, F, FP]
        Returns:
            np.ndarray of shape (4,) with [S', E', F', FP']
        '''
        
        # Clamping to 
        clamped = np.array([
            max(state_vec[0], CLOSE_TO_ZERO),              # S > 0
            max(state_vec[1], CLOSE_TO_ZERO),              # E > 0
            min(max(state_vec[2], CLOSE_TO_ZERO), CLOSE_TO_ONE),  # 0 < F < 1
            min(max(state_vec[3], CLOSE_TO_ZERO), CLOSE_TO_ONE),  # 0 < FP < 1
        ])

        saved = self.state.copy()
        self.state = {
            k: np.float128(v)
            for k, v in zip(STATE_KEYS, clamped)
        }
        result = self.system_map()
        self.state = saved
        return np.array([
            float(result['S']), float(result['E']),
            float(result['F']), float(result['FP']),
        ])

    def find_fixed_point(self, initial_guess=None, warmup_steps=500, tol=1e-10):
        '''
        Find a fixed point x* of the map G(x*) = x* using
        scipy.optimize.least_squares with Trust Region Reflective (trf),
        which supports box constraints to keep F and FP in (0, 1).

        Strategy: simulate forward `warmup_steps` iterations from the current
        state to get close to the attractor, then refine with least_squares.
        Uses both the last warmup state and the orbit mean as candidates,
        keeping whichever yields the smallest residual at an interior point.

        Args:
            initial_guess: dict with keys 'S','E','F','FP', or None to
                           use the warm-start strategy.
            warmup_steps:  number of forward iterations for warm start.
            tol:           convergence tolerance on the residual norm.
        Returns:
            dict with keys:
                'fixed_point' : dict {'S','E','F','FP'}
                'residual_norm': float — ||G(x*) - x*||
                'converged'   : bool
                'info'        : least_squares result object
        '''
        def residual(x):
            return self._evaluate_map_vec(x) - x

        lower = np.array([CLOSE_TO_ZERO, CLOSE_TO_ZERO, CLOSE_TO_ZERO, CLOSE_TO_ZERO])
        upper = np.array([np.inf,         np.inf,         CLOSE_TO_ONE,  CLOSE_TO_ONE])

        candidates = []

        if initial_guess is not None:
            candidates.append(np.array([float(initial_guess[k]) for k in STATE_KEYS]))
        else:
            saved = self.state.copy()
            tail_len = max(warmup_steps // 2, 50)
            orbit = []
            for _wi in range(warmup_steps):
                result = self.system_map()
                self.state = {
                    'S': result['S'], 'E': result['E'],
                    'F': result['F'], 'FP': result['FP'],
                }
                if _wi >= warmup_steps - tail_len:
                    orbit.append([float(self.state[k]) for k in STATE_KEYS])
            x_last = np.array([float(self.state[k]) for k in STATE_KEYS])
            self.state = saved

            orbit_arr = np.array(orbit)
            x_mean = orbit_arr.mean(axis=0)

            x_fallback = np.array([float(saved[k]) for k in STATE_KEYS])
            for arr in (x_mean, x_last):
                arr[~np.isfinite(arr)] = x_fallback[~np.isfinite(arr)]

            candidates.append(x_mean)
            candidates.append(x_last)

        best_result = None
        best_norm = np.inf

        for _, x0 in enumerate(candidates):
            x0 = np.clip(x0, lower, upper)

            if not np.all(np.isfinite(residual(x0))):
                continue

            try:
                ls_result = least_squares(
                    residual, x0, bounds=(lower, upper),
                    method='trf', max_nfev=5000,
                )
            except ValueError:
                continue

            x_star = ls_result.x
            res_norm = float(np.linalg.norm(residual(x_star)))

            is_boundary = (
                x_star[2] < 1e-6 or x_star[2] > 1 - 1e-6 or
                x_star[3] < 1e-6 or x_star[3] > 1 - 1e-6 or
                x_star[0] < 1e-6
            )

            if is_boundary and best_result is not None:
                continue
            if (not is_boundary and best_result is not None
                    and best_norm < np.inf and res_norm > best_norm):
                continue
            if res_norm < best_norm or (is_boundary == False):
                best_result = ls_result
                best_norm = res_norm

        if best_result is None:
            return {
                'fixed_point': {k: float('nan') for k in STATE_KEYS},
                'residual_norm': np.inf,
                'converged': False,
                'info': None,
            }

        x_star = best_result.x
        res_norm = float(np.linalg.norm(residual(x_star)))
        fp_dict = {k: v for k, v in zip(STATE_KEYS, x_star)}

        return {
            'fixed_point': fp_dict,
            'residual_norm': res_norm,
            'converged': res_norm < tol,
            'info': best_result,
        }

    def jacobian(self, state=None, h=None):
        '''
        Compute the 4×4 Jacobian of the map G at a given state using
        central finite differences:

            J_ji = (G_j(x + h*e_i) - G_j(x - h*e_i)) / (2h)

        Perturbation size defaults to eps^(1/3) * max(1, |x_i|) where
        eps ≈ 2.2e-16 (float64 machine epsilon), giving O(h²) accuracy.

        Args:
            state: dict {'S','E','F','FP'} or None (uses current self.state)
            h:     scalar perturbation override, or None for adaptive step
        Returns:
            np.ndarray of shape (4, 4)
        '''
        if state is None:
            state = self.state
        x0 = np.array([float(state[k]) for k in STATE_KEYS])
        eps_machine = np.finfo(np.float64).eps
        n = len(x0)
        J = np.zeros((n, n))

        for i in range(n):
            hi = h if h is not None else (eps_machine ** (1.0 / 3.0)) * max(1.0, abs(x0[i]))
            x_fwd = x0.copy()
            x_bwd = x0.copy()
            x_fwd[i] += hi
            x_bwd[i] -= hi
            J[:, i] = (self._evaluate_map_vec(x_fwd) - self._evaluate_map_vec(x_bwd)) / (2.0 * hi)

        return J

    def stability_analysis(self, initial_guess=None, warmup_steps=500, tol=1e-10):
        '''
        Full stability analysis: find the fixed point, compute the Jacobian,
        extract eigenvalues (via numpy.linalg.eig — LAPACK QR iteration),
        and classify stability.

        For a discrete map, the fixed point is stable iff the spectral
        radius rho = max|lambda_i| < 1.

        Args:
            initial_guess: passed to find_fixed_point()
            warmup_steps:  passed to find_fixed_point()
            tol:           passed to find_fixed_point()
        Returns:
            dict with keys:
                'fixed_point'    : dict {'S','E','F','FP'}
                'converged'      : bool — whether the fixed-point solver converged
                'residual_norm'  : float
                'jacobian'       : np.ndarray (4,4)
                'eigenvalues'    : np.ndarray of complex eigenvalues
                'spectral_radius': float — max |lambda_i|
                'stable'         : bool — True iff spectral_radius < 1
                'classification' : str
        '''
        fp_result = self.find_fixed_point(
            initial_guess=initial_guess,
            warmup_steps=warmup_steps,
            tol=tol,
        )
        fp = fp_result['fixed_point']

        J = self.jacobian(state=fp)

        if not np.all(np.isfinite(J)):
            return {
                'fixed_point': fp,
                'converged': fp_result['converged'],
                'residual_norm': fp_result['residual_norm'],
                'jacobian': J,
                'eigenvalues': np.array([np.inf] * 4),
                'spectral_radius': np.inf,
                'stable': False,
                'classification': 'degenerate (Jacobian contains NaN/Inf)',
            }

        eigenvalues = np.linalg.eig(J)[0]
        moduli = np.abs(eigenvalues)
        rho = float(np.max(moduli))

        margin = 1e-6
        has_complex = any(abs(ev.imag) > 1e-10 for ev in eigenvalues)

        if not fp_result['converged']:
            classification = "no fixed point found (solver did not converge)"
            stable = False
        elif rho < 1.0 - margin:
            classification = "stable spiral" if has_complex else "stable node"
            stable = True
        elif rho > 1.0 + margin:
            classification = "unstable spiral" if has_complex else "unstable node"
            stable = False
        else:
            classification = "Neimark-Sacker boundary (marginal)"
            stable = rho < 1.0

        return {
            'fixed_point': fp,
            'converged': fp_result['converged'],
            'residual_norm': fp_result['residual_norm'],
            'jacobian': J,
            'eigenvalues': eigenvalues,
            'spectral_radius': rho,
            'stable': stable,
            'classification': classification,
        }

    # FUNCTION PROPERTIES
    @property
    def state(self):
        return self._state
    @state.setter
    def state(self, value):
        self._state = value
        
    @property
    def params(self):
        return self._params
    @params.setter
    def params(self, value):
        self._params = value
        
    @property
    def type(self):
        return self._type
        
    @property
    def nondim_params(self):
        params = self.params.copy()
        return {
            'gamma_m': params['gamma_m'] / (params['pw0'] * (params['r'] * params['K']) ** (params['e_sm'] / 2.0)),
            'gamma_p': params['gamma_p'] * params['r'] * params['K'],
            'gamma_f': params['gamma_f'] * params['pw0'],
            'gamma_s': params['gamma_s'] * params['r'],
            'gamma_e': params['gamma_e'] * params['c0'],
            'gamma_fp': params['gamma_fp'],
            'e_sm': params['e_sm'], 'e_sw': params['e_sw'], 'e_d': params['e_d'],
            'F_threshold': params['F_threshold'],
            'mu': (params['q0'] * params['pw0'] * params['K']) / params['c0'],
            'q': params['q1'] / params['q0'],
            'pw': params['pw1'] / params['pw0'],
            'c': params['c1'] / params['c0'],
        }
        