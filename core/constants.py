VAR_COLORS = {
    'S': '#4682B4',
    'E': '#2E8B57', 
    'F': '#DC143C',
    'FP': '#DA70D6',
    'Pm': '#FF8C00',
    'Pw': '#8B4513',
    'Rev': '#6A5ACD', 
    'Cost': '#708090',
    'Harvest': '#C8960C',
}

DEFAULT_INIT_STATE = {'S': 0.6, 'E': 0.3, 'F': 0.1, 'FP': 0.1}

DEFAULT_PARAMS = {
    'gamma_m': 10.0,
    'gamma_p': 1.0,
    'gamma_s': 1.0,
    'gamma_e': 0.225,
    'gamma_f': 0.2,
    'gamma_fp': 1.0,


    'e_d': 1.0,
    'e_sw': 0.95,
    'e_sm': 1.0,

    'K': 1.0,
    'F_threshold': 0.5,
    'F_min': 0.0,
    'F_max': 1.0,
    'FP_min': 0.0,
    'FP_max': 1.0,
    'r': 0.225,

    'q0': 0.07,
    'q1': 0.15,
    'pw0': 1.0,
    'pw1': 0.81,
    'c0': 0.9,
    'c1': 0.153
}
