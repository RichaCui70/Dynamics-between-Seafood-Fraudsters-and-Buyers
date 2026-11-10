"""Story welfare panels (Fig. F, multiples) with the recommended low-value parameterization.

eps_sw = 0.3 and gamma_P calibrated so the honest equilibrium is unchanged
(gamma_P ~ 1.32e-3, see ``true_low_value_figures.py``). Writes
``figures/true low value black market/tests/
blast_fig_f_story_welfare_panels_multiples_esw_0.3_gammap_1.32e-3.pdf``.

Usage:
    python recommended_welfare_panels.py
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import blast_fishing_figures as figures  # noqa: E402
import true_low_value_figures as runner  # noqa: E402
from blast_fishing_figures import DEFAULT_PARAMS  # noqa: E402

RECOMMENDED_WHOLESALE_ELASTICITY = 0.3
OUTPUT_DIRECTORY = os.path.join(runner.OUTPUT_DIRECTORY, 'tests')
FILE_NAME = 'blast_fig_f_story_welfare_panels_multiples_esw_0.3_gammap_1.32e-3'


def save_pdf(figure, _original_file_stem: str):
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    figure.savefig(os.path.join(OUTPUT_DIRECTORY, f'{FILE_NAME}.pdf'), bbox_inches='tight',
                   pad_inches=0.02)
    plt.close(figure)
    print(f'  wrote figures/true low value black market/tests/{FILE_NAME}.pdf')


def main():
    wholesale_price_scaling = runner.calibrated_wholesale_price_scaling(
        RECOMMENDED_WHOLESALE_ELASTICITY)
    DEFAULT_PARAMS['e_sw'] = RECOMMENDED_WHOLESALE_ELASTICITY
    DEFAULT_PARAMS['gamma_p'] = wholesale_price_scaling
    print(f'eps_sw = {RECOMMENDED_WHOLESALE_ELASTICITY}, gamma_P = {wholesale_price_scaling:.4e}')
    figures.save_figure = save_pdf
    figures.apply_paper_style()
    figures.figure_story_welfare_panels(y_axis_as_multiple=True)


if __name__ == '__main__':
    main()
