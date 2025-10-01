"""
Generate residual verification plot for -5/4 ln(k) coefficient

Author: Arvind Naladiga Venkat
License: MIT License

Plots ln(c_k) - B_m*√k versus ln(k) to verify the logarithmic correction term.
Requires: growth_ck_20000.csv from partition_convolution_main.py

Usage:
    python plot_residuals.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
growth_data = pd.read_csv('../data/growth_ck_20000.csv')
variant_5 = growth_data[growth_data['variant'] == 5]

k_vals = variant_5['k'].values
log_ck = variant_5['log_c_k'].values

# Theoretical B_m* = 2*pi for m=5
B_5_star = 2 * np.pi

# Compute residuals
residuals = log_ck - B_5_star * np.sqrt(k_vals)

# Fit to verify -5/4 slope
ln_k = np.log(k_vals)
coeffs = np.polyfit(ln_k, residuals, 1)
fitted_line = coeffs[0] * ln_k + coeffs[1]

print(f"Fitted slope: {coeffs[0]:.4f}")
print(f"Theoretical: -1.25")
print(f"Relative error: {100*abs(coeffs[0] + 1.25)/1.25:.1f}%")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(ln_k, residuals, 'o', alpha=0.5, label='Data', markersize=3)
plt.plot(ln_k, fitted_line, 'r-', linewidth=2, 
         label=f'Fitted slope = {coeffs[0]:.3f}')
plt.axhline(y=-1.25*ln_k.mean(), color='g', linestyle='--', 
            linewidth=1.5, label='Expected: -5/4 ln(k)')
plt.xlabel('ln(k)', fontsize=12)
plt.ylabel('ln(c_k) - B*√k', fontsize=12)
plt.title('Verification of -5/4 ln(k) coefficient (Variant 5)', fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../paper/figures/residuals_verification.pdf')
print("Saved to ../paper/figures/residuals_verification.pdf")
plt.show()
