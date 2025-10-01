# Asymptotic Analysis of Partition Convolutions from Ramanujan's Congruences

This repository contains the computational verification and complete code for the paper "Asymptotic Analysis of Partition Convolutions Arising from Ramanujan's Congruences."

**Pre-print (Zenodo):** [https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)
* **DOI** - 10.5281/zenodo.XXXX
* **URL** - https://doi.org/10.5281/zenodo.XXXX

## Abstract

We establish precise asymptotic growth rates for integer-valued partition convolutions
c_k^(m) = Σ [p(mj+a_m)/m] · p(k-j) constructed using Ramanujan's classical congruences for m ∈ {5,7,11}. Using discrete Laplace approximation, we prove ln(c_k^(m)) = π√(2/3)√(m+1)√k - (5/4)ln(k) + A_m + O(k^(-1/2)) with explicit constants. Computational verification extends to k=9509, confirming superpolynomial vanishing relative to binomial coefficients. Statistical analysis of residues modulo primes shows uniform distribution, indicating no hidden arithmetic structure beyond Ramanujan's original congruences.

## Key Results

- **Growth constant:** B_5* = 2π ≈ 6.283, B_7* = 4π/√3 ≈ 7.255, B_11* = 2π√2 ≈ 8.886
- **Logarithmic correction:** -5/4 ln(k) term rigorously derived and computationally verified
- **Vanishing thresholds:** Ratio c_(p-2)/C(2p-1,p-1) < 10^(-10) first at primes p = 41, 47, 59 respectively
- **Statistical uniformity:** Residues pass χ² and KS tests with p-values > 0.4


## Repository Contents

-   `partition_convolution_analysis.py`: The core Python script to run the computation and analysis.
-   `analysis_results.py`: Post-processing and plotting after running the python script above.
-   `README.md`: This documentation file.
-   `requirements.txt`: A list of the Python dependencies required to run the script.
-   `LICENSE`: The MIT License file for the project.


## Requirements

-   Python 3.8+
-   Libraries listed in `requirements.txt`: `pandas`, `sympy`, `numpy`, `matplotlib`, `scipy`

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/partition-binomial-divisibility-analyzer.git](https://github.com/your-username/partition-binomial-divisibility-analyzer.git)
    cd partition-binomial-divisibility-analyzer
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main analysis:
```bash
python code/partition_convolution_analysis.py
```

This will:

- Compute partition values up to n ≈ 220,000 using Euler's pentagonal recurrence
- Calculate convolutions for all three Ramanujan variants (m=5,7,11)
- Analyze vanishing behavior at prime indices up to 20,000
- Generate statistical tests for residue distributions
- Save results to CSV and PKL files

Warning: Full computation takes several hours. Results are pre-computed in the data/ directory.
Generate Plots

To reproduce figures from the paper:
```bash
python code/analysis_results.py
```

This creates:

- Vanishing ratio plots
- Residue distribution histograms
- Growth rate verification plots
- Statistical summary tables
  


### Key Features

* Exact computation: Uses iterative Euler pentagonal recurrence (no recursion depth errors)
* High precision: Decimal arithmetic for exact ratios even with huge numbers
* Three variants: Handles m ∈ {5,7,11} using Ramanujan congruences p(5n+4)≡0 (mod 5), etc.
* Robust handling: Manages float underflow for ratios < 10^(-5000)
* Statistical analysis: χ² and Kolmogorov-Smirnov tests for uniformity.


### Computational Details
The code computes:

Partition function p(n) for n up to 220,006
Convolutions c_k^(m) for k up to 9509
Exact ratios c_(p-2)/C(2p-1,p-1) for primes up to 20,000
Residue analysis D_WP(p) = C(2p-1,p-1) - c_(p-2) - 1 (mod p)
Growth coefficient regression with R² > 0.9999

At p=9511, the largest prime computed:

Variant 5: c_9509 has 259 digits, ratio ≈ 10^(-5465)
Variant 7: c_9509 has 300 digits, ratio ≈ 10^(-5424)
Variant 11: c_9509 has 369 digits, ratio ≈ 10^(-5355)


---














## Citation

If you use this work, please cite the paper using the Zenodo archive.

@misc{naladiga2025partition,
  author = {Naladiga Venkat, Arvind},
  title = {Asymptotic Analysis of Partition Convolutions Arising from Ramanujan's Congruences},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://doi.org/10.5281/zenodo.XXXXXXX}
}

---

## License

The content of this repository is dual-licensed:

- **MIT License** for `analyze_partitions.py` See the [LICENSE](LICENSE) file for details.
- **CC BY 4.0** (Creative Commons Attribution 4.0 International) for all other content (results.txt, README, etc.)



## Author

- **Arvind N. Venkat** - [arvind.venkat01@gmail.com](mailto:arvind.venkat01@gmail.com)
