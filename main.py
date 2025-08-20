import numpy as np
import matplotlib.pyplot as plt

from olg_solver import solve_ss

def main():
    K = solve_ss()
    print(f"\\nMain: 最終的な資本ストック K = {K:.4f}")


if __name__ == "__main__":
    main()