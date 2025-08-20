import numpy as np
from numba import njit


@njit
def maliar_grid(a_min, a_max, N, theta):
    """
    Maliarグリッド関数
    低資産域により多くのグリッドポイントを配置する非線形グリッド
    """
    a_grid = np.empty(N)
    for i in range(1, N+1):
        a_grid[i-1] = a_min + (a_max - a_min) * ((i-1)/(N-1))**theta
    return a_grid


@njit
def inverse_interp_aprime_point_numba(a_grid, aprime_value):
    """
    1つの aprime_value を、グローバルな a_grid 上で線形補間により逆補間する。
    Parameters:
    - a_grid: 資産グリッド（numpy array）
    - aprime_value: スカラー値（次期資産の1点）
    Returns:
    - aprime_interp: 線形補間により a_grid 上で再構成された aprime_value
    - idx: 左のインデックス
    - idx+1: 右のインデックス
    - weight_L: 左の重み
    - weight_R: 右の重み
    """
    if aprime_value <= a_grid[0]:
        return a_grid[0], 0, 1, 1.0, 0.0
    elif aprime_value >= a_grid[-1]:
        last_idx = len(a_grid) - 1
        return a_grid[-1], last_idx-1, last_idx, 0.0, 1.0
    else:
        # バイナリサーチの代わりに線形サーチ（numbaでより効率的）
        idx = 0
        for i in range(len(a_grid) - 1):
            if a_grid[i] <= aprime_value < a_grid[i + 1]:
                idx = i
                break
        
        a_left = a_grid[idx]
        a_right = a_grid[idx + 1]
        weight_R = (aprime_value - a_left) / (a_right - a_left)
        weight_L = 1.0 - weight_R
        aprime_interp = weight_L * a_left + weight_R * a_right
        return aprime_interp, idx, idx + 1, weight_L, weight_R


def inverse_interp_aprime_point(hp, aprime_value):
    """
    1つの aprime_value を、グローバルな a_grid 上で線形補間により逆補間する。
    Parameters:
    - aprime_value: スカラー値（次期資産の1点）
    Returns:
    - aprime_interp: 線形補間により a_grid 上で再構成された aprime_value
    """
    return inverse_interp_aprime_point_numba(hp.a_grid, aprime_value)