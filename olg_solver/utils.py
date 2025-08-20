from typing import Tuple, TYPE_CHECKING
import numpy as np
from numba import njit

if TYPE_CHECKING:
    from .setting import Setting


@njit
def maliar_grid(a_min: float, a_max: float, N: int, theta: float) -> np.ndarray:
    """
    Maliarグリッド関数
    低資産域により多くのグリッドポイントを配置する非線形グリッド
    
    Parameters
    ----------
    a_min : float
        グリッドの最小値
    a_max : float
        グリッドの最大値
    N : int
        グリッドポイント数
    theta : float
        グリッドの曲率パラメータ（1.0で等間隔、>1.0で低資産域により密）
        
    Returns
    -------
    np.ndarray
        Maliarグリッド配列
    """
    a_grid = np.empty(N)
    for i in range(1, N+1):
        a_grid[i-1] = a_min + (a_max - a_min) * ((i-1)/(N-1))**theta
    return a_grid


@njit
def inverse_interp_aprime_point_numba(
    a_grid: np.ndarray, 
    aprime_value: float
) -> Tuple[float, int, int, float, float]:
    """
    1つの aprime_value を、グローバルな a_grid 上で線形補間により逆補間する。
    
    Parameters
    ----------
    a_grid : np.ndarray
        資産グリッド（単調増加）
    aprime_value : float
        補間したい次期資産の値
        
    Returns
    -------
    Tuple[float, int, int, float, float]
        - aprime_interp: 線形補間により a_grid 上で再構成された aprime_value
        - idx_left: 左のインデックス
        - idx_right: 右のインデックス  
        - weight_left: 左の重み
        - weight_right: 右の重み
        
    Notes
    -----
    weight_left + weight_right = 1.0
    aprime_interp = weight_left * a_grid[idx_left] + weight_right * a_grid[idx_right]
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


def inverse_interp_aprime_point(hp: "Setting", aprime_value: float) -> Tuple[float, int, int, float, float]:
    """
    1つの aprime_value を、グローバルな a_grid 上で線形補間により逆補間する。
    
    Parameters
    ----------
    hp : Setting
        OLGモデルの設定オブジェクト
    aprime_value : float
        補間したい次期資産の値
        
    Returns
    -------
    Tuple[float, int, int, float, float]
        inverse_interp_aprime_point_numba の戻り値と同じ
        
    See Also
    --------
    inverse_interp_aprime_point_numba : numba最適化版の実装
    """
    return inverse_interp_aprime_point_numba(hp.a_grid, aprime_value)