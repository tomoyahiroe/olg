from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt
from numba import njit

if TYPE_CHECKING:
    from .setting import Setting


@njit
def calculate_asset_supply_numba(
    mu_dist_box: npt.NDArray[np.floating],
    policy_fun_box: npt.NDArray[np.floating],
    NJ: int,
    Nl: int,
    Na: int
) -> float:
    """
    家計の資産供給を計算する（numba最適化版）
    
    Parameters
    ----------
    mu_dist_box : npt.NDArray[np.floating]
        人口分布配列 (NJ, Nl, Na)
    policy_fun_box : npt.NDArray[np.floating]
        政策関数配列（実数値） (NJ, Nl, Na)
    NJ : int
        年齢数
    Nl : int
        生産性グリッド数
    Na : int
        今期資産グリッド数
        
    Returns
    -------
    float
        総資産供給
    """
    A_supply = 0.0
    for h in range(NJ):       # 年齢
        for i_l in range(Nl):     # スキル
            for i_a in range(Na): # 今期資産インデックス
                mu = mu_dist_box[h, i_l, i_a]
                if mu == 0:
                    continue
                aprime = policy_fun_box[h, i_l, i_a]  # 次期資産（実数値）
                A_supply += aprime * mu
    return A_supply


def calculate_asset_supply(
    hp: 'Setting',
    mu_dist_box: npt.NDArray[np.floating],
    policy_fun_box: npt.NDArray[np.floating]
) -> float:
    """
    家計の資産供給を計算する（hpインスタンス使用版）
    
    Parameters
    ----------
    hp : Setting
        OLGモデル設定インスタンス
    mu_dist_box : npt.NDArray[np.floating]
        人口分布配列 (NJ, Nl, Na)
    policy_fun_box : npt.NDArray[np.floating]
        政策関数配列（実数値） (NJ, Nl, Na)
        
    Returns
    -------
    float
        総資産供給
    """
    return calculate_asset_supply_numba(mu_dist_box, policy_fun_box, hp.NJ, hp.Nl, hp.Na)