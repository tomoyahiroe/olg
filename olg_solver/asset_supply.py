import numpy as np
from numba import njit


@njit
def calculate_asset_supply_numba(mu_dist_box, policy_fun_box, NJ, Nl, Na):
    """
    家計の資産供給を計算する（numba最適化版）
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


def calculate_asset_supply(hp, mu_dist_box, policy_fun_box):
    """
    家計の資産供給を計算する（hpインスタンス使用版）
    """
    return calculate_asset_supply_numba(mu_dist_box, policy_fun_box, hp.NJ, hp.Nl, hp.Na)