import numpy as np
from numba import njit
from .utils import inverse_interp_aprime_point_numba


@njit
def update_distribution_numba(mu_dist_box, policy_fun_box, h_dist, a_grid, P, NJ, Nl, Na):
    """
    分布を前向きに更新する（numba最適化版）
    """
    # すべての状態分布mu_distを0に初期化
    mu_dist_box.fill(0.0)
    
    a0_index = 0  # 最小資産のインデックス
    mu_0 = 0.5 * h_dist[0]  # 世代全体を1とする場合の人口比

    mu_dist_box[0, 0, a0_index] = mu_0  # スキル low
    mu_dist_box[0, 1, a0_index] = mu_0  # スキル high

    # 繰り返し、1期ずつ前向きに分布を更新していく
    for h in range(NJ-1):  # 年齢: 0歳〜
        for i_l in range(Nl):  # 今期のスキル
            for i_a in range(Na):  # 今期の資産
                mu = mu_dist_box[h, i_l, i_a]
                #if mu == 0:
                #    continue  # 対象の状態（年齢,状態,資産）に人がいない（=0）の場合スキップ

                # 最適な次期資産の値（実数）を取得
                aprime = policy_fun_box[h, i_l, i_a]

                # 補間用のインデックスと重みを取得
                _, i_opt1, i_opt2, weight_1, weight_2 = inverse_interp_aprime_point_numba(a_grid, aprime)

                for j_l in range(Nl):  # 次期スキル
                    pij = P[i_l, j_l]  # スキル遷移確率

                    # 線形補間で次期分布に割り当て
                    mu_dist_box[h + 1, j_l, i_opt1] += mu * pij * weight_1
                    mu_dist_box[h + 1, j_l, i_opt2] += mu * pij * weight_2


def update_distribution(hp, mu_dist_box, policy_fun_box, h_dist):
    """
    分布を前向きに更新する（hpインスタンス使用版）
    """
    update_distribution_numba(mu_dist_box, policy_fun_box, h_dist, hp.a_grid, hp.P, hp.NJ, hp.Nl, hp.Na)