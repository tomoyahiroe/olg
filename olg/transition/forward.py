import numpy as np
from numba import njit
from ..ss.utils import inverse_interp_aprime_point_numba


@njit
def update_distribution_period_numba(
    mea_current, mea_next, opt_indexes_t, a_grid, aprime_grid, P, NJ, Nl, Na
):
    """
    1期間の分布更新（numba最適化版）
    """
    # 0歳の家計：資産0、スキル50%ずつ
    birth_mass = 1.0 / NJ
    mea_next[0, 0, 0] = 0.5 * birth_mass  # low productivity
    mea_next[0, 1, 0] = 0.5 * birth_mass  # high productivity

    # 1歳以上の家計：前期の最適化結果に基づいて分布更新
    for h in range(NJ - 1):  # h=0,1,...,NJ-2
        for i_l in range(Nl):
            for i_a in range(Na):
                mu = mea_current[h, i_l, i_a]

                if mu > 0:  # 人口が存在する場合のみ処理
                    # 最適な次期資産のインデックスを取得
                    opt_index = opt_indexes_t[h, i_l, i_a]
                    aprime = aprime_grid[opt_index]

                    # 線形補間用のインデックスと重みを取得
                    _, i_opt1, i_opt2, weight_1, weight_2 = (
                        inverse_interp_aprime_point_numba(a_grid, aprime)
                    )

                    # スキル遷移確率を使って分布を更新
                    for j_l in range(Nl):
                        prob = P[i_l, j_l]  # スキル遷移確率

                        # 線形補間で次期分布に割り当て
                        mea_next[h + 1, j_l, i_opt1] += mu * prob * weight_1
                        mea_next[h + 1, j_l, i_opt2] += mu * prob * weight_2


def solve_forward_transition(tr_setting, setting, opt_indexes, mu_ini):
    """
    1→T期（前向き）の分布更新

    numba最適化版を使用して高速化
    """
    # 移行過程の状態分布の箱を用意 (NT, NJ, Nl, Na)
    mu_dist_path = np.zeros((tr_setting.NT, setting.NJ, setting.Nl, setting.Na))

    # t=1期: 初期定常状態の分布を設定
    mu_dist_path[0] = mu_ini.copy()
    mea_current = mu_ini.copy()

    # t=2,3,...,T期まで前向きに分布を更新
    for t in range(1, tr_setting.NT):
        print(f"  分布更新: 期間 {t}/{tr_setting.NT - 1}")

        # 次期の分布を初期化
        mea_next = np.zeros((setting.NJ, setting.Nl, setting.Na))

        # numba最適化された分布更新
        update_distribution_period_numba(
            mea_current,
            mea_next,
            opt_indexes[t - 1],
            setting.a_grid,
            setting.aprime_grid,
            setting.P,
            setting.NJ,
            setting.Nl,
            setting.Na,
        )

        # 計算結果を保存
        mu_dist_path[t] = mea_next.copy()
        mea_current = mea_next.copy()

    return mu_dist_path
