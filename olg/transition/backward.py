import numpy as np
from numba import njit
from ..ss.utils import inverse_interp_aprime_point_numba


@njit
def calculate_factor_prices_numba(K_t, L, alpha, delta):
    """要素価格の計算（numba版）"""
    r_t = alpha * (K_t / L) ** (alpha - 1) - delta
    w_t = (1 - alpha) * (K_t / L) ** alpha
    return r_t, w_t


@njit
def calculate_income_numba(h, i_l, r_t, w_t, tau_t, p_t, l_grid, Njw):
    """所得の計算（numba版）"""
    if h < Njw:  # 労働期間
        return (1 - tau_t) * w_t * l_grid[i_l]
    else:  # 引退期間
        return p_t


@njit
def solve_household_age_numba(
    h,
    NJ,
    Nl,
    Na,
    Naprime,
    setting_params,
    value_fun_next,
    r_t,
    y_h,
    vfun_current,
    afun_current,
    afun_index_current,
):
    """
    特定年齢hの家計最適化問題を解く（numba版）

    setting_params = (a_grid, aprime_grid, P, beta, gamma)
    """
    a_grid, aprime_grid, P, beta, gamma = setting_params

    if h == NJ - 1:  # 最終年齢
        for i_l in range(Nl):
            for i_a in range(Na):
                # 最終期の資産は0
                afun_current[h, i_l, i_a] = 0.0
                afun_index_current[h, i_l, i_a] = 0

                # 予算制約より消費を計算
                c = y_h[i_l] + (1 + r_t) * a_grid[i_a] - 0.0

                # 価値関数を計算
                if gamma == 1:
                    vfun_current[h, i_l, i_a] = np.log(c)
                else:
                    vfun_current[h, i_l, i_a] = c ** (1 - gamma) / (1 - gamma)
    else:
        for i_l in range(Nl):
            for i_a in range(Na):
                a = a_grid[i_a]

                # 価値関数の初期化
                vtemp = np.full(Naprime, -1e10)
                accmax = Naprime

                # 次期資産の選択肢をグリッドサーチ
                for j_a in range(Naprime):
                    aprime = aprime_grid[j_a]

                    # 当期消費の計算
                    c = y_h[i_l] + (1 + r_t) * a - aprime

                    # 消費がマイナスの場合、それ以降のグリッドは除外
                    if c <= 0:
                        accmax = j_a
                        break

                    # 線形補間のためのインデックスと重みを取得
                    _, ac1vec, ac2vec, weight_1, weight_2 = (
                        inverse_interp_aprime_point_numba(a_grid, aprime)
                    )

                    # 来期の期待価値を計算
                    vprime = 0.0
                    for j_l in range(Nl):
                        prob = P[i_l, j_l]
                        V_next = (
                            weight_1 * value_fun_next[h + 1, j_l, ac1vec]
                            + weight_2 * value_fun_next[h + 1, j_l, ac2vec]
                        )
                        vprime += prob * V_next

                    # 価値関数を計算
                    if gamma == 1:
                        vtemp[j_a] = np.log(c) + beta * vprime
                    else:
                        vtemp[j_a] = c ** (1 - gamma) / (1 - gamma) + beta * vprime

                # 最大価値関数を探す
                if accmax > 0:
                    opt_index = np.argmax(vtemp[:accmax])
                    vfun_current[h, i_l, i_a] = vtemp[opt_index]
                    afun_index_current[h, i_l, i_a] = opt_index
                    afun_current[h, i_l, i_a] = aprime_grid[opt_index]


def solve_backward_transition(tr_setting, setting, K_path, opt_indexes, aprimes, V_fin):
    """
    T→1期（後ろ向き）の政策関数計算

    numba最適化版を使用して高速化

    Returns
    -------
    np.ndarray
        価値関数配列 (T, NJ, Nl, Na) - 各期の価値関数
    """
    # 年齢別人口分布（定常状態と同じ）
    L = setting.Njw / setting.NJ  # 簡略化

    # 全期間の経済環境を事前計算
    factor_prices = {}
    for t in range(tr_setting.NT):
        K_t = K_path[t]
        psi_t = tr_setting.rhoT[t]
        tau_t = tr_setting.tauT[t]

        r_t, w_t = calculate_factor_prices_numba(K_t, L, setting.alpha, setting.delta)
        p_t = psi_t * w_t

        factor_prices[t] = (r_t, w_t, p_t, tau_t)

    # setting パラメータをタプルにまとめる（numba用）
    setting_params = (
        setting.a_grid,
        setting.aprime_grid,
        setting.P,
        setting.beta,
        setting.gamma,
    )

    # 価値関数保存用の配列
    V_start = np.zeros(
        (tr_setting.NT, setting.NJ, setting.Nl, setting.Na)
    )  # 各期の価値関数

    # 最初にt+1期の価値関数として最終定常状態の価値関数を設定
    value_fun_box = V_fin.copy()

    # T期から1期まで後ろ向きに計算
    for t in range(tr_setting.NT - 1, -1, -1):
        print(f"  期間 {t + 1}/{tr_setting.NT} を計算中...")

        r_t, w_t, p_t, tau_t = factor_prices[t]

        # 年齢別所得を事前計算
        y_matrix = np.zeros((setting.NJ, setting.Nl))
        for h in range(setting.NJ):
            for i_l in range(setting.Nl):
                y_matrix[h, i_l] = calculate_income_numba(
                    h, i_l, r_t, w_t, tau_t, p_t, setting.l_grid, setting.Njw
                )

        # 当期の価値関数・政策関数の箱
        vfun_current = np.zeros((setting.NJ, setting.Nl, setting.Na))
        afun_current = np.zeros((setting.NJ, setting.Nl, setting.Na))
        afun_index_current = np.zeros(
            (setting.NJ, setting.Nl, setting.Na), dtype=np.int32
        )

        # 各年齢の最適化を実行
        for h in range(setting.NJ - 1, -1, -1):  # 最終年齢から逆順
            solve_household_age_numba(
                h,
                setting.NJ,
                setting.Nl,
                setting.Na,
                setting.Naprime,
                setting_params,
                value_fun_box,
                r_t,
                y_matrix[h],
                vfun_current,
                afun_current,
                afun_index_current,
            )

        # 結果を保存
        aprimes[t] = afun_current.copy()
        opt_indexes[t] = afun_index_current.copy()

        # 価値関数を保存
        V_start[t, :, :, :] = vfun_current.copy()

        # 次のループのために価値関数を更新
        value_fun_box = vfun_current.copy()

    return V_start
