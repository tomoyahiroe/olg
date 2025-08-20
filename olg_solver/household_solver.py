import numpy as np
from numba import njit
from .utils import inverse_interp_aprime_point_numba


@njit
def solve_household_backward_numba(value_fun_box, policy_fun_box, optaprime_index_box,
                                   y_list, a_grid, aprime_grid, P, 
                                   r, beta, gamma, NJ, Nl, Na, Naprime, Njw):
    """
    家計の最適化問題を後ろ向きに解く（numba最適化版）
    """
    # 最終期（年齢NJ-1）の政策関数、価値関数を求める
    for i_l in range(Nl):
        for i_a in range(Na):
            aprime = 0.0
            c = y_list[NJ-1, i_l] + (1 + r) * a_grid[i_a] - aprime
            if gamma == 1:
                value_fun = np.log(c)
            else:
                value_fun = c ** (1 - gamma) / (1 - gamma)
            value_fun_box[NJ-1, i_l, i_a] = value_fun
            optaprime_index_box[NJ-1, i_l, i_a] = 0
            policy_fun_box[NJ-1, i_l, i_a] = aprime

    # 後ろ向きに価値関数・政策関数を計算
    for h in range(NJ-2, -1, -1):  # NJ-2歳から0歳まで繰り返し
        for i_l in range(Nl):  # 今期の生産性i_lについて繰り返し
            y = y_list[h, i_l]
            
            for i_a in range(Na):  # 今期の資産i_aについて繰り返し
                a = a_grid[i_a]
                vtemp = -1e10 * np.ones(Naprime)  # j_a に対する一時的な価値関数
                accmax = Naprime  # サーチされるグリッドの最大値

                for j_a in range(Naprime):  # 次期の資産j_aについて繰り返し
                    aprime = aprime_grid[j_a]
                    
                    # 補間用のインデックスと重みを取得（numba版）
                    _, ac1vec, ac2vec, weight_L, weight_R = inverse_interp_aprime_point_numba(a_grid, aprime)
                    
                    c = y + (1 + r) * a - aprime

                    if c <= 0:
                        accmax = j_a - 1  # 消費が負になったので、それ以降のグリッドは打ち切り
                        break

                    vprime = 0.0
                    for j_l in range(Nl):  # 次期の生産性l_jについて繰り返し
                        prob_l = P[i_l, j_l]
                        V_next = weight_L * value_fun_box[h + 1, j_l, ac1vec] + weight_R * value_fun_box[h + 1, j_l, ac2vec]
                        vprime += prob_l * V_next

                    if gamma == 1:
                        vtemp[j_a] = np.log(c) + beta * vprime
                    else:
                        vtemp[j_a] = c ** (1 - gamma) / (1 - gamma) + beta * vprime

                if accmax >= 0:
                    opt_index = 0
                    max_val = vtemp[0]
                    for idx in range(1, min(accmax + 1, Naprime)):
                        if vtemp[idx] > max_val:
                            max_val = vtemp[idx]
                            opt_index = idx

                    value_fun_box[h, i_l, i_a] = vtemp[opt_index]
                    optaprime_index_box[h, i_l, i_a] = opt_index
                    policy_fun_box[h, i_l, i_a] = aprime_grid[opt_index]


def solve_household_backward(hp, value_fun_box, policy_fun_box, optaprime_index_box, y_list, r):
    """
    家計の最適化問題を後ろ向きに解く（hpインスタンス使用版）
    """
    solve_household_backward_numba(
        value_fun_box, policy_fun_box, optaprime_index_box,
        y_list, hp.a_grid, hp.aprime_grid, hp.P, 
        r, hp.beta, hp.gamma, hp.NJ, hp.Nl, hp.Na, hp.Naprime, hp.Njw
    )