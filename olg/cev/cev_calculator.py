"""
CEV Calculator Module

消費補償変分（Consumption Equivalent Variation）の計算機能
"""

import numpy as np
import numpy.typing as npt
from typing import Any


def calculate_cev(
    tr_setting: Any,
    initial_result: Any,
    value_functions: npt.NDArray,
    mu_dists: npt.NDArray,
) -> npt.NDArray:
    """
    CEVを計算（価値関数の分布で重み付け平均を取ってからCEV計算）

    Parameters
    ----------
    tr_setting : TransitionSetting
        移行過程設定
    initial_result : SteadyStateResult
        初期定常状態の結果
    value_functions : np.ndarray
        価値関数 (T, NJ, Nl, Na)
    mu_dists : np.ndarray
        移行過程の分布パス (NT, NJ, Nl, Na)

    Returns
    -------
    np.ndarray
        CEV平均 (NT, NJ, Nl)
    """
    # 割引係数の計算
    beta_list = np.empty(initial_result.hp.NJ)
    for h in range(initial_result.hp.NJ):
        beta_list[h] = sum(
            [initial_result.hp.beta**H for H in range(initial_result.hp.NJ - h)]
        )

    CEV_avg = np.empty((tr_setting.NT, initial_result.hp.NJ, initial_result.hp.Nl))

    for t in range(tr_setting.NT):
        for h in range(initial_result.hp.NJ):
            discount = beta_list[h]
            for i_l in range(initial_result.hp.Nl):
                if t == 0:
                    # 改革時点の価値関数と比較
                    # 価値関数の分布で重み付け平均を計算
                    total_mass_trans = np.sum(mu_dists[t, h, i_l, :])
                    total_mass_init = np.sum(initial_result.mu_dist_box[h, i_l, :])

                    if total_mass_trans > 0 and total_mass_init > 0:
                        vf_trans_avg = (
                            np.sum(
                                value_functions[0, h, i_l, :] * mu_dists[t, h, i_l, :]
                            )
                            / total_mass_trans
                        )
                        vf_init_avg = (
                            np.sum(
                                initial_result.value_fun_box[h, i_l, :]
                                * initial_result.mu_dist_box[h, i_l, :]
                            )
                            / total_mass_init
                        )
                        CEV_avg[t, h, i_l] = (
                            np.exp((vf_trans_avg - vf_init_avg) / discount) - 1
                        )
                    else:
                        CEV_avg[t, h, i_l] = 0.0
                else:
                    # t期以降の価値関数と比較
                    total_mass_trans = np.sum(mu_dists[t, h, i_l, :])
                    total_mass_init = np.sum(initial_result.mu_dist_box[h, i_l, :])

                    if total_mass_trans > 0 and total_mass_init > 0:
                        vf_trans_avg = (
                            np.sum(
                                value_functions[t - 1, h, i_l, :]
                                * mu_dists[t, h, i_l, :]
                            )
                            / total_mass_trans
                        )
                        vf_init_avg = (
                            np.sum(
                                initial_result.value_fun_box[h, i_l, :]
                                * initial_result.mu_dist_box[h, i_l, :]
                            )
                            / total_mass_init
                        )
                        CEV_avg[t, h, i_l] = (
                            np.exp((vf_trans_avg - vf_init_avg) / discount) - 1
                        )
                    else:
                        CEV_avg[t, h, i_l] = 0.0

    return CEV_avg
