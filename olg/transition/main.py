"""
OLG移行過程分析のメイン実行モジュール

このモジュールは移行過程分析の統合実行機能を提供します。
"""

import numpy as np

from ..ss.solve_ss import solve_ss
from .capital_guess import create_capital_guess
from .setting import TransitionSetting
from .transition_solver import solve_transition_path


def create_policy_function_boxes(tr_setting, setting):
    """
    移行過程の政策関数とそのインデックスの箱を作成

    Parameters
    ----------
    tr_setting : TransitionSetting
        移行過程設定
    setting : Setting
        OLGモデル設定

    Returns
    -------
    tuple
        - opt_indexes: 政策関数インデックス (NT, NJ, Nl, Na)
        - aprimes: 政策関数実数値 (NT, NJ, Nl, Na)
    """
    shape = (tr_setting.NT, setting.NJ, setting.Nl, setting.Na)
    opt_indexes = np.zeros(shape, dtype=np.int32)
    aprimes = np.zeros(shape, dtype=np.float64)
    return opt_indexes, aprimes


def run_transition_analysis(
    tr_setting, initial_setting, final_setting, K_path_guess=None
):
    """
    移行過程分析を実行

    Parameters
    ----------
    tr_setting : TransitionSetting
        移行過程設定
    initial_setting : Setting
        初期定常状態用設定
    final_setting : Setting
        最終定常状態用設定

    Returns
    -------
    tuple
        (initial_result, final_result, K_path, opt_indexes, aprimes, value_functions)
        - initial_result: 初期定常状態の結果
        - final_result: 最終定常状態の結果
        - K_path: 収束した資本パス
        - opt_indexes: 政策関数インデックス
        - aprimes: 政策関数実数値
        - value_functions: 価値関数情報
    """
    print("=== OLG移行過程分析 ===")

    print(f"移行期間: {tr_setting.NT} 期")
    print(f"初期定常状態設定: ψ = {initial_setting.psi:.3f}")
    print(f"最終定常状態設定: ψ = {final_setting.psi:.3f}")

    # 1. 初期・最終定常状態の計算
    print("\n=== 初期定常状態の計算 ===")
    initial_result = solve_ss(initial_setting)
    mu_ini = initial_result.mu_dist_box
    K_ini = initial_result.K

    print("\n=== 最終定常状態の計算 ===")
    final_result = solve_ss(final_setting)
    V_fin = final_result.value_fun_box
    K_fin = final_result.K

    # 2. 移行過程の初期設定
    print("\n=== 移行過程の初期設定 ===")
    if K_path_guess is not None:
        K_path = K_path_guess
    else:
        K_path = create_capital_guess(tr_setting, K_ini, K_fin)

    # 政策関数とそのインデックスの箱を用意
    opt_indexes, aprimes = create_policy_function_boxes(tr_setting, initial_setting)

    # 移行過程の反復計算（価値関数も取得）
    print("\n=== 移行過程の反復計算 ===")
    converged_K_path, value_functions = solve_transition_path(
        tr_setting, initial_setting, K_path, opt_indexes, aprimes, V_fin, mu_ini
    )

    return (
        initial_result,
        final_result,
        converged_K_path,
        opt_indexes,
        aprimes,
        value_functions,
    )


def run_default_transition_analysis():
    """
    デフォルト設定で移行過程分析を実行

    Returns
    -------
    tuple
        run_transition_analysis()と同じ戻り値
    """
    # デフォルト移行過程設定を作成
    tr_setting = TransitionSetting(
        NT=100,  # 移行期間
        TT=25,  # 政策変更期間
        psi_ini=0.5,  # 初期所得代替率
        psi_fin=0.25,  # 最終所得代替率
    )

    tr_setting.print_transition_summary()

    # 初期・最終定常状態用のSetting作成
    print("\n=== 定常状態用設定の作成 ===")
    initial_setting, final_setting = tr_setting.create_ss_settings(
        Na=101,
        Naprime=2001,  # 資産グリッド数  # 政策関数用資産グリッド数
    )

    return run_transition_analysis(tr_setting, initial_setting, final_setting)


if __name__ == "__main__":
    # デフォルト設定で移行過程分析を実行
    results = run_default_transition_analysis()
    print("\n=== 分析完了 ===")
    print("結果は results タプルに格納されました。")
