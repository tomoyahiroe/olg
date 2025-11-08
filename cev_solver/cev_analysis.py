"""
CEV Analysis Module

CEV分析の統合機能を提供するモジュール
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from .cev_calculator import calculate_cev
from .cev_plotter import plot_cev_by_age, plot_cev_by_cohort


def run_cev_analysis(
    tr_setting: Any,
    initial_result: Any,
    value_functions: Dict[str, np.ndarray],
    mu_dists: np.ndarray,
    plot_period: int = 0,
    plot_age: int = 0,
    show_plots: bool = True,
    save_plots: bool = False,
    save_prefix: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    CEV分析を実行

    Parameters
    ----------
    tr_setting : TransitionSetting
        移行過程設定
    initial_result : SteadyStateResult
        初期定常状態の結果
    value_functions : dict
        価値関数情報 {'V_init': 改革時点価値関数, 'V_start': 各期出生時価値}
    mu_dists : np.ndarray
        移行過程の分布パス (NT, NJ, Nl, Na)
    plot_period : int, default=0
        プロットする期間（0ベース）
    plot_age : int, default=0
        コホートプロット用の年齢インデックス
    show_plots : bool, default=True
        プロットを表示するかどうか
    save_plots : bool, default=False
        プロットを保存するかどうか
    save_prefix : str, optional
        保存ファイル名のプレフィックス

    Returns
    -------
    dict
        CEV分析結果
        - 'cev_avg': CEV平均 (NT, NJ, Nl)
    """
    print("=== CEV分析開始 ===")

    # CEVの計算
    print("CEVを計算中...")
    cev_avg = calculate_cev(tr_setting, initial_result, value_functions, mu_dists)

    print("CEV計算完了")

    # 3. プロット作成
    if show_plots or save_plots:
        print("プロットを作成中...")

        # 年齢別CEVプロット
        if show_plots:
            plot_cev_by_age(cev_avg, period=plot_period, initial_result=initial_result)

        if save_plots and save_prefix:
            save_path = f"{save_prefix}_cev_by_age_period_{plot_period}.png"
            plot_cev_by_age(
                cev_avg,
                period=plot_period,
                initial_result=initial_result,
                save_path=save_path,
            )

        # コホート別CEVプロット
        if show_plots:
            plot_cev_by_cohort(cev_avg, age=plot_age, initial_result=initial_result)

        if save_plots and save_prefix:
            save_path = f"{save_prefix}_cev_by_cohort_age_{plot_age}.png"
            plot_cev_by_cohort(
                cev_avg,
                age=plot_age,
                initial_result=initial_result,
                save_path=save_path,
            )

        print("プロット完了")

    # 4. 結果の要約統計
    print("\n=== CEV分析結果サマリー ===")
    print(f"期間数: {cev_avg.shape[0]}")
    print(f"年齢数: {cev_avg.shape[1]}")
    print(f"スキル数: {cev_avg.shape[2]}")

    print(f"\n第{plot_period+1}期の年齢別CEV統計:")
    for i_l in range(cev_avg.shape[2]):
        skill_name = "Low" if i_l == 0 else "High"
        cev_mean = np.mean(cev_avg[plot_period, :, i_l])
        print(f"  {skill_name} skill - CEV平均: {cev_mean:.4f}")

    return {"cev_avg": cev_avg}
