"""
OLG Model Integration Module

This module provides a simple interface to run OLG transition analysis
using the functions defined in olg_transition_solver.
"""

import pickle
import os
from datetime import datetime
from olg_solver.transition_setting import TransitionSetting
from olg_transition_solver import (
    run_transition_analysis,
)


def save_results_to_pkl(results, prefix="results", timestamp=True):
    """
    結果をpickleファイルとして保存

    Parameters
    ----------
    results : tuple
        (initial_result, final_result, K_path, opt_indexes, aprimes, value_functions)
    prefix : str
        ファイル名のプレフィックス
    timestamp : bool
        タイムスタンプをファイル名に含めるか
    """
    # 結果をアンパック
    initial_result, final_result, K_path, opt_indexes, aprimes, value_functions = (
        results
    )

    # タイムスタンプ付きのディレクトリを作成
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results_{timestamp_str}"
    else:
        output_dir = "results"

    os.makedirs(output_dir, exist_ok=True)

    # 各結果をpklファイルとして保存
    components = {
        "initial_result": initial_result,
        "final_result": final_result,
        "K_path": K_path,
        "opt_indexes": opt_indexes,
        "aprimes": aprimes,
        "value_functions": value_functions,
    }

    saved_files = []
    for name, data in components.items():
        filename = f"{output_dir}/{prefix}_{name}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        saved_files.append(filename)
        print(f"Saved: {filename}")

    # 全体も一つのファイルとして保存
    full_filename = f"{output_dir}/{prefix}_full.pkl"
    with open(full_filename, "wb") as f:
        pickle.dump(results, f)
    saved_files.append(full_filename)
    print(f"Saved: {full_filename}")

    return saved_files


def run_custom_transition_analysis():
    """カスタム設定で移行過程分析を実行"""

    # カスタム移行過程設定を作成
    tr_setting = TransitionSetting(
        NT=100,  # 移行期間
        TT=25,  # 政策変更期間
        psi_ini=0.5,  # 初期所得代替率
        psi_fin=0.000000001,  # 最終所得代替率
    )

    # 初期・最終定常状態用のSetting作成
    initial_setting, final_setting = tr_setting.create_ss_settings(
        Na=101, Naprime=2001  # 資産グリッド数  # 政策関数用資産グリッド数
    )

    # 移行過程分析を実行
    results = run_transition_analysis(tr_setting, initial_setting, final_setting)

    return results


if __name__ == "__main__":
    results = run_custom_transition_analysis()

    print("\n=== カスタム設定結果の保存 ===")
    saved_files = save_results_to_pkl(results, prefix="psi_50to0_by25T")

    print("\n=== 分析完了 ===")
    print("結果は results_default, results_custom に格納され、")
    print("以下のpklファイルとして保存されました:")
    print("\nカスタム設定:")
    for f in saved_files:
        print(f"  {f}")
