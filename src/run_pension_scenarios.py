"""
年金所得代替率の複数シナリオ分析スクリプト

50% → 0%, 25%, 50%, 75%, 100% の5つのシナリオを実行し、
それぞれ別フォルダに結果を保存する。
"""

import os
import pickle
from datetime import datetime

from olg.transition import run_transition_analysis
from olg.transition.setting import TransitionSetting


def save_results_to_pkl(results, output_dir="results"):
    """結果をpickleファイルとして保存"""

    os.makedirs(output_dir, exist_ok=True)

    # 結果辞書から各要素を取得
    components = {
        "initial_result": results["initial_result"],
        "final_result": results["final_result"],
        "K_path": results["converged_K_path"],
        "opt_indexes": results["opt_indexes"],
        "aprimes": results["aprimes"],
        "value_functions": results["value_functions"],
        "mu_dist_path": results["mu_dist_path"],
    }

    saved_files = []
    for name, data in components.items():
        filename = f"{output_dir}/{name}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        saved_files.append(filename)

    # 全体も一つのファイルとして保存
    full_filename = f"{output_dir}/full.pkl"
    with open(full_filename, "wb") as f:
        pickle.dump(results, f)
    saved_files.append(full_filename)

    return saved_files


def run_pension_scenario(psi_fin, output_dir_name):
    """
    単一の年金シナリオを実行

    Parameters
    ----------
    psi_fin : float
        最終所得代替率
    output_dir_name : str
        出力ディレクトリ名
    """
    print(f"\n=== シナリオ: 50% → {psi_fin * 100:.0f}% ===")

    # 移行過程設定
    tr_setting = TransitionSetting(
        NT=100,  # 移行期間
        TT=25,  # 政策変更期間
        psi_ini=0.5,  # 初期所得代替率 50%
        psi_fin=psi_fin,  # 最終所得代替率
    )

    # 初期・最終定常状態用設定
    initial_setting, final_setting = tr_setting.create_ss_settings(Na=101, Naprime=2001)

    # 移行過程分析を実行
    results = run_transition_analysis(tr_setting, initial_setting, final_setting)

    # 結果を保存
    saved_files = save_results_to_pkl(results, output_dir_name)

    print(f"結果を {output_dir_name} に保存しました")
    print(f"保存ファイル数: {len(saved_files)}")

    return results


def main():
    """メイン実行関数"""

    print("=== 年金所得代替率シナリオ分析 ===")

    # シナリオ設定 (最終所得代替率, 出力ディレクトリ名)
    scenarios = [
        (0.0000001, "psi_50to00"),  # 50% → 0%
        (0.25, "psi_50to25"),  # 50% → 25%
        (0.50, "psi_50to50"),  # 50% → 50% (変化なし)
        (0.75, "psi_50to75"),  # 50% → 75%
        (1.00, "psi_50to100"),  # 50% → 100%
    ]

    results_dict = {}

    # 各シナリオを実行
    for psi_fin, output_dir in scenarios:
        try:
            start_time = datetime.now()
            results = run_pension_scenario(psi_fin, output_dir)
            end_time = datetime.now()

            results_dict[output_dir] = results

            duration = (end_time - start_time).total_seconds()
            print(f"実行時間: {duration:.1f}秒")

        except Exception as e:
            print(f"エラーが発生しました (シナリオ: {output_dir}): {e}")
            continue

    print("\n=== 全シナリオ完了 ===")
    print(f"成功したシナリオ数: {len(results_dict)}/5")
    print("出力フォルダ:")
    for output_dir in results_dict.keys():
        print(f"  {output_dir}/")


if __name__ == "__main__":
    main()
