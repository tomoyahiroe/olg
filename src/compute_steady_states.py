"""
異なる所得代替率での定常状態計算スクリプト

0%, 25%, 50%, 75%, 100% の5つの所得代替率で定常状態を計算し、
それぞれ別フォルダに結果を保存する。
"""

import os
import pickle
from datetime import datetime

from olg.ss.setting import Setting
from olg.ss.solve_ss import solve_ss


def save_steady_state_result(result, output_dir="steady_state_results"):
    """定常状態結果をpickleファイルとして保存"""

    os.makedirs(output_dir, exist_ok=True)

    # SteadyStateResultオブジェクト全体を保存
    result_filename = f"{output_dir}/steady_state_result.pkl"
    with open(result_filename, "wb") as f:
        pickle.dump(result, f)

    # 主要な結果を個別ファイルとしても保存
    components = {
        "K": result.K,  # 資本ストック
        "L": result.L,  # 労働供給
        "r": result.r,  # 利子率
        "w": result.w,  # 賃金率
        "p": result.p,  # 年金給付
        "value_fun_box": result.value_fun_box,  # 価値関数
        "policy_fun_box": result.policy_fun_box,  # 政策関数（実数値）
        "optaprime_index_box": result.optaprime_index_box,  # 政策関数（インデックス）
        "mu_dist_box": result.mu_dist_box,  # 状態分布
    }

    saved_files = []
    for name, data in components.items():
        filename = f"{output_dir}/{name}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        saved_files.append(filename)

    saved_files.append(result_filename)
    return saved_files


def compute_steady_state(psi, output_dir_name):
    """
    単一の所得代替率で定常状態を計算

    Parameters
    ----------
    psi : float
        所得代替率
    output_dir_name : str
        出力ディレクトリ名
    """
    print(f"\n=== 所得代替率: {psi * 100:.0f}% の定常状態計算 ===")

    # 定常状態設定
    setting = Setting(
        psi=psi,  # 所得代替率
        Na=101,  # 資産グリッド数
        Naprime=2001,  # 政策関数用資産グリッド数
    )

    # 定常状態を計算
    result = solve_ss(setting)

    # 結果を保存
    saved_files = save_steady_state_result(result, output_dir_name)

    print(f"結果を {output_dir_name} に保存しました")
    print(f"保存ファイル数: {len(saved_files)}")
    print(f"均衡資本ストック K = {result.K:.4f}")
    print(f"利子率 r = {result.r:.4f}")
    print(f"賃金率 w = {result.w:.4f}")
    print(f"年金給付 p = {result.p:.4f}")

    return result


def main():
    """メイン実行関数"""

    print("=== 異なる所得代替率での定常状態計算 ===")

    # シナリオ設定 (所得代替率, 出力ディレクトリ名)
    # 0-50%の範囲をより詳細に実験
    scenarios = [
        (0.0000001, "ss_psi_00"),  # 0%
        (0.05, "ss_psi_05"),  # 5%
        (0.10, "ss_psi_10"),  # 10%
        (0.15, "ss_psi_15"),  # 15%
        (0.20, "ss_psi_20"),  # 20%
        (0.25, "ss_psi_25"),  # 25%
        (0.30, "ss_psi_30"),  # 30%
        (0.35, "ss_psi_35"),  # 35%
        (0.40, "ss_psi_40"),  # 40%
        (0.45, "ss_psi_45"),  # 45%
        (0.50, "ss_psi_50"),  # 50%
        (0.75, "ss_psi_75"),  # 75%
        (1.00, "ss_psi_100"),  # 100%
    ]

    results_dict = {}

    # 各シナリオを実行
    for psi, output_dir in scenarios:
        try:
            start_time = datetime.now()
            result = compute_steady_state(psi, output_dir)
            end_time = datetime.now()

            results_dict[output_dir] = result

            duration = (end_time - start_time).total_seconds()
            print(f"実行時間: {duration:.1f}秒")

        except Exception as e:
            print(f"エラーが発生しました (シナリオ: {output_dir}): {e}")
            continue

    print("\n=== 全シナリオ完了 ===")
    print(f"成功したシナリオ数: {len(results_dict)}/{len(scenarios)}")
    print("出力フォルダ:")
    for output_dir in results_dict.keys():
        print(f"  {output_dir}/")

    # 結果サマリーを表示
    if results_dict:
        print("\n=== 結果サマリー ===")
        print("所得代替率 | 資本ストック | 利子率   | 賃金率   | 年金給付")
        print("---------|------------|--------|--------|--------")
        for output_dir, result in results_dict.items():
            psi_str = output_dir.split("_")[-1] + "%"
            print(
                f"{psi_str:>8s} | {result.K:>10.4f} | "
                f"{result.r:>6.4f} | {result.w:>6.4f} | {result.p:>6.4f}"
            )


if __name__ == "__main__":
    main()
