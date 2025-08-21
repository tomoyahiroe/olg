import numpy as np
import matplotlib.pyplot as plt

from olg_solver import solve_ss, Setting, SteadyStateResult, TransitionSetting
from olg_transition_solver import create_capital_guess, solve_transition_path

def main():
    print("=== OLG移行過程分析 ===")
    
    # 移行過程の設定を作成
    tr_setting = TransitionSetting(
        NT=50,    # テスト用に短い移行期間
        TT=5,    # テスト用に短い政策変更期間
        psi_ini=0.5,  # 初期所得代替率
        psi_fin=0.25  # 最終所得代替率
    )
    
    # 移行過程設定のサマリーを表示
    tr_setting.print_transition_summary()
    
    # 初期・最終定常状態用のSetting作成（軽量設定）
    print("\\n=== 定常状態用設定の作成 ===")
    initial_setting, final_setting = tr_setting.create_ss_settings(
        Na=101,        # 軽量テスト用
        Naprime=1001   # 軽量テスト用
    )
    
    print(f"初期定常状態設定: ψ = {initial_setting.psi:.3f}")
    print(f"最終定常状態設定: ψ = {final_setting.psi:.3f}")
    
    # 1. 初期・最終定常状態の計算
    print("\\n=== 初期定常状態の計算 ===")
    print("初期定常状態を計算中...")
    initial_result = solve_ss(initial_setting)
    mu_ini = initial_result.mu_dist_box
    K_ini = initial_result.K
    
    print("\\n=== 最終定常状態の計算 ===")
    print("最終定常状態を計算中...")
    final_result = solve_ss(final_setting)
    V_fin = final_result.value_fun_box
    K_fin = final_result.K
    
    # 2. 移行過程の初期設定
    print("\\n=== 移行過程の初期設定 ===")
    K_path = create_capital_guess(tr_setting, K_ini, K_fin)
    
    # 2.i. 政策関数とそのインデックスの箱を用意
    opt_indexes, aprimes = create_policy_function_boxes(tr_setting, initial_setting)

    # 2.ii. 次のステップを収束するまで繰り返す
    print("\\n=== 移行過程の反復計算 ===")
    converged_K_path = solve_transition_path(
        tr_setting, initial_setting, K_path, opt_indexes, aprimes, V_fin, mu_ini
    )
    
    return tr_setting, initial_result, final_result, converged_K_path, opt_indexes, aprimes

def create_policy_function_boxes(tr_setting, setting):
    """
    移行過程の政策関数とそのインデックスの箱を作成
    - opt_indexes: 政策関数インデックス (NT, NJ, Nl, Na)
    - aprimes: 政策関数実数値 (NT, NJ, Nl, Na)
    
    Parameters
    ----------
    tr_setting : TransitionSetting
        移行過程設定
    setting : Setting
        定常状態設定（次元情報取得用）
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (opt_indexes, aprimes) - 政策関数インデックスと実数値の箱
    """
    # 期間×年齢×スキル×資産の4次元配列
    shape = (tr_setting.NT, setting.NJ, setting.Nl, setting.Na)
    
    # 政策関数インデックスの箱（整数型）
    opt_indexes = np.zeros(shape, dtype=np.int32)
    
    # 政策関数実数値の箱（浮動小数点型）
    aprimes = np.zeros(shape, dtype=np.float64)
    
    return opt_indexes, aprimes


if __name__ == "__main__":
    main()