from .backward import solve_backward_transition
from .forward import solve_forward_transition
from .market_clearing import check_market_clearing


def solve_transition_path(
    tr_setting, setting, K_path, opt_indexes, aprimes, V_fin, mu_ini
):
    """
    移行過程の反復アルゴリズムを実行

    numba最適化版の関数群を使用して高速化

    Parameters
    ----------
    tr_setting : TransitionSetting
        移行過程設定
    setting : Setting
        定常状態設定
    K_path : np.ndarray
        初期資本パス (NT,)
    opt_indexes : np.ndarray
        政策関数インデックス (NT, NJ, Nl, Na)
    aprimes : np.ndarray
        政策関数実数値 (NT, NJ, Nl, Na)
    V_fin : np.ndarray
        最終定常状態の価値関数 (NJ, Nl, Na)
    mu_ini : np.ndarray
        初期定常状態の分布 (NJ, Nl, Na)

    Returns
    -------
    tuple
        (収束した資本パス, 価値関数)
        価値関数は (T, NJ, Nl, Na) の配列
    """
    K_path_current = K_path.copy()
    value_functions = None  # 価値関数を保存

    for iteration in range(tr_setting.maxiterTR):
        print(f"Iteration {iteration + 1}")

        # a. T→1期（後ろ向き）: 政策関数を計算
        value_functions = solve_backward_transition(
            tr_setting, setting, K_path_current, opt_indexes, aprimes, V_fin
        )

        # b. 1→T期（前向き）: 分布更新
        mu_dist_path = solve_forward_transition(
            tr_setting, setting, opt_indexes, mu_ini
        )

        # c. 分布更新後のチェック
        K_path_new, max_error = check_market_clearing(
            tr_setting, setting, K_path_current, mu_dist_path, aprimes
        )

        print(f"  Market clearing error: {max_error:.6e}")

        # 収束判定
        if max_error < tr_setting.errKTol:
            print(f"  収束しました！（誤差: {max_error:.6e}）")
            K_path_current = K_path_new
            break

        # 資本パスを更新
        K_path_current = K_path_new

    return K_path_current, value_functions
