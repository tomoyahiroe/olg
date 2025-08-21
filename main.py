import numpy as np
import matplotlib.pyplot as plt

from olg_solver import solve_ss, Setting, SteadyStateResult, TransitionSetting

def main():
    print("=== OLG移行過程分析 ===")
    
    # 移行過程の設定を作成
    tr_setting = TransitionSetting(
        NT=50,    # テスト用に短い移行期間
        TT=20,    # テスト用に短い政策変更期間
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
        tr_setting, initial_setting, K_path, opt_indexes, aprimes, V_fin
    )
    
    return tr_setting, initial_result, final_result, converged_K_path, opt_indexes, aprimes

def solve_transition_path(tr_setting, setting, K_path, opt_indexes, aprimes, V_fin):
    """
    移行過程の反復アルゴリズムを実行
    
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
        
    Returns
    -------
    np.ndarray
        収束した資本パス
    """
    K_path_current = K_path.copy()
    
    for iteration in range(tr_setting.maxiterTR):
        print(f"Iteration {iteration + 1}")
        
        # a. T→1期（後ろ向き）: 政策関数を計算
        solve_backward_transition(
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
        
        # 今回は1回だけ実行してテスト
        break
    
    return K_path_current

def solve_backward_transition(tr_setting, setting, K_path, opt_indexes, aprimes, V_fin):
    """
    T→1期（後ろ向き）の政策関数計算
    
    移行過程用の家計最適化を実装
    家計は将来の政策変化を予見して最適化する
    """
    from olg_solver.utils import inverse_interp_aprime_point_numba
    
    # 年齢別人口分布（定常状態と同じ）
    h_dist = np.ones(setting.NJ) / setting.NJ
    L = np.sum(h_dist[:setting.Njw])
    
    # 全期間の経済環境を事前計算
    factor_prices = calculate_all_factor_prices(tr_setting, setting, K_path, L)
    
    # 最初にt+1期の価値関数として最終定常状態の価値関数を設定
    value_fun_box = V_fin.copy()
    
    # T期から1期まで後ろ向きに計算
    for t in range(tr_setting.NT - 1, -1, -1):  # T-1, T-2, ..., 0
        print(f"  期間 {t+1}/{tr_setting.NT} を計算中...")
        
        # 当期の価値関数・政策関数の箱
        vfun_current = np.zeros((setting.NJ, setting.Nl, setting.Na))
        afun_current = np.zeros((setting.NJ, setting.Nl, setting.Na))
        afun_index_current = np.zeros((setting.NJ, setting.Nl, setting.Na), dtype=int)
        
        # 家計最適化問題を解く
        solve_household_transition_period(
            t, tr_setting, setting, factor_prices, value_fun_box, 
            vfun_current, afun_current, afun_index_current
        )
        
        # 結果を保存
        aprimes[t] = afun_current.copy()
        opt_indexes[t] = afun_index_current.copy()
        
        # 次のループのために価値関数を更新
        value_fun_box = vfun_current.copy()

def calculate_all_factor_prices(tr_setting, setting, K_path, L):
    """
    全期間の要素価格を事前計算
    
    Returns
    -------
    dict
        各期の r_t, w_t, p_t を含む辞書
    """
    factor_prices = {}
    
    for t in range(tr_setting.NT):
        K_t = K_path[t]
        psi_t = tr_setting.rhoT[t]
        
        r_t = setting.alpha * (K_t / L) ** (setting.alpha - 1) - setting.delta
        w_t = (1 - setting.alpha) * (K_t / L) ** setting.alpha
        p_t = psi_t * w_t
        tau_t = tr_setting.tauT[t]
        
        factor_prices[t] = {
            'r': r_t,
            'w': w_t, 
            'p': p_t,
            'tau': tau_t,
            'psi': psi_t
        }
    
    return factor_prices

def solve_household_transition_period(t, tr_setting, setting, factor_prices, value_fun_next, 
                                    vfun_current, afun_current, afun_index_current):
    """
    period t での家計最適化問題を解く
    
    家計は将来の政策変化を予見して最適化する
    """
    from olg_solver.utils import inverse_interp_aprime_point_numba
    
    # 当期の要素価格と所得を取得
    r_t = factor_prices[t]['r']
    w_t = factor_prices[t]['w']
    tau_t = factor_prices[t]['tau']
    
    # 1. 最終年齢（NJ-1、0ベース）の政策関数・価値関数を求める
    for i_l in range(setting.Nl):
        for i_a in range(setting.Na):
            # 最終期の資産は0
            afun_current[setting.NJ-1, i_l, i_a] = 0.0
            afun_index_current[setting.NJ-1, i_l, i_a] = 0
            
            # 所得を計算（将来の政策変化を予見）
            y = calculate_future_income(setting.NJ-1, i_l, t, tr_setting, setting, factor_prices)
            
            # 予算制約より消費を計算
            c = y + (1 + r_t) * setting.a_grid[i_a] - 0.0  # 次期資産=0
            
            # 価値関数を計算
            if setting.gamma == 1:
                vfun_current[setting.NJ-1, i_l, i_a] = np.log(c)
            else:
                vfun_current[setting.NJ-1, i_l, i_a] = c**(1 - setting.gamma) / (1 - setting.gamma)
    
    # 2. 後ろ向きに各年齢の政策関数・価値関数を求める
    for h in range(setting.NJ-2, -1, -1):  # NJ-2歳から0歳まで
        for i_l in range(setting.Nl):
            # 所得を計算（将来の政策変化を予見）
            y = calculate_future_income(h, i_l, t, tr_setting, setting, factor_prices)
            
            for i_a in range(setting.Na):
                a = setting.a_grid[i_a]
                
                # 価値関数の初期化（大きなマイナス値）
                vtemp = np.full(setting.Naprime, -1e10)
                accmax = setting.Naprime
                
                # 次期資産の選択肢をグリッドサーチ
                for j_a in range(setting.Naprime):
                    aprime = setting.aprime_grid[j_a]
                    
                    # 当期消費の計算
                    c = y + (1 + r_t) * a - aprime
                    
                    # 消費がマイナスの場合、それ以降のグリッドは除外
                    if c <= 0:
                        accmax = j_a
                        break
                    
                    # 線形補間のためのインデックスと重みを取得
                    _, ac1vec, ac2vec, weight_1, weight_2 = inverse_interp_aprime_point_numba(setting.a_grid, aprime)
                    
                    # 来期の期待価値を計算
                    vprime = 0.0
                    for j_l in range(setting.Nl):
                        prob = setting.P[i_l, j_l]  # 生産性遷移確率
                        V_next = (weight_1 * value_fun_next[h+1, j_l, ac1vec] + 
                                 weight_2 * value_fun_next[h+1, j_l, ac2vec])
                        vprime += prob * V_next
                    
                    # 価値関数を計算
                    if setting.gamma == 1:
                        vtemp[j_a] = np.log(c) + setting.beta * vprime
                    else:
                        vtemp[j_a] = c**(1 - setting.gamma) / (1 - setting.gamma) + setting.beta * vprime
                
                # 最大価値関数を探す
                if accmax > 0:
                    opt_index = np.argmax(vtemp[:accmax])
                    vfun_current[h, i_l, i_a] = vtemp[opt_index]
                    afun_index_current[h, i_l, i_a] = opt_index
                    afun_current[h, i_l, i_a] = setting.aprime_grid[opt_index]

def calculate_future_income(h, i_l, t, tr_setting, setting, factor_prices):
    """
    年齢h、生産性i_lの家計が期間tで直面する所得を計算
    
    家計は将来の政策変化を予見して、現在の年齢に対応する将来期間での所得を計算
    """
    current_age = h
    future_period = t  # 単純化：現在期間の政策を使用
    
    # 将来期間がtr_setting.NTを超える場合は最終期の政策を使用
    if future_period >= tr_setting.NT:
        future_period = tr_setting.NT - 1
    
    w_t = factor_prices[future_period]['w']
    tau_t = factor_prices[future_period]['tau']
    p_t = factor_prices[future_period]['p']
    
    if current_age < setting.Njw:  # 労働期間
        return (1 - tau_t) * w_t * setting.l_grid[i_l]
    else:  # 引退期間
        return p_t

def solve_forward_transition(tr_setting, setting, opt_indexes, mu_ini):
    """
    1→T期（前向き）の分布更新
    
    移行過程の分布経路を計算
    """
    from olg_solver.utils import inverse_interp_aprime_point_numba
    
    # 移行過程の状態分布の箱を用意 (NT, NJ, Nl, Na)
    mu_dist_path = np.zeros((tr_setting.NT, setting.NJ, setting.Nl, setting.Na))
    
    # t=1期: 初期定常状態の分布を設定
    mu_dist_path[0] = mu_ini.copy()
    mea_current = mu_ini.copy()  # 計算過程で使用する現在期の分布
    
    # t=2,3,...,T期まで前向きに分布を更新
    for t in range(1, tr_setting.NT):  # t=1,2,...,NT-1 (0ベース)
        print(f"  分布更新: 期間 {t}/{tr_setting.NT-1}")
        
        # 次期の分布を初期化
        mea_next = np.zeros((setting.NJ, setting.Nl, setting.Na))
        
        # 0歳の家計：資産0、スキル50%ずつ
        birth_mass = 1.0 / setting.NJ  # 年齢別人口は均等
        mea_next[0, 0, 0] = 0.5 * birth_mass  # low productivity
        mea_next[0, 1, 0] = 0.5 * birth_mass  # high productivity
        
        # 1歳以上の家計：前期の最適化結果に基づいて分布更新
        for h in range(setting.NJ - 1):  # h=0,1,...,NJ-2 (次期でh+1歳になる)
            for i_l in range(setting.Nl):
                for i_a in range(setting.Na):
                    mu = mea_current[h, i_l, i_a]
                    
                    if mu > 0:  # 人口が存在する場合のみ処理
                        # 最適な次期資産のインデックスを取得
                        opt_index = opt_indexes[t-1, h, i_l, i_a]  # t-1期の政策関数を使用
                        aprime = setting.aprime_grid[opt_index]
                        
                        # 線形補間用のインデックスと重みを取得
                        _, i_opt1, i_opt2, weight_1, weight_2 = inverse_interp_aprime_point_numba(setting.a_grid, aprime)
                        
                        # スキル遷移確率を使って分布を更新
                        for j_l in range(setting.Nl):
                            prob = setting.P[i_l, j_l]  # スキル遷移確率
                            
                            # 線形補間で次期分布に割り当て
                            mea_next[h+1, j_l, i_opt1] += mu * prob * weight_1
                            mea_next[h+1, j_l, i_opt2] += mu * prob * weight_2
        
        # 計算結果を保存
        mu_dist_path[t] = mea_next.copy()
        mea_current = mea_next.copy()
    
    return mu_dist_path

def check_market_clearing(tr_setting, setting, K_path_current, mu_dist_path, aprimes):
    """
    分布更新後のチェック：市場クリア条件と資本パス更新
    
    """
    # 新しい資本パスを計算
    K_path_new = np.zeros(tr_setting.NT)
    error_path = np.zeros(tr_setting.NT)
    
    # K_1は所与（初期条件）
    K_path_new[0] = K_path_current[0]
    error_path[0] = 0.0
    
    # t=2,3,...,T期の資本ストックを計算
    for t in range(1, tr_setting.NT):
        # 期間tでの総資産供給を計算
        A_supply_t = calculate_total_asset_supply(setting, mu_dist_path[t-1], aprimes[t-1])
        
        # 新しい資本ストック（次期の期首資本）
        K_path_new[t] = A_supply_t
        
        # 市場クリア誤差を計算
        error_path[t] = abs(K_path_new[t] - K_path_current[t])
    
    # 最大誤差を計算
    max_error = np.max(error_path)
    
    # 資本パスを更新（調整係数を使用）
    if max_error > tr_setting.errKTol:
        # K_1は所与なので更新しない
        for t in range(1, tr_setting.NT):
            K_path_new[t] = K_path_current[t] + tr_setting.adjK_TR * (K_path_new[t] - K_path_current[t])
    
    return K_path_new, max_error

def calculate_total_asset_supply(setting, mu_dist, aprimes_policy):
    """
    総資産供給を計算
    
    A_t = Σ_h Σ_i_l Σ_i_a a'_{h,i_l,i_a} * μ_{h,i_l,i_a}
    """
    total_supply = 0.0
    
    for h in range(setting.NJ):
        for i_l in range(setting.Nl):
            for i_a in range(setting.Na):
                mu = mu_dist[h, i_l, i_a]
                if mu > 0:  # 人口が存在する場合のみ
                    aprime = aprimes_policy[h, i_l, i_a]  # 次期資産選択
                    total_supply += aprime * mu
    
    return total_supply
    

def create_capital_guess(tr_setting, K_ini, K_fin):
    """
    移行過程における総資本の当て推量を作成
    - 教科書版: 30期間かけてK_iniからK_finに線形増加し、以降はK_finで一定
    
    Parameters
    ----------
    tr_setting : TransitionSetting
        移行過程設定
    K_ini : float
        初期定常状態の資本ストック
    K_fin : float
        最終定常状態の資本ストック
        
    Returns
    -------
    np.ndarray
        総資本パス KT (NT,)
    """
    NT = tr_setting.NT
    KT = np.zeros(NT)
    
    # 教科書版の実装（30期間で収束）
    convergence_period = min(30, NT)  # 収束期間（最大30期、またはNTまで）
    
    for t in range(NT):
        if t < convergence_period:
            # 線形補間: K_t = K_ini + (t) * (K_fin - K_ini) / (convergence_period - 1)
            # t=0でK_ini, t=convergence_period-1でK_fin
            if convergence_period > 1:
                KT[t] = K_ini + t * (K_fin - K_ini) / (convergence_period - 1)
            else:
                KT[t] = K_fin
        else:
            # 収束期間以降はK_finで一定
            KT[t] = K_fin
    
    return KT

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
    # 期間×年齢×スキル×資本の4次元配列
    shape = (tr_setting.NT, setting.NJ, setting.Nl, setting.Na)
    
    # 政策関数インデックスの箱（整数型）
    opt_indexes = np.zeros(shape, dtype=np.int32)
    
    # 政策関数実数値の箱（浮動小数点型）
    aprimes = np.zeros(shape, dtype=np.float64)
    
    return opt_indexes, aprimes


if __name__ == "__main__":
    main()