import numpy as np
from numba import njit


@njit
def calculate_total_asset_supply_numba(mu_dist, aprimes_policy, NJ, Nl, Na):
    """
    総資産供給を計算（numba最適化版）
    
    A_t = Σ_h Σ_i_l Σ_i_a a'_{h,i_l,i_a} * μ_{h,i_l,i_a}
    """
    total_supply = 0.0
    
    for h in range(NJ):
        for i_l in range(Nl):
            for i_a in range(Na):
                mu = mu_dist[h, i_l, i_a]
                if mu > 0:  # 人口が存在する場合のみ
                    aprime = aprimes_policy[h, i_l, i_a]  # 次期資産選択
                    total_supply += aprime * mu
    
    return total_supply


@njit
def update_capital_path_numba(K_path_current, K_path_new, adjK_TR, NT):
    """
    資本パスの更新（numba最適化版）
    """
    # K_1は所与なので更新しない
    for t in range(1, NT):
        K_path_new[t] = K_path_current[t] + adjK_TR * (K_path_new[t] - K_path_current[t])


def check_market_clearing(tr_setting, setting, K_path_current, mu_dist_path, aprimes):
    """
    分布更新後のチェック：市場クリア条件と資本パス更新
    
    numba最適化版を使用して高速化
    """
    # 新しい資本パスを計算
    K_path_new = np.zeros(tr_setting.NT)
    error_path = np.zeros(tr_setting.NT)
    
    # K_1は所与（初期条件）
    K_path_new[0] = K_path_current[0]
    error_path[0] = 0.0
    
    # t=2,3,...,T期の資本ストックを計算
    for t in range(1, tr_setting.NT):
        # 期間tでの総資産供給を計算（numba最適化版）
        A_supply_t = calculate_total_asset_supply_numba(
            mu_dist_path[t-1], aprimes[t-1], 
            setting.NJ, setting.Nl, setting.Na
        )
        
        # 新しい資本ストック（次期の期首資本）
        K_path_new[t] = A_supply_t
        
        # 市場クリア誤差を計算
        error_path[t] = abs(K_path_new[t] - K_path_current[t])
    
    # 最大誤差を計算
    max_error = np.max(error_path)
    
    # 資本パスを更新（調整係数を使用）
    if max_error > tr_setting.errKTol:
        update_capital_path_numba(K_path_current, K_path_new, tr_setting.adjK_TR, tr_setting.NT)
    
    return K_path_new, max_error