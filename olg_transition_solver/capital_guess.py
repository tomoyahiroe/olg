import numpy as np
from numba import njit


@njit
def create_capital_guess_numba(NT, K_ini, K_fin):
    """
    移行過程における総資本の当て推量を作成（numba最適化版）
    
    教科書版: 30期間かけてK_iniからK_finに線形増加し、以降はK_finで一定
    """
    KT = np.zeros(NT)
    
    # 教科書版の実装（30期間で収束）
    convergence_period = min(30, NT)
    
    for t in range(NT):
        if t < convergence_period:
            if convergence_period > 1:
                KT[t] = K_ini + t * (K_fin - K_ini) / (convergence_period - 1)
            else:
                KT[t] = K_fin
        else:
            KT[t] = K_fin
    
    return KT


def create_capital_guess(tr_setting, K_ini, K_fin):
    """
    移行過程における総資本の当て推量を作成
    
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
    return create_capital_guess_numba(tr_setting.NT, K_ini, K_fin)