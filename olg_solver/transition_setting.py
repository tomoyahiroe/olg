from typing import Tuple
import numpy as np
import numpy.typing as npt
from .setting import Setting


class TransitionSetting:
    """
    移行過程計算用の設定クラス
    
    移行過程に必要な固有パラメータと時間軸での政策パスを管理します。
    """
    
    # 移行過程固有パラメータ（整数型）
    NT: int          # 移行期間
    TT: int          # 政策変更の収束期間
    maxiterTR: int   # 移行過程用の最大繰り返し回数
    
    # 移行過程固有パラメータ（浮動小数点型）
    psi_ini: float   # 初期定常状態の所得代替率
    psi_fin: float   # 最終定常状態の所得代替率
    errKTol: float   # 移行過程用の市場クリア誤差許容度
    adjK_TR: float   # 移行過程用の資本更新調整係数
    
    # 移行過程用の時間軸配列
    rhoT: npt.NDArray[np.floating]    # 時間ごとの所得代替率パス (NT,)
    tauT: npt.NDArray[np.floating]    # 時間ごとの税率パス (NT,)

    def __init__(
        self,
        # 移行過程固有パラメータ
        NT: int = 100,        # 移行期間
        TT: int = 25,         # 政策変更の収束期間
        psi_ini: float = 0.5, # 初期定常状態の所得代替率
        psi_fin: float = 0.25,# 最終定常状態の所得代替率
        errKTol: float = 1e-3,# 移行過程用の市場クリア誤差許容度
        maxiterTR: int = 300, # 移行過程用の最大繰り返し回数
        adjK_TR: float = 0.05,# 移行過程用の資本更新調整係数
        
        # 税率計算用パラメータ（人口分布計算に必要）
        NJ: int = 61,         # 年齢数（税率計算用）
        Njw: int = 45         # 労働期間（税率計算用）
    ) -> None:
        
        # 移行過程固有パラメータの設定
        self.NT = NT
        self.TT = TT
        self.psi_ini = psi_ini
        self.psi_fin = psi_fin
        self.errKTol = errKTol
        self.maxiterTR = maxiterTR
        self.adjK_TR = adjK_TR
        
        # 税率計算用パラメータ
        self.NJ = NJ
        self.Njw = Njw
        
        # 時間軸での所得代替率パスを設定
        self.rhoT = self._create_replacement_rate_path()
        
        # 時間軸での税率パスを設定
        self.tauT = self._create_tax_rate_path()
    
    def _create_replacement_rate_path(self) -> npt.NDArray[np.floating]:
        """
        時間軸での所得代替率パスを作成
        
        1期目のpsi_iniからTT期目のpsi_finまで線形に減少し、
        以降は最終期まで psi_fin で一定
        
        Returns
        -------
        npt.NDArray[np.floating]
            所得代替率パス (NT,)
        """
        rhoT = np.zeros(self.NT)
        
        # 1期目からTT期目まで線形に変化
        for t in range(self.TT):
            rhoT[t] = self.psi_ini + ((self.psi_fin - self.psi_ini) / (self.TT - 1)) * t
        
        # TT期目以降は psi_fin で一定
        rhoT[self.TT:] = self.psi_fin
        
        return rhoT
    
    def _create_tax_rate_path(self) -> npt.NDArray[np.floating]:
        """
        時間軸での税率パスを作成
        
        各期の所得代替率に基づいて税率を計算:
        τ_t = ψ_t × (退職者人口) / (労働者人口)
        
        Returns
        -------
        npt.NDArray[np.floating]
            税率パス (NT,)
        """
        tauT = np.zeros(self.NT)
        
        # 年齢別人口分布（均等分布を仮定）
        h_dist = np.ones(self.NJ) / self.NJ
        
        # 退職者と労働者の人口比率
        retiree_ratio = np.sum(h_dist[self.Njw:])
        worker_ratio = np.sum(h_dist[:self.Njw])
        
        for t in range(self.NT):
            tauT[t] = self.rhoT[t] * retiree_ratio / worker_ratio
        
        return tauT
    
    def create_ss_settings(self, **kwargs) -> Tuple[Setting, Setting]:
        """
        初期・最終定常状態用のSettingインスタンスを作成
        
        psi以外のパラメータは両方で同じ値を使用します。
        
        Parameters
        ----------
        **kwargs
            Settingクラスのパラメータを上書きする場合に指定
            
        Returns
        -------
        Tuple[Setting, Setting]
            (初期定常状態用Setting, 最終定常状態用Setting)
        """
        # 初期定常状態用の設定
        initial_params = kwargs.copy()
        initial_params['psi'] = self.psi_ini
        initial_setting = Setting(**initial_params)
        
        # 最終定常状態用の設定（psiのみ変更）
        final_params = kwargs.copy()
        final_params['psi'] = self.psi_fin
        final_setting = Setting(**final_params)
        
        return initial_setting, final_setting
    
    # def get_replacement_rate(self, t: int) -> float:
    #     """
    #     指定した期間の所得代替率を取得
    #     
    #     Parameters
    #     ----------
    #     t : int
    #         期間（0ベース）
    #         
    #     Returns
    #     -------
    #     float
    #         所得代替率
    #     """
    #     if 0 <= t < self.NT:
    #         return self.rhoT[t]
    #     else:
    #         raise ValueError(f"期間 t={t} は範囲外です。0 <= t < {self.NT} である必要があります。")
    
    # def get_tax_rate(self, t: int) -> float:
    #     """
    #     指定した期間の税率を取得
    #     
    #     Parameters
    #     ----------
    #     t : int
    #         期間（0ベース）
    #         
    #     Returns
    #     -------
    #     float
    #         税率
    #     """
    #     if 0 <= t < self.NT:
    #         return self.tauT[t]
    #     else:
    #         raise ValueError(f"期間 t={t} は範囲外です。0 <= t < {self.NT} である必要があります。")
    
    def print_transition_summary(self) -> None:
        """移行過程設定のサマリーを表示"""
        print(f"\n=== 移行過程設定サマリー ===")
        print(f"移行期間: {self.NT} 期")
        print(f"政策変更収束期間: {self.TT} 期")
        print(f"初期所得代替率: {self.psi_ini:.3f}")
        print(f"最終所得代替率: {self.psi_fin:.3f}")
        print(f"移行過程用収束判定閾値: {self.errKTol:.1e}")
        print(f"移行過程用最大繰り返し回数: {self.maxiterTR}")
        print(f"移行過程用資本更新調整係数: {self.adjK_TR:.3f}")
        print(f"")
        print(f"主要期間の所得代替率:")
        key_periods = [0, self.TT//4, self.TT//2, self.TT-1, self.NT-1]
        for t in key_periods:
            if t < self.NT:
                print(f"  第{t+1}期: ψ = {self.rhoT[t]:.3f}, τ = {self.tauT[t]:.3f}")