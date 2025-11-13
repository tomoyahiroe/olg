from typing import Callable
import numpy as np
import numpy.typing as npt
from numba import njit
from .utils import maliar_grid


class Setting:
    """
    設定クラス（Setting）
    このクラスは、OLG（Overlapping Generations）モデルにおける主要なパラメータを管理します。
    割引因子、リスク回避度、資本分配率、減耗率、年金の所得代替率、モデル期間、労働・引退開始年齢、初期資産、収束判定閾値などを属性として保持します。
    また、CRRA型効用関数および限界効用関数を定義し、人口分布および各年齢の労働生産性を初期化します。
    """

    # 基本パラメータ（整数型）
    NJ: int  # モデルの期間（世代数）20歳から80歳まで
    Njw: int  # 働く期間 20歳から64歳まで働く
    Nl: int  # 生産性のグリッドの数 {high,low}の2種類
    Na: int  # 今期の資産グリッドの数
    Naprime: int  # 来季の資産グリッドの数
    maxiter: int  # 最大繰り返し回数

    # 基本パラメータ（浮動小数点型）
    l_dif: float  # 生産性の違い
    a_max: float  # 資本グリッドの最大値
    a_min: float  # 資本グリッドの最小値
    alpha: float  # 資本分配率
    beta: float  # 割引因子
    gamma: float  # 相対的リスク回避度（異時点間の代替弾力性の逆数）
    delta: float  # 固定資本減耗率
    psi: float  # 年金の平均所得代替率
    K0: float  # 初期資産
    tol: float  # 収束判定の閾値
    lambdaR: float  # 資本更新調整係数

    # グリッド配列
    h_grid: npt.NDArray[np.floating]  # 年齢グリッド
    l_grid: npt.NDArray[np.floating]  # 生産性グリッド
    a_grid: npt.NDArray[np.floating]  # 今期資産グリッド
    aprime_grid: npt.NDArray[np.floating]  # 次期資産グリッド
    P: npt.NDArray[np.floating]  # 生産性遷移確率行列

    # 効用関数
    utility: Callable[[float], float]  # CRRA型効用関数
    mutility: Callable[[float], float]  # 限界効用関数

    def __init__(
        self,
        NJ: int = 61,  # モデルの期間,20歳から80歳まで生きる
        Njw: int = 45,  # 働く期間,20歳から64歳まで働く
        # Njr: int = 20,      # 勤労期の初期,20歳から働く 使ってない
        Nl: int = 2,  # 生産性のグリッドの数, {high,low}の2種類
        l_dif: float = 0.3,  # 生産性の違い（6.3設定に合わせて0.3→0.2）
        Na: int = 201,  # 今期の資産グリッドの数
        a_max: float = 25,  # 資本グリッドの最大値
        a_min: float = 0,  # 資本グリッドの最小値
        Naprime: int = 8001,  # 来季の資産グリッドの数
        alpha: float = 0.4,  # 資本分配率
        beta: float = 0.98,  # 割引因子
        gamma: float = 1,  # 相対的リスク回避度(異時点間の代替弾力性の逆数)
        delta: float = 0.08,  # 固定資本減耗率
        psi: float = 0.5,  # 年金の平均所得代替率
        K0: float = 6.0,  # 初期資産
        tol: float = 1e-3,  # 収束判定の閾値
        maxiter: int = 2000,  # 最大繰り返し回数
        lambdaR: float = 0.2,  # 資本更新調整係数
        # jw: int = 20,      # 勤労期の初期 j work
        # jr: int = 45,      # 引退期の初期 j retire
    ) -> None:
        # パラメータを設定する
        self.NJ = NJ
        self.Njw = Njw
        # self.Njr     = Njr
        self.Nl = Nl
        self.l_dif = l_dif
        self.Na = Na
        self.Naprime = Naprime
        self.a_max = a_max
        self.a_min = a_min
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.psi = psi
        self.K0 = K0
        self.tol = tol
        self.maxiter = maxiter
        self.lambdaR = lambdaR

        # グリッドの設定
        h_grid = np.linspace(1, NJ, NJ)  # 年齢のグリッド
        l_grid = np.zeros(Nl)
        l_grid[0] = 1.0 - l_dif
        l_grid[1] = 1.0 + l_dif
        a_grid = maliar_grid(
            a_min, a_max, Na, 1.20
        )  # 今期資産グリッド（Maliarグリッド）
        aprime_grid = maliar_grid(
            a_min, a_max, Naprime, 1.20
        )  # 次期資産グリッド（Maliarグリッド）

        self.h_grid = h_grid
        self.l_grid = l_grid
        self.a_grid = a_grid
        self.aprime_grid = aprime_grid

        # 生産性の遷移確率行列Pの用意
        P_HH = 0.8
        P_LL = 0.8

        P = np.array([[P_HH, 1 - P_HH], [1 - P_LL, P_LL]])
        self.P = P

        # CRRA型効用関数と限界効用を定義する
        if self.gamma == 1:
            # 対数効用の場合
            self.utility: Callable[[float], float] = np.log
            self.mutility: Callable[[float], float] = njit(lambda x: 1 / x)
        else:
            # CRRA効用の場合（self.gammaを直接参照）
            self.utility = njit(lambda x: x ** (1 - self.gamma) / (1 - self.gamma))
            self.mutility = njit(lambda x: x ** (-self.gamma))
