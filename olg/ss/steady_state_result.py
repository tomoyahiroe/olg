from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

if TYPE_CHECKING:
    from .setting import Setting


@dataclass
class SteadyStateResult:
    """
    定常状態計算の結果を格納するクラス

    移行過程計算で再利用するための情報を含む
    """

    # 基本結果
    K: float  # 収束した資本ストック
    r: float  # 均衡利子率
    w: float  # 均衡賃金率
    p: float  # 均衡年金
    tau: float  # 均衡税率
    L: float  # 総労働供給

    # 関数・分布情報（移行過程で必要）
    value_fun_box: npt.NDArray[np.floating]  # 価値関数 (NJ, Nl, Na)
    policy_fun_box: npt.NDArray[np.floating]  # 政策関数（実数値） (NJ, Nl, Na)
    optaprime_index_box: npt.NDArray[np.int_]  # 政策関数（インデックス） (NJ, Nl, Na)
    mu_dist_box: npt.NDArray[np.floating]  # 人口分布 (NJ, Nl, Na)
    y_list: npt.NDArray[np.floating]  # 所得配列 (NJ, Nl)
    h_dist: npt.NDArray[np.floating]  # 年齢別人口分布 (NJ,)

    # 計算統計
    iterations: int  # 収束までの反復回数
    market_diff: float  # 最終市場差
    errm: float  # 最終人口合計誤差
    computation_time: float  # 計算時間（秒）

    # 設定情報
    hp: "Setting"  # 使用した設定インスタンス

    def print_summary(self) -> None:
        """計算結果のサマリーを表示"""
        print(f"\n=== 定常状態計算結果 ===")
        print(f"収束した資本ストック K = {self.K:.4f}")
        print(f"均衡利子率 r = {self.r:.4f}")
        print(f"均衡賃金率 w = {self.w:.4f}")
        print(f"均衡年金 p = {self.p:.4f}")
        print(f"均衡税率 τ = {self.tau:.4f}")
        print(f"総労働供給 L = {self.L:.4f}")
        print(f"")
        print(f"計算統計:")
        print(f"  反復回数: {self.iterations}")
        print(f"  最終市場差: {self.market_diff:.6e}")
        print(f"  最終人口合計誤差: {self.errm:.6e}")
        print(f"  計算時間: {self.computation_time:.2f}秒")
