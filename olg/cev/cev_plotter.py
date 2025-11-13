"""
CEV Plotter Module

消費補償変分（Consumption Equivalent Variation）のプロット機能
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
from typing import Any, Optional


def plot_cev_by_age(
    cev_avg: npt.NDArray,
    period: int = 0,
    initial_result: Optional[Any] = None,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    年齢別CEVをプロット

    Parameters
    ----------
    cev_avg : np.ndarray
        CEV平均 (NT, NJ, Nl)
    period : int, default=0
        プロットする期間（0ベース）
    initial_result : SteadyStateResult, optional
        初期定常状態の結果
    figsize : tuple, default=(10, 6)
        図のサイズ
    save_path : str, optional
        保存パス（指定した場合は保存）
    """
    # 年齢軸の設定
    if initial_result is not None:
        ages = np.arange(20, 20 + initial_result.hp.NJ)
        l_grid = initial_result.hp.l_grid
    else:
        # デフォルト値
        NJ = cev_avg.shape[1]
        ages = np.arange(20, 20 + NJ)
        l_grid = np.array([0.7, 1.3])  # デフォルト生産性

    plt.figure(figsize=figsize)

    # 標準CEVをプロット
    plt.plot(
        ages,
        cev_avg[period, :, 0],
        "--k",
        linewidth=2,
        label=f"Low skill (l={l_grid[0]:.1f})",
    )
    plt.plot(
        ages,
        cev_avg[period, :, 1],
        "-k",
        linewidth=2,
        label=f"High skill (l={l_grid[1]:.1f})",
    )

    # ゼロ線
    plt.axhline(y=0, color="gray", linestyle=":", alpha=0.7)

    plt.xlabel("Age")
    plt.ylabel("CEV")

    plt.title(f"CEV by Age: Period {period + 1}")

    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存または表示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"図を保存しました: {save_path}")
    else:
        plt.show()


def plot_cev_by_cohort(
    cev_avg: npt.NDArray,
    age: int = 0,
    initial_result: Optional[Any] = None,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    コホート別CEVをプロット（特定年齢での時系列）

    Parameters
    ----------
    cev_avg : np.ndarray
        CEV平均 (NT, NJ, Nl)
    age : int, default=0
        プロットする年齢インデックス（0=20歳）
    initial_result : SteadyStateResult, optional
        初期定常状態の結果
    figsize : tuple, default=(10, 6)
        図のサイズ
    save_path : str, optional
        保存パス（指定した場合は保存）
    """
    # 生産性グリッドの設定
    if initial_result is not None:
        l_grid = initial_result.hp.l_grid
    else:
        l_grid = np.array([0.7, 1.3])  # デフォルト生産性

    plt.figure(figsize=figsize)

    # コホート（期間）軸
    cohorts = np.arange(1, len(cev_avg) + 1)

    plt.plot(
        cohorts,
        cev_avg[:, age, 1],
        "-k",
        linewidth=2,
        label=f"High skill (l={l_grid[1]:.1f})",
    )
    plt.plot(
        cohorts,
        cev_avg[:, age, 0],
        "--k",
        linewidth=2,
        label=f"Low skill (l={l_grid[0]:.1f})",
    )

    # ゼロ線
    plt.axhline(y=0, color="gray", linestyle=":", alpha=0.7)

    plt.xlabel("Cohort")
    plt.ylabel("CEV")
    plt.title(f"CEV by Cohort: Age {20 + age}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存または表示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"図を保存しました: {save_path}")
    else:
        plt.show()
