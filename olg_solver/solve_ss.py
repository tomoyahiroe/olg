import numpy as np
import time
from .setting import Setting
from .household_solver import solve_household_backward
from .distribution_updater import update_distribution
from .asset_supply import calculate_asset_supply
from .plot_asset_path import plot_asset_path


def solve_ss():
    """
    メイン関数
    ここでは、OLGモデルの設定を初期化し、必要な計算を行います。
    
    Returns:
    float: 収束した資本ストック K
    """
    # numba最適化されたメインループ（hpインスタンス使用版）
    #hp = Setting(NJ = 10, Njw = 7, Na = 50, Naprime = 1000)
    hp = Setting()

    # 繰り返しに入る前の準備
    # 年齢×スキル×資産,この三次元のようなイメージの箱を作る
    value_fun_box   = np.zeros((hp.NJ, hp.Nl, hp.Na))              # 価値関数の箱
    optaprime_index_box = np.zeros((hp.NJ, hp.Nl, hp.Na), dtype=int)   # 政策関数の箱（インデックス）
    policy_fun_box  = np.zeros((hp.NJ, hp.Nl, hp.Na))              # 政策関数の箱（実数値）

    # 2. 労働所得グリッド（初期化）
    y_matrix        = np.zeros((hp.NJ, hp.Nl))

    # 3. 人口分布（初期化）
    mu_dist_box     = np.zeros((hp.NJ, hp.Nl, hp.Na))         # 年齢×スキル×資産 の人口分布

    # 4. 年齢別人口分布（死も人口成長もない → 一様分布）
    h_dist          = np.ones(hp.NJ) / hp.NJ

    # 5. 税率（年金支払いのための均衡税率）
    tau = hp.psi * np.sum(h_dist[hp.Njw:]) / np.sum(h_dist[:hp.Njw])

    # 6. 総労働供給
    L = np.sum(h_dist[:hp.Njw]) # 生産性を入れる？、今回の平均生産性は1だから気持ちは1で割っている,tauchenでやる場合には調整が必要

    # 初期化
    market_diff = 1.0
    errm = 1.0
    diff_iteration = 0
    K = hp.K0

    print("numba最適化されたOLGモデルを実行中...")
    start_time = time.time()

    while (market_diff > hp.tol or errm > hp.tol) and diff_iteration < hp.maxiter:
        diff_iteration += 1

        # 所与のKからr,w,pを計算する
        r = hp.alpha * (K / L) ** (hp.alpha - 1) - hp.delta
        w = (1 - hp.alpha) * (K / L) ** hp.alpha
        p = hp.psi * w

        # 所得関数（リスト）を設定する
        y_list = y_matrix.copy()  # (年齢, スキル)
        for h in range(hp.NJ):
            for i_l in range(hp.Nl):
                if h < hp.Njw:  # 労働期間
                    y_list[h, i_l] = (1 - tau) * w * hp.l_grid[i_l] # tau = 0.214286
                else:  # 引退期間
                    y_list[h, i_l] = p

        # hpインスタンスを使用した家計の最適化問題を後ろ向きに解く
        solve_household_backward(hp, value_fun_box, policy_fun_box, optaprime_index_box, y_list, r)

        # hpインスタンスを使用した分布更新
        update_distribution(hp, mu_dist_box, policy_fun_box, h_dist)

        # 人口合計が1か確認
        total_mass = np.sum(mu_dist_box)
        errm = np.abs(total_mass - 1.0)

        # hpインスタンスを使用した資産供給計算
        A_supply = calculate_asset_supply(hp, mu_dist_box, policy_fun_box)

        # 家計側資産合計との差を計算
        market_diff = np.abs(K - A_supply)

        # 資本の更新
        K = K + hp.lambdaR * (A_supply - K)

        # 進捗表示
        if diff_iteration % 5 == 0 or diff_iteration <= 5:
            print(f"Iteration {diff_iteration}: market_diff = {market_diff:.6e}, errm = {errm:.6e}")

    end_time = time.time()
    print(f"\\n計算完了! 実行時間: {end_time - start_time:.2f}秒")
    print(f"総イテレーション数: {diff_iteration}")
    print(f"最終市場差: {market_diff:.6e}")
    print(f"最終人口合計誤差: {errm:.6e}")
    print(f"収束した資本ストック K = {K:.4f}")
    
    # 資産パスの推移をプロット
    plot_asset_path(hp, mu_dist_box, policy_fun_box)
    
    # 収束した資本ストックを返す
    return K