import numpy as np
import matplotlib.pyplot as plt

from olg_solver import solve_ss, Setting

def main():
    # テスト用の軽量設定（グリッド数を少なくして高速実行）
    hp_test = Setting(
        Na=101,        # 今期の資産グリッド数を201→51に削減
        Naprime=1001  # 来期の資産グリッド数を8001→1001に削減
    )
    
    print("テスト実行中（軽量設定）...")
    K = solve_ss(hp_test)
    print(f"\\nMain: 最終的な資本ストック K = {K:.4f}")
    
    # 必要に応じて標準設定での実行
    # print("\\n標準設定での実行...")
    # K_standard = solve_ss()  # デフォルト設定を使用
    # print(f"Main: 標準設定での資本ストック K = {K_standard:.4f}")


if __name__ == "__main__":
    main()