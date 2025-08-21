# Todo

0. ~~solve_ss関数が正しい挙動をしているか確認。現在の設定で、K = 6.1675あたりになればOK~~
1. ~~olg_solverの完全な型定義~~
2. ~~非明示的な関数依存を解消。ちゃんとメソッドインジェクションする（solve_ssでちゃんと引数にhpを入れるなど）~~
3. ~~solve_ssを移行過程の途中で使用するのが便利なように、returnする変数を工夫する（resultクラスを定義するなど）~~
5. Settingクラスを移行過程にも使用できるようにする（olg_solverのSettingとは別に一般のSettingを作るべき？）
6. ~~移行過程の実装~~


# n年齢 j の人の消費補償変分

$c^E$ は比較対象となる経済における消費、$C^B$ はベースライン(初期定常状態)における消費を表す。

$$\mathbf{E} \sum_{i = j}^J \beta^{j-i} u(c_i^B (1 + \lambda)) = \mathbf{E} \sum_{i=j}^J \beta^{j-i} u(c_i^E)$$