# 教科書Juliaコードをもとにした準コード

1. パラメータを設定する
    - $\alpha = 0.40$
    - $\beta = 0.98$
    - $\delta = 0.08$
    - $N_T=100$：（移行期間）
    - $N_J=61$：（20歳から80歳まで生きる）
    - $N_{JW}=45$：（20歳から64歳まで働く）
    - $N_a=101$：今期の資産グリッドの数
    - $N_{a'}=2001$：来季の資産グリッドの数
    - $N_l=2$：スキルは{low, high}の2種類
    - $a_{min}=0$：資産グリッドの最小値
    - $a_{max}=25$：資産グリッドの最大値
    - $curvA=1.2$：資産グリッド（maliarグリッド）生成のための係数
    - 効用関数の設定：$u(\cdot) = \ln(\cdot)$
    - 偶然の死は存在しないので、最終期に全ての資産を使い切る
2. グリッドの設定
    - 年齢のグリッド $(h=1,\cdots,N_{JW},\cdots ,N_J)$
    - 労働生産性グリッド $(i_l=1,\cdots,N_l)$
        - $dif=0.2$
        
        $$
        (l_{low},l_{high})=(1-dif,1+dif)
        $$
        
    - 今期資産グリッド $(i_a=1,\cdots,N_a)$ （maliarグリッド）
        
        $$
        a_{i_a} =  a_{min} + \left( \frac{i_a-1}{N-1} \right)^{curvA}(a_{max} - a_{min}) 
        $$
        
    - 次期資産グリッド$(j_{a}=1,\cdots,N_{a'})$（maliarグリッド）
        
        $$
        a_{j_a} =  a_{min} + \left( \frac{j_a-1}{N-1} \right)^{curvA}(a_{max} - a_{min}) 
        $$
        
3. 生産性の遷移確率行列$P$の用意（$N_l\times N_l$行列）
    
    $$
    P=\begin{bmatrix}
       p_{ll} & p_{lh} \\
       p_{hl} & p_{hh}
    \end{bmatrix}=\begin{bmatrix}
       0.8 & 0.2 \\
       0.2 & 0.8
    \end{bmatrix}
    $$
    
4. 反復アルゴリズムに関するパラメータ設定
- 定常状態計算
    - 市場クリア誤差許容度`tol` $\epsilon = 0.001$
    - 最大繰返し回数`maxiter = 2000`
    - 資本更新調整係数`adjK` 0.2
- 移行過程計算
    - 市場クリア誤差許容度`errKTol` $\epsilon = 0.0001$
    - 最大繰返し回数`maxiterTR = 300`
    - 資本更新調整係数`adjK` 0.05
1. 人口分布を設定
    
    ```julia
    # age distribution (assume no death and no pop growth. sum to 1)
    meaJ = 1/Nj .* ones(Nj);
    
    # aggregate labor supply
    L = sum(meaJ[1:Njw]);
    ```
    
    - （アーカイブ）年齢分布
        
        死亡、人口成長はない。労働生産性の遷移確率を用いて、反復計算から生産性の定常状態を導出する。年齢の異質性はないため $(\forall h,\theta_h=1)$ 、ここでは簡易的にスキルの異質性のみ考慮してコードを実行する。
        
        ```python
        import numpy as np
        l_grid = np.array([0.8, 1.2])
        muZ_old = np.ones(len(l_grid)) / len(l_grid) # 初期分布
        
        tol=1E-11
        maxit=100^4
        Pz = [[0.8,0.2],[0.2,0.8]]
        
        for it in range(maxit):
          muZ_new = muZ_old @ Pz
          if np.max(np.abs(muZ_new - muZ_old)) < tol: break
          muZ_old = muZ_new
        
        # 退職後も働くと仮定した場合の総労働供給を求める
        Lbar = sum(muZ_new*l_grid)
        print(muZ_new)
        print(Lbar)
        
        # [0.5 0.5]
        # 1.0
        ```
        
        したがって退職者が働いた場合を含めた総労働供給は１であり、年齢の異質性はないため、各年齢での労働供給量は $1/N_{J}$ である。
        
        $$
        \sum^{N_{J}}_{h=1}\sum^{N_l}_{i_l=1}l\sum^{N_a}_{i_a=1 }\theta_h\mu_{t,h,i_a,i_l}=1\\
        \quad  \sum^{N_l}_{i_l=1}l\sum^{N_a}_{i_a=1 }\theta_h\mu_{t,h,i_a,i_l}=\frac{1}{N_J}
        $$
        
        （コード上のmeaJとははこれのことか？ $\mu_{t,h,i_a,i_l}$ は確率変数であるため、合計して１する必要もあったと考えられる？）
        
    - 総労働 $\left\lbrace L_t \right\rbrace^T_{t=1}$ : 上記の労働供給量を労働年齢分足し上げる。
        
        $$
        
        L^s_t=\sum^{N_{JW}}_{h=1}\sum^{N_l}_{i_l=1}l\sum^{N_a}_{i_a=1}\theta_h\mu_{t,h,i_a,i_l}=\frac{N_{JW}}{N_J}
        $$
        
2. 次期資産（細かい）→今期資産（粗い）への逆補間のためのweightを計算
- $a'_{\_,\ j_a,\ \_}$を内分点とする連続する$(a_{\_,\ i_a,\ \_},a_{\_,\ i_{a+1},\ \_})$を探し、$(a_{ac1vec_{j_a}},a_{ac2vec_{j_a}})$とする
    
    $$
    a'_{\_,\ j_a,\ \_}= weight_{j_a,1} a_{ac1vec_{j_a}}+ weight_{j_a,2} a_{ac2vec_{j_a}}
    $$
    
    つまり、上記のような$(a_{ac1vec_{j_a}},a_{ac2vec_{j_a}})$を探す（$weight_{j_a,1},weight_{j_a,2}$は重み）
    

# １．初期・最終定常状態導出

[教科書Juliaコード準コード](https://www.notion.so/Julia-22f32ebc8099800d8302edee75d46433?pvs=21) 

移行前後の2つの定常状態を求める。

1. 初期定常状態：所得代替率 $\psi_{ini}=0.5$
    
    分布を $\mu^{ini}_{h,i_a,i_l}$ 、総資本を $K_{ini}$ とする。
    
2. 最終定常状態：所得代替率 $\psi_{fin}=0.25$ 
    
    価値関数を $V'^{fin}_{h,i_a,i_l}$、総資本を $K_{fin}$ とする。
    

# ２．移行過程

### ⅰ. 初期設定

1. 移行過程における 所得代替率の初期値を設定 
    
    ```julia
    # path of SS replacement rate
    rhoT = zeros(NT);
    TT = 25; # coverged to rho1 in TT years
    for tc in 1:TT
        rhoT[tc] = rho0 + ((rho1-rho0)/(TT-1))*(tc-1);
    end
    rhoT[TT+1:NT] .= rho1;
    
    # path of tax rate
    tauT = zeros(NT);
    for tc in 1:NT
        tauT[tc] = rhoT[tc]*sum(meaJ[Njw+1:Nj])/sum(meaJ[1:Njw]);
    end
    tau0 = copy(tauT[1]);
    tau1 = copy(tauT[NT]);
    ```
    
- 所得代替率 $\left\lbrace \psi_t \right\rbrace^T_{t=1}$
    
    1期目の0.5から25期目の0.25まで線形に減少し、以降、最終期まで0.25で一定
    
    $$
    ⁍
    $$
    
- 年金保険料率 $\left\lbrace \tau_t \right\rbrace^T_{t=1}$
    
    $$
    \begin{aligned}
    \left\lbrace \psi_t \right\rbrace^T_{t=1}に基づき\tau_t &= \psi_t \frac{\sum^{N_{J}}_{h=N_{JW}+1}\sum^{N_l}_{i_l=1}l\sum^{N_a}_{i_a=1 }\theta_j\mu_{t,h,i_a,i_l}}{\sum^{N_{JW}}_{h=1}\sum^{N_l}_{i_l=1}l\sum^{N_a}_{i_a=1 }\theta_j\mu_{t,h,i_a,i_l}}\\
    &=\psi_t \frac{\sum^{N_{J}}_{h=N_{JW}+1}\frac{1}{N_J}}{\sum^{N_{JW}}_{h=1}\frac{1}{N_J}}\\
    &=\psi_t\frac{N_J-N_{JW}}{N_{JW}}
    \end{aligned}
    $$
    
1. 移行過程における行き（後ろ向き方向）での総資本を当て推量
    - 総資本 $\left\lbrace K_t \right\rbrace^T_{t=1}$
    ただし $K_1=K_{ini},\ K_T=K_{fin},\ K_t=K_{ini}+(t-1)\cdot\frac{K_{fin}-K_{ini}}{T-1}$
        - 教科書のコードでは総資本の遷移パスを単純な線形補間ではなく、30期間かけて $*K_{ini}*K_{ini}$$K_{fin}$ に線形増加し、以降は $K_t=K_{fin}$ を取るという当て推量をしている。
            
            $$
            K_t=K_{ini}+(t-1) \cdot\frac{K_{fin}-K_{ini}}{29} \quad(t\leq30)\\
            
            K_t=K_{fin} \quad(t\geq31)
            $$
            
2. 移行過程の政策関数とそのインデックスの箱を用意。 期間×年齢×スキル×資本
    
    ```julia
     # policy function (initialization)
     afunGT = zeros(Int64,NT,Nj,Ne,Na);
    
     # asset (level): (ADDED IN JULIA CODE)
     afunT = zeros(NT,Nj,Ne,Na);
    ```
    
    $$
    \text{opt\_index}_{t,h,i_a,i_l}\\
    a'_{t,h,i_a,i_l}
    $$
    

## ⅱ. 次のステップ(a,b,c)を収束するまで繰り返す。（繰り返し記号： $n$ ）

### a. T→1 期（後ろ向き）

最終的に政策関数とそのインデックスの経路 $\{a'^{[n]}_{t,h,i_a,i_l}\}^T_{t=1},\{\text{opt\_index}^{[n]}_{t,h,i_a,i_l}\}^T_{t=1}$ を導出する。その過程において価値関数 $V^{t,[n]}_{h,i_a,i_l}$ を使う。

1. $t=T$ 期
    1. 最終定常状態の価値関数を来期の価値関数として $(V^{T+1,[n]}_{h,i_a,i_l}=V'^{fin}_{h,i_a,i_l})$ 、 $T$ 期の価値関数 $V^{T,[n]}_{h,i_a,i_l}$ と 政策関数 $a'^{[n]}_{t=T,h,i_a,i_l}$ 、そしてその政策関数のインデックス $\text{opt\_index}^{[n]}_{t=T,h,i_a,i_l}$を導出する。下のグラフでは視認性のため価値関数の添え字 $i_a,i_l$ を省略する。
        
        ![diagram-20250807 (8).png](attachment:9c23ee76-a403-43a2-8cea-8d039cab8ec4:diagram-20250807_(8).png)
        
        1. 所与の$K^{[n]}_T$から、$r^{[n]}_T,w^{[n]}_T,p^{[n]}_T$を計算する
            - $r^{[n]}_T = \alpha\left(\frac{K^{[n]}_T}{L}\right)^{\alpha-1}- \delta$
            - $w^{[n]}_T = (1-\alpha)\left(\frac{K^{[n]}_T}{L}\right)^{\alpha}$
            - $p^{[n]}_T = \psi_T w^{[n]}_T$
        2. 所得関数を設定する。$y^{T,[n]}_{{h,i_l}}$
            
            $$
            y^{T,[n]}_{{h,i_l}} = 
            \begin{dcases}
            \begin{aligned}
            &(1-\tau)w^{[n]}_Tl_{i_l} \ &&h=1,2,\cdots,N_{JW}
            \\
            &p^{[n]}_T \ \  &&h=N_{JW}+1,\cdots,N_{J}
            \end{aligned}
            \end{dcases}
            $$
            
            実際にはここで${h,i_l}$ごとに繰り返して計算してlistに格納しておく
            
        3. プログラミング簡易化のため、この期においてのみ使用する価値関数、政策関数の箱を用意。 （年齢×スキル×資産）
            
            ```julia
            # initialization
            vfun1 = zeros(Nj,Ne,Na); # new value function (at time tc)
            afunG = zeros(Int64,Nj,Ne,Na); # solution grid
            afun = zeros(Nj,Ne,Na);  # solution level
            ```
            
            $$
            V^{t,[n]}_{h,i_a,i_l}\\
            \text{opt\_index}^{t,[n]}_{h,i_a,i_l}\\
            a'^{\ t,[n]}_{h,i_a,i_l}
            $$
            
    
    1. 家計の最適化問題を後ろ向きに解く。
        1. 最終年齢（年齢$N_J$）の政策関数、価値関数を求める。
            1. **繰り返し**（$i_l=1,\cdots,N_l$）
                1. **繰り返し**（$i_a=1,\cdots,N_a$）
                    1. 最終期の資産は０なので、政策関数$a'^{\ T ,[n]}_{h=N_J,i_a,i_l}=0$
                    2. 予算制約式より$c^{T,[n]}_{h=N_J,i_a,i_l} = y^{T,[n]}_{{h=N_J,i_l}}+(1+r)a_{{\_,\ i_a,\ \_ }}$なので、価値関数は$V^{T,[n]}_{h=N_J,i_a,i_l}= u\left(y^{T,[n]}_{{N_J,i_l}}+(1+r)a_{{\_,\ i_a,\ \_ }}\right)$
        2. **繰り返し**$（h= N_{J-1},\cdots,1歳）$${h}$ごとの政策関数、価値関数を求める
            1. **繰り返し$（i_l= 1,\cdots,N_l）$**
                1. 所得設定関数から所得を取り出す$y=y^{T,[n]}_{{h,i_l}}$ 
                2. **繰り返し**$（i_a= 1,\cdots,N_a）$
                    1. 消費がマイナスになった場合にペナルティが発生するように、価値関数$vtemp_{\_,\ \  i_a,\ \ \_ }$の初期値を大きなマイナスの値にしておく
                        - この部分は本来は必要ないかもしれないが、教科書のコードではやっているので、念の為書いておく
                    2. サーチする対象となるグリッドの最大値を$accmax$とし、$accmax=N_{a'}$とする
                        1. **繰り返し$（j_a= 1,\cdots,N_{a'}）$**価値関数を最大にする$j_a$をグリッドサーチで特定
                            1. 当期消費の計算
                            
                            $$
                            c = y +(1+r)a_{\_,\ \  i_a,\ \ \_ } - a'_{\_,\ \  j_a,\ \ \_ }
                            $$
                            
                            1. 消費$c$がマイナスになった場合、$accmax=j_{a}-1$として、$j_a$に関するループをやめる
                                - ここで消費がマイナスになる次期資産のグリッドはサーチ対象から外してしまうので、本来は価値関数にペナルティをつける必要はないはず、、、
                            2. 今期の価値関数を計算：最終定常状態の価値関数を来期の価値関数として使う
                                
                                $$
                                \begin{aligned}
                                vpr 
                                
                                &=\sum_{j_l=1}^{2}p_{i_l,j_l}\left[weight_{j_a,1}V_{h+1,ac1vec_{j_a},j_l}^{T+1,[n]}+weight_{j_a,2}V_{h+1,ac2vec_{j_a},j_l}^{T+1[n]}\right]\\
                                \end{aligned}
                                $$
                                
                                $$
                                vtemp_{j_a}^{[n]}  = u(c)+ \beta \times vpr
                                
                                $$
                                
                    3. $vtemp^{[n]}_{1},\cdots,vtemp^{[n]}_{accmax}$の中から最大の値を探し、その最大値を$V^{T,[n]}_{h,i_a,i_l}$に格納する。その最大を取ったインデックス$j_a^*$を$\text{opt\_index}^{T,[n]}_{h,i_a,i_l}$に格納する。値$a'^{\ T,[n]}_{j_a^*}$を、$a'^{\ T,[n]}_{h,i_a,i_l}$に格納する。
            2. $\text{opt\_index}^{T,[n]}_{h,i_a,i_l}$を $\text{opt\_index}^{[n]}_{t=T,h,i_a,i_l}$に格納する。$a'^{\ T,[n]}_{h,i_a,i_l}$を $a'^{[n]}_{t=T,h,i_a,i_l}$に格納する。

1. 繰り返し$（t= T-1,T-2,\cdots,1期）$
    
    直前で求めた１つ先の価値関数を来期の価値関数として $(V^{t+1,[n]}_{h,i_a,i_l})$ 、 $t$ 期の価値関数 $V^{t,[n]}_{h,i_a,i_l}$ と 政策関数 $a'^{[n]}_{t=t,h,i_a,i_l}$ 、そしてその政策関数のインデックス $\text{opt\_index}^{[n]}_{t=t,h,i_a,i_l}$を導出する。下のグラフでは視認性のため価値関数の添え字 $i_a,i_l$ を省略する。
    
    ![diagram-20250807 (7).png](attachment:631e0d7e-17fe-4d21-8ced-190d56e58c50:diagram-20250807_(7).png)
    
    1. 所与の$K^{[n]}_t$から、$r^{[n]}_t,w^{[n]}_t,p^{[n]}_t$を計算する
        - $r^{[n]}_t = \alpha\left(\frac{K^{[n]}_t}{L}\right)^{\alpha-1}- \delta$
        - $w^{[n]}_t = (1-\alpha)\left(\frac{K^{[n]}_t}{L}\right)^{\alpha}$
        - $p^{[n]} = \psi_t w^{[n]}_t$
    2. 所得関数を設定する。$y^{t,[n]}_{{h,i_l}}$
        
        $$
        y^{t,[n]}_{{h,i_l}} = 
        \begin{dcases}
        \begin{aligned}
        &(1-\tau)w^{[n]}_tl_{i_l} \ &&h=1,2,\cdots,N_{JW}
        \\
        &p^{[n]}_t \ \  &&h=N_{JW}+1,\cdots,N_{J}
        \end{aligned}
        \end{dcases}
        $$
        
        実際にはここで${h,i_l}$ごとに繰り返して計算してlistに格納しておく
        
    3. プログラミング簡易化のため、この期においてのみ使用する価値関数、政策関数の箱を用意。 （年齢×スキル×資産）
        
        ```julia
        # initialization
        vfun1 = zeros(Nj,Ne,Na); # new value function (at time tc)
        afunG = zeros(Int64,Nj,Ne,Na); # solution grid
        afun = zeros(Nj,Ne,Na);  # solution level
        ```
        
        $$
        V^{t,[n]}_{h,i_a,i_l}\\
        \text{opt\_index}^{t,[n]}_{h,i_a,i_l}\\
        a'^{\ t,[n]}_{h,i_a,i_l}
        $$
        
    4. 家計の最適化問題を後ろ向きに解く。
        1. 最終年齢（年齢$N_J$）の政策関数、価値関数を求める。
            1. **繰り返し**（$i_l=1,\cdots,N_l$）
                1. 最終期の資産は０なので、政策関数$a'^{\ t,[n]}_{h=N_J,i_a,i_l}=0$
                2. 予算制約式より$c^{[n]}_{h=N_J,i_a,i_l} = y^{t,[n]}_{{h=N_J,i_l}}+(1+r)a_{{\_,\ i_a,\ \_}}$なので、価値関数は$V^{t,[n]}_{h=N_J,i_a,i_l}= u\left(y^{t,[n]}_{{N_J,i_l}}+(1+r)a_{{\_,\ i_a,\ \_}}\right)$
        2. **繰り返し**$（h= N_{J-1},\cdots,1歳）$${h}$ごとの政策関数、価値関数を求める
            1. **繰り返し$（i_l= 1,\cdots,N_l）$**
                1. 所得設定関数から所得を取り出す$y=y^{t,[n]}_{{h,i_l}}$ 
                2. **繰り返し**$（i_a= 1,\cdots,N_a）$
                    1. 消費がマイナスになった場合にペナルティが発生するように、価値関数$vtemp_{\_,\ i_a,\ \_}$の初期値を大きなマイナスの値にしておく
                        - この部分は本来は必要ないかもしれないが、教科書のコードではやっているので、念の為書いておく
                    2. サーチする対象となるグリッドの最大値を$accmax$とし、$accmax=N_{a’}$とする
                        1. **繰り返し$（j_a= 1,\cdots,N_{a'}）$**効用最大にする$j_a$をグリッドサーチで特定
                            1. 当期消費の計算
                            
                            $$
                            c = y +(1+r)a_{\_,\ i_a,\ \_} - a'_{\_,\ j_a,\ \_} 
                            $$
                            
                            1. 消費$c$がマイナスになった場合、$accmax=j_{a}-1$として、$j_a$に関するループをやめる
                                - ここで消費がマイナスになる次期資産のグリッドはサーチ対象から外してしまうので、本来は価値関数にペナルティをつける必要はないはず、、、
                            2. 今期の価値関数を計算：１期前で求めた価値関数を来期の価値関数として使う
                                
                                $$
                                \begin{aligned}
                                vpr 
                                
                                &=\sum_{j_l=1}^{2}p_{i_l,j_l}\left[weight_{j_a,1}V_{h+1,ac1vec_{j_a},j_l}^{t+1,[n]}+weight_{j_a,2}V_{h+1,ac2vec_{j_a},j_l}^{t+1[n]}\right]\\
                                \end{aligned}
                                $$
                                
                                $$
                                vtemp_{j_a}^{[n]}  = u(c^{[n]})+ \beta  \times vpr 
                                
                                $$
                                
                    3. $vtemp^{[n]}_{1},\cdots,vtemp^{[n]}_{accmax}$の中から最大の値を探し、その最大値を$V^{t,[n]}_{h,i_a,i_l}$に格納する。その最大を取ったインデックス$j_a^*$を$\text{opt\_index}_{h,i_a,i_l}^{t,[n]}$に格納する。値$a'^{\ t,[n]}_{j_a^*}$を、$a'^{\ t,[n]}_{h,i_a,i_l}$に格納する。
        3. $\text{opt\_index}^{t,[n]}_{h,i_a,i_l}$ を $\text{opt\_index}^{[n]}_{t=t,h,i_a,i_l}$ に格納する。 $a'^{\ t,[n]}_{h,i_a,i_l}$ を $a'^{[n]}_{t=t,h,i_a,i_l}$ に格納する。

以上より $\{a'^{[n]}_{t,h,i_a,i_l}\}^T_{t=1},\{\text{opt\_index}^{[n]}_{t,h,i_a,i_l}\}^T_{t=1}$ を得る。

### b. 1→T 期（前向き）

最終的に状態分布の経路 $\{\mu_{t,h,i_a,i_l}^{[n]}\}^T_{t=2}$を導出する。

1. 移行過程の状態分布を入れるための箱を用意。 期間×年齢×スキル×資本
    
    $$
    \mu^{[n]}_{t,h,i_a,i_l}
    $$
    
    ただし計算過程においては年齢×スキル×資本の箱を用いる。
    
    $$
    \mu^{t,[n]}_{h,i_a,i_l}
    $$
    
    ```julia
    meaT = zeros(NT,Nj,Ne,Na) # initialization
    mea0 = zeros(Nj,Ne,Na);   # initialization
            
    
    meaT[1,:,:,:] .= copy(mea_SS0) # dist in the initial SS
    
    mea0 .= copy(mea_SS0);
    ```
    
2. $t=1$ 期
    
    初期定常状態の状態分布を今期の状態分布 $\mu_{h,i_{a},i_l}^{t=1,[n]}=\mu^{ini}_{h,i_a,i_l}$とする。「a. T→1 期（後ろ向き）」で求めた政策関数インデックス $\text{opt\_index}^{[n]}_{t=1,h,i_a,i_l}$ を所与として、来期の状態分布 $\mu_{t=2,h,i_{opt1},j_l}^{[n]}$ を求める。
    
    1. プログラミング簡易化のため、この期においてのみ使用する政策関数インデックスを取り出す。（年齢×スキル×資産） 
        
        $$
        \text{opt\_index}^{t=1,[n]}_{h,i_a,i_l}=\text{opt\_index}^{[n]}_{t=1,h,i_a,i_l}
        $$
        
    2. $0$歳の家計については、全員が資産$*0*$、スキルはhighとlow、50％ずつ均等に存在する。
        
        $$
        \mu^{t=2,[n]}_{h=0,i_a=0,i_l=low}=\mu^{t=2,[n]}_{h=0,i_a=0,i_l=high}=0.5\times \frac{1}{N_j}
        $$
        
    3. 繰り返し$（h= 1,\cdots,N_{J-1}歳）$1歳ずつ前向きに分布を更新していく。
        1. 繰り返し$（i_l= 1,\cdots,N_l）$
            1. 繰り返し$（i_a= 1,\cdots,N_a）$ （定常状態計算ではここに繰り返し記号）
                1. 現在の状態が$(h,a_{i_{a}},l_{i_{l}})$であれば、次期の資産として$\text{opt\_index}^{t=1,[n]}_{h,i_a,i_l}$番目の次期資産が選ばれるので、
                    - これに対応する今期の資産のグリッド（粗いグリッド）の番号は、$ac1vec_{{\text{opt\_index}_{t=1,h,i_a,i_l}}},ac2vec_{{\text{opt\_index}_{t=1,h,i_a,i_l}}}$になるので、これらを$i_{opt1},i_{opt2}$とする
                    - これに対応するweightは、$weight_{{\text{opt\_index}_{t=1,h,i_a,i_l},1}},weight_{{\text{opt\_index}_{t=1,h,i_a,i_l},2}}$ になる
                2. スキルの遷移確率を用いて分布を更新する
                    1. 繰り返し$(j_l=1,\cdots,N_l)$ 
                    
                    $$
                    \mu_{h+1,i_{opt1},j_l}^{t=2,[n],new}=\mu_{h+1,i_{opt1},j_l}^{t=2,[n],old}+\mu^{t=1,[n]}_{h,i_a,i_l}\times p_{i_l,j_l}\times weight_{{\text{opt\_index}_{h,i_a,i_l},1}},
                    \\ \mu_{h+1,i_{opt2},j_l}^{t=2,[n],new}=\mu_{h+1,i_{opt2},j_l}^{t=2,[n],old}+\mu^{t=1,[n]}_{h,i_a,i_l}\times p_{i_l,j_l}\times weight_{{\text{opt\_index}_{h,i_a,i_l},2}},
                    $$
                    
            2. これで求まる $\mu_{h+1,i_a,i_l}^{t=2,[n]}$ を $\mu_{t=2,h+1,i_a,i_l}^{[n]}$ に格納する。
3. 繰り返し$(t=2,3,\cdots,T-1期)$
    
    １期前で求めた状態分布を今期の状態分布 $\mu^{t,[n]}_{h,i_a,i_l}$ とする。「a. T→1 期（後ろ向き）」で求めた政策関数インデックス $\text{opt\_index}^{[n]}_{t=t,h,i_a,i_l}$ を所与として、来期の状態分布 $\mu_{t=t+1,h,i_{a},i_l}^{[n]}$ を求める。
    
    1. プログラミング簡易化のため、この期においてのみ使用する政策関数インデックスを取り出す。 （年齢×スキル×資産）
        
        $$
        \text{opt\_index}^{t,[n]}_{h,i_a,i_l}=\text{opt\_index}^{[n]}_{t,h,i_a,i_l}
        $$
        
    2. $0$歳の家計については、全員が資産$*0*$、スキルはhighとlow、50％ずつ均等に存在すると仮定する。
        
        $$
        \mu^{t+1,[n]}_{h=0,i_a=0,i_l=low}=\mu^{t+1,[n]}_{h=0,i_a=0,i_l=high}=0.5\times \frac{1}{N_j}
        $$
        
    3. 繰り返し$（h= 1,\cdots,N_{J-1}）$1期ずつ前向きに分布を更新していく。
        1. 繰り返し$（i_l= 1,\cdots,N_l）$
            1. 繰り返し$（i_a= 1,\cdots,N_a）$
                1. 現在の状態が$(h,a_{i_{a}},l_{i_{l}})$であれば、次期の資産として$\text{opt\_index}^{t,[n]}_{t,h,i_a,i_l}$番目の次期資産が選ばれるので、
                    - これに対応する今期の資産のグリッド（粗いグリッド）の番号は、$ac1vec_{{\text{opt\_index}_{t,h,i_a,i_l}}},ac2vec_{{\text{opt\_index}_{t,h,i_a,i_l}}}$になるので、これらを$i_{opt1},i_{opt2}$とする
                    - これに対応するweightは、$weight_{{\text{opt\_index}_{t,h,i_a,i_l},1}},weight_{{\text{opt\_index}_{t,h,i_a,i_l},2}}$ になる
                2. スキルの遷移確率を用いて分布を更新する
                    1. 繰り返し$(j_l=1,\cdots,N_l)$
                    
                    $$
                    \mu_{h+1,i_{opt1},j_l}^{t+1,[n],new}=\mu_{h+1,i_{opt1},j_l}^{t+1,[n],old}+\mu_{h,i_a,i_l}^{t,[n]}\times p_{i_l,j_l}\times weight_{{\text{opt\_index}_{h,i_a,i_l},1}},
                    \\ \mu_{h+1,i_{opt2},j_l}^{t+1,[n],new}=\mu_{h+1,i_{opt2},j_l}^{t+1,[n],old}+\mu_{h,i_a,i_l}^{t,[n]}\times p_{i_l,j_l}\times weight_{{\text{opt\_index}_{h,i_a,i_l},2}},
                    $$
                    
        2. これで求まる $\mu_{h+1,i_{a},i_l}^{t+1,[n]}$ を $\mu_{t+1,h+1,i_{a},i_l}^{[n]}$ に格納する。
4. 以上より $\{\mu_{t,h,i_a,i_l}^{[n]}\}^T_{t=2}$ を得る。

### c. 分布更新後のチェック

```julia
# ============= #
#  COMPUTE KT1  #
# ============= # 

 errKT = zeros(NT);
 KT1[1] = copy.(KT0[1]); # predetermined
 errKT[1] = 0.0;
            

  for tc in 1:NT-1

    afun .= copy(afunT[tc,:,:,:]); # saving for the next period
    mea0 .= copy(meaT[tc,:,:,:]);

    KT1[tc+1] = sum(mea0.*afun); # saving at the begginig of next period

    errKT[tc+1] = abs(KT1[tc]-KT0[tc]);

end

            errK = maximum(errKT);
            errKvec[iterTR] = errK;

            # update guess KT0
            if errK > errKTol

                # KT0[1] is predetermined
                for tc in 2:NT
                    KT0[tc] += adjK*(KT1[tc]-KT0[tc]);
                end

```

1. 資産の合計を計算する
    
    $$
    A^{[n]}_t = \sum_{h=1}^{N_J} \sum_{i_a=1}^{N_a}\sum_{i_l=1}^{N_l}  a_{\_,\ i_a,\ \_} \cdot \mu_{t,h,i_a,i_l}^{[n]}
    
    $$
    
2. 資本市場均衡のチェック
    
    $$
    \max_{t=1,\dotsc,T}||K^{[n]}_t-A^{[n]}_t||<\epsilon
    $$
    
3. 総資本 $K_t$  を収束していない場合、資本ストックを更新する

$$
K_t^{[n+1]}=K^{[n]}_t+adjK \cdot(A^{[n]}_t-K^{[n]}_t)
$$