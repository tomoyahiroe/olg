from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from .setting import Setting


def plot_asset_path(
    hp: 'Setting',
    mu_dist_box: npt.NDArray[np.floating],
    policy_fun_box: npt.NDArray[np.floating]
) -> None:
    """
    Plot asset paths for high and low productivity agents (consumption-normalized version)
    
    Parameters
    ----------
    hp : Setting
        OLGモデル設定インスタンス
    mu_dist_box : npt.NDArray[np.floating]
        人口分布配列 (NJ, Nl, Na)
    policy_fun_box : npt.NDArray[np.floating]
        政策関数配列（実数値） (NJ, Nl, Na)
    """
    # First, recalculate necessary economic variables
    h_dist = np.ones(hp.NJ) / hp.NJ
    tau = hp.psi * np.sum(h_dist[hp.Njw:]) / np.sum(h_dist[:hp.Njw])
    L = np.sum(h_dist[:hp.Njw])
    
    # Calculate current capital stock (equilibrium state)
    K = 0.0
    for h in range(hp.NJ):
        for i_l in range(hp.Nl):
            for i_a in range(hp.Na):
                mu = mu_dist_box[h, i_l, i_a]
                if mu > 0:
                    K += policy_fun_box[h, i_l, i_a] * mu
    
    # Calculate factor prices
    r = hp.alpha * (K / L) ** (hp.alpha - 1) - hp.delta
    w = (1 - hp.alpha) * (K / L) ** hp.alpha
    
    # Calculate income
    yvec = np.empty((hp.NJ, hp.Nl))
    for h in range(hp.NJ):
        for i_l in range(hp.Nl):
            if h < hp.Njw:  # Working period
                yvec[h, i_l] = (1 - tau) * w * hp.l_grid[i_l]
            else:  # Retirement period
                yvec[h, i_l] = hp.psi * w
    
    # Calculate consumption
    cons = np.zeros((hp.NJ, hp.Nl, hp.Na))
    for h in range(hp.NJ):
        for i_l in range(hp.Nl):
            for i_a in range(hp.Na):
                cons[h, i_l, i_a] = yvec[h, i_l] + (1 + r) * hp.a_grid[i_a] - policy_fun_box[h, i_l, i_a]
    
    # Calculate average consumption by age
    cfunJ = np.zeros(hp.NJ)
    for h in range(hp.NJ):
        total_mass = np.sum(mu_dist_box[h])
        if total_mass > 0:
            cfunJ[h] = np.sum(cons[h] * mu_dist_box[h]) / total_mass
    
    # Normalization coefficient (normalize by average consumption at first age)
    norm = 1.0 / cfunJ[0] if cfunJ[0] > 0 else 1.0
    
    # Calculate average assets by age and skill
    avg_assets_high = np.zeros(hp.NJ)  # Average assets for high productivity
    avg_assets_low = np.zeros(hp.NJ)   # Average assets for low productivity
    
    for h in range(hp.NJ):
        # High productivity (i_l = 1) average assets
        total_mass_high = np.sum(mu_dist_box[h, 1, :])
        if total_mass_high > 0:
            avg_assets_high[h] = np.sum(policy_fun_box[h, 1, :] * mu_dist_box[h, 1, :]) / total_mass_high
        
        # Low productivity (i_l = 0) average assets  
        total_mass_low = np.sum(mu_dist_box[h, 0, :])
        if total_mass_low > 0:
            avg_assets_low[h] = np.sum(policy_fun_box[h, 0, :] * mu_dist_box[h, 0, :]) / total_mass_low
    
    # Plot (normalized)
    ages = np.arange(hp.NJ) + 20  # Start from age 20
    
    plt.figure(figsize=(12, 8))
    plt.plot(ages, norm * avg_assets_high, 'k-', linewidth=3, label=f'High productivity (l={hp.l_grid[1]:.1f})')
    plt.plot(ages, norm * avg_assets_low, 'k-.', linewidth=3, label=f'Low productivity (l={hp.l_grid[0]:.1f})')
    
    # Display retirement age as vertical line
    retirement_age = 20 + hp.Njw
    plt.axvline(x=retirement_age, color='gray', linestyle='--', alpha=0.7, label=f'Retirement age: {retirement_age}')
    
    plt.xlabel('Age')
    plt.ylabel('Assets (normalized by consumption=1)')
    plt.title('OLG Model: Asset Path Evolution by Productivity (Consumption Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(20, 20 + hp.NJ - 1)
    plt.ylim(0, None)
    
    # Display additional information as text
    info_text = f"Parameters:\nα={hp.alpha}, β={hp.beta}, γ={hp.gamma}\nδ={hp.delta}, ψ={hp.psi}\nNorm. coeff.: {norm:.3f}"
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()