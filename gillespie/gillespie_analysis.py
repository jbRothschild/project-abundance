import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

"""
def plot_pop(time_points ,pops, n_simulations):
    # Set up subplots
    fig, ax = plt.subplots(1, pops.shape[2], figsize=(14, 5))

    # Plot state trajectories
    for i in range(n_simulations):
        ax.plot(time_points, pops[i,:,0], '-', lw=0.3, alpha=0.2, color=sns.color_palette()[0])
    # Plot state mean
    ax.plot(time_points, pops[:,:,0].mean(axis=0), '-', lw=6, color=sns.color_palette()[2])
    # Label axes
    ax.set_xlabel('dimensionless time')
    ax.set_ylabel('probability of being found in state')

    plt.tight_layout()
    plt.show()
"""
"""
MODELS = {'original' : {'first_mte_dist' :  first_mte_dist, 'capitan_dist' : capitan_dist, 'distribution': distribution }

}
"""
