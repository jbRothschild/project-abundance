import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns

DIR_INPUT = "input"
DIR_OUTPUT = "output"

for dirs in [DIR_INPUT, DIR_OUTPUT]:
    if not os.path.exists(dirs):
        os.makedirs(dirs)

# globals for heatmaps and equations (to prevent circular imports)
KON = 1.
KP = 10.
T = 1000.
KF = 1.0

#color_palette = plt.cm.get_cmap('Classic', 10)  # unnecessary unless you want different colours
#colour_palette = sns.color_palette("muted", 10)
colour_palette = sns.color_palette("muted",10)
sns.palplot(colour_palette)
#plt.show()

COLOR_SCHEME = {'c' : 'k',
                'koff' : colour_palette[3],
                'simple_fisher' : colour_palette[4],
                'numerical_fisher_sp' : colour_palette[9],
                'numerical_fisher' : colour_palette[2],
                'heuristic' : colour_palette[1]}
