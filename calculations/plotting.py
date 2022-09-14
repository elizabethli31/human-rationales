import sys
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as mt
import seaborn as sns
import numpy as np

sys.path.append('../')
from fidelity.fidelity import Fidelity


# Params from paper
sns.set(font_scale=3.0, rc={
    "lines.linewidth": 3,
    "lines.markersize":20,
    "ps.useafm": True,
    "axes.facecolor": 'white',
    "font.sans-serif": ["Helvetica"],
    "pdf.use14corefonts" : True,
    "text.usetex": False,
    })
sns.set(style="whitegrid")
LINEWIDTH = 3
MARKERSIZE = 10
TICKLABELSIZE = 14
LEGENDLABELSIZE = 14
LABELSIZE = 23
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 1.0

OUTPUT_DIR = '../plots/'

# Rationale percentages
# stats_data = pd.read_csv('../calculations/stats.csv')
# rationale_data = {"classes": [], "rationale percentage": []}
# classes = stats_data['classes'].tolist()
# rationale_perc = stats_data['mean_rationale_percent_classes'].apply(lambda c: c*100).tolist()
# rationale_data['classes'] = classes
# rationale_data['rationale percentage'] = rationale_perc
# rationale_df = pd.DataFrame(rationale_data)

# bar = sns.barplot(x='classes', y="rationale percentage", data=rationale_df, palette='colorblind')
# plt.savefig(OUTPUT_DIR+'rationale_percentage.png', bbox_inches = 'tight', dpi=300)

# Plot experiment
fidelity = Fidelity()
models = ['LogisticRegression', 'LSTM', 'RandomForest', 'RoBERTa']
# plot_data = {'model': [],
#              'accuracy': [],
#              'sufficiency': [],
#              'comprehensiveness': [],
#              'normalized_comprehensiveness': [],
#              'normalized_sufficiency': []
#             }

# for model in models:
#     feature_df = pd.read_csv('feature_' + model + '.csv')

#     y_hat = feature_df["predicted_classes"].to_numpy()
#     y = feature_df["true_classes"].to_numpy()
#     prob_y_hat = feature_df["prob_y_hat"].to_numpy()
#     prob_y_hat_alpha = feature_df["prob_y_hat_alpha"].to_numpy()
#     prob_y_hat_alpha_comp = feature_df["prob_y_hat_alpha_comp"].to_numpy()
#     null_difference = feature_df["null_diff"].to_numpy()

#     sufficiency = fidelity.compute(prob_y_hat=prob_y_hat,
#                                              prob_y_hat_alpha=prob_y_hat_alpha,
#                                              normalization=False)
        
#     comprehensiveness = fidelity.compute(prob_y_hat=prob_y_hat,
#                                                     prob_y_hat_alpha=prob_y_hat_alpha_comp,
#                                                     fidelity_type="comprehensiveness",
#                                                     normalization=False)


#     normalized_sufficiency = fidelity.compute(prob_y_hat=prob_y_hat,
#                                              prob_y_hat_alpha=prob_y_hat_alpha,
#                                              normalization=True, null_difference=null_difference)

#     normalized_comprehensiveness = fidelity.compute(prob_y_hat=prob_y_hat,
#                                                     prob_y_hat_alpha=prob_y_hat_alpha_comp,
#                                                     fidelity_type="comprehensiveness",
#                                                     normalization=True, null_difference=null_difference)

#     plot_data['model'].append(model)
#     plot_data['accuracy'].append(mt.accuracy_score(y, y_hat))
#     plot_data['sufficiency'].append(sufficiency)
#     plot_data['comprehensiveness'].append(comprehensiveness)
#     plot_data['normalized_sufficiency'].append(normalized_sufficiency)
#     plot_data['normalized_comprehensiveness'].append(normalized_comprehensiveness)

# plot_data_df = pd.DataFrame(plot_data)

# plt.figure()
# accuracy_bar = sns.barplot(x='model', y='accuracy', data=plot_data_df, palette='colorblind')
# plt.ylim(0, 1)
# plt.savefig(OUTPUT_DIR + 'accuracy.png')

# plt.figure()
# suff_bar = sns.barplot(x='model', y='sufficiency', data=plot_data_df, palette='colorblind')
# plt.savefig(OUTPUT_DIR + 'suff.png')

# plt.figure()
# comp_bar = sns.barplot(x='model', y='comprehensiveness', data=plot_data_df, palette='colorblind')
# plt.savefig(OUTPUT_DIR + 'comp.png')

# By Class
by_class_data = {'model': [],
                 'class': [],
                 'accuracy': [],
                 'sufficiency': [],
                 'comprehensiveness': [],
                 'normalized_comprehensiveness': [],
                 'normalized_sufficiency': [],
                 'null_diff': []
                }
for model in models:
    class_df = pd.read_csv('feature_' + model + '.csv')

    for id in [0, 1]:
        df = class_df[class_df['true_classes']==id]
        y_hat = df['predicted_classes'].to_numpy()
        y = df['true_classes'].to_numpy()
        prob_y_hat = df['prob_y_hat'].to_numpy()
        prob_y_hat_alpha = df['prob_y_hat_alpha'].to_numpy()
        prob_y_hat_alpha_comp = df['prob_y_hat_alpha_comp'].to_numpy()
        null_difference = df['null_diff'].to_numpy()

        sufficiency = fidelity.compute(prob_y_hat=prob_y_hat,
                                       prob_y_hat_alpha=prob_y_hat_alpha,
                                       normalization=False)

        comprehensiveness = fidelity.compute(prob_y_hat=prob_y_hat,
                                             prob_y_hat_alpha=prob_y_hat_alpha_comp,
                                             fidelity_type="comprehensiveness",
                                             normalization=False)
        
        normalized_sufficiency = fidelity.compute(prob_y_hat=prob_y_hat,
                                                  prob_y_hat_alpha=prob_y_hat_alpha,
                                                  null_difference=null_difference,
                                                  normalization=True)

        normalized_comprehensiveness = fidelity.compute(prob_y_hat=prob_y_hat,
                                                    prob_y_hat_alpha=prob_y_hat_alpha_comp,
                                                    fidelity_type="comprehensiveness",
                                                    null_difference=null_difference,
                                                    normalization=True)

        by_class_data['model'].append(model)
        by_class_data['accuracy'].append(mt.accuracy_score(y, y_hat))
        by_class_data['class'].append('class' + str(id))
        by_class_data['sufficiency'].append(sufficiency)
        by_class_data['comprehensiveness'].append(comprehensiveness)
        by_class_data['normalized_comprehensiveness'].append(normalized_comprehensiveness)
        by_class_data['normalized_sufficiency'].append(normalized_sufficiency)
        by_class_data['null_diff'].append(np.mean(null_difference))

class_data_df = pd.DataFrame(by_class_data)

plt.figure()
acc_class = sns.barplot(x='model', y='accuracy', hue='class', data=class_data_df, palette='colorblind')
plt.savefig(OUTPUT_DIR + 'accuracy_class.png')

plt.figure()
suff_class = sns.barplot(x='model', y='sufficiency', hue='class', data=class_data_df, palette='colorblind')
plt.savefig(OUTPUT_DIR + 'suff_class.png')

plt.figure()
comp_class = sns.barplot(x='model', y='comprehensiveness', hue='class', data=class_data_df, palette='colorblind')
plt.savefig(OUTPUT_DIR + 'comp_class.png')

plt.figure()
acc_bar_class = sns.barplot(x='model', y='accuracy', hue='class', data=class_data_df, palette='colorblind')
plt.savefig(OUTPUT_DIR + 'accuracy_class.png')

plt.figure()
norm_comp = sns.barplot(x='model', y='normalized_comprehensiveness', hue='class', data=class_data_df, palette='colorblind')
plt.savefig(OUTPUT_DIR + 'norm_comp.png')

plt.figure()
suff_comp = sns.barplot(x='model', y='normalized_sufficiency', hue='class', data=class_data_df, palette='colorblind')
plt.savefig(OUTPUT_DIR + 'norm_suff.png')

plt.figure()
null_class = sns.barplot(x='model', y='null_diff', hue='class', data=class_data_df, palette='colorblind')
plt.savefig(OUTPUT_DIR + 'null_diff.png')



