# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.20.0
#   kernelspec:
#     display_name: curateit
#     language: python
#     name: python3
# ---

# %%
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
import numpy as np
from scipy.cluster.hierarchy import leaves_list
import matplotlib as mpl
import dendropy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from typing import List


def kl_divergence(p, q, x_range):
    epsilon = 1e-10
    q = np.maximum(q, epsilon)

    mask = p > epsilon
    kl_integrand = np.zeros_like(p)
    kl_integrand[mask] = p[mask] * np.log2(p[mask] / q[mask])

    return np.trapezoid(kl_integrand, x_range)


def jensen_shannon_distance(p, q, x_range):
    m = 0.5 * (p + q)

    return np.sqrt((kl_divergence(p, m, x_range) + kl_divergence(q, m, x_range)) / 2)



def write_cat_itol_dataset_symbol(
    filename: str,
    ids: List[str],
    data: List[str],
    dataset_name: str,
    symbol: int = 2,
    size: int = 1,
    fill: float = 1.0,
    position: float = 1.0,
    legend_title: str = "Legend",
):
    """
    Create an ITOL annotation file for visualizing a tree with colours.
    """

    colour_range = "bright"

    categories = set(data)
    colour_map = {category: idx for idx, category in enumerate(categories)}
    palette = sns.color_palette(colour_range, len(categories)).as_hex()
    colours = []
    for d in data:
        cat_idx = colour_map[d]
        hex_code = palette[cat_idx]
        colours.append(hex_code)

    symbols = [symbol] * len(data)
    sizes = [size] * len(data)
    fills = [fill] * len(data)
    positions = [position] * len(data)

    legend_shapes = ["2" for x in range(len(categories))]
    legend_colours = [palette[colour_map[x]] for x in categories]
    legend_labels = [x for x in categories]
    legend_scales = ["1" for _ in range(len(categories))]
    legend_invert = ["0" for _ in range(len(categories))]

    with open(filename, "w") as file:
        print("DATASET_SYMBOL", file=file)
        print("SEPARATOR COMMA", file=file)
        print(f"DATASET_LABEL,{dataset_name}", file=file)
        print("COLOR,#ffff00", file=file)
        print(f"LEGEND_TITLE,{legend_title}", file=file)
        print(f"LEGEND_SHAPES,{','.join(legend_shapes)}", file=file)
        print(f"LEGEND_COLORS,{','.join(legend_colours)}", file=file)
        print(f"LEGEND_LABELS,{','.join(legend_labels)}", file=file)
        print(f"LEGEND_SHAPE_SCALES,{','.join(legend_scales)}", file=file)
        print(f"LEGEND_SHAPE_INVERT,{','.join(legend_invert)}", file=file)
        print("DATA", file=file)
        print("#NODE_ID, SYMBOL, SIZE, COLOR, FILL, POSITION, LABEL", file=file)

        for id, sym, size, colour, fill, pos, label in zip(
            ids, symbols, sizes, colours, fills, positions, data
        ):
            file.write(f"{id},{sym},{size},{colour},{fill},{pos},{label}\n")



# %% [markdown]
# # ADK 

# %% [markdown]
# ### Compute the JS distance between samples

# %%
master_adk_data = pd.read_csv("/Users/uqsporra/git_repos/TreeGazer_paper/data/adk/dataset_adk.tsv", sep='\t')
feature = "km"
datasets = [f"../results/adk/unprocessed/log10_{feature}_tg_tied_30_trials.csv", 
            f"../results/adk/unprocessed/log10_{feature}_knn_30_trials.csv", 
            f"../results/adk/unprocessed/log10_{feature}_wde_30_trials.csv", 
            f"../results/adk/unprocessed/log10_{feature}_gp_30_trials.csv"
            ]

outputs = [f"../results/adk/log10_{feature}_tg_tied_30_trials.csv", 
            f"../results/adk/log10_{feature}_knn_30_trials.csv", 
            f"../results/adk/log10_{feature}_wde_30_trials.csv", 
            f"../results/adk/log10_{feature}_gp_30_trials.csv"]

for dataset, output in zip(datasets, outputs):
    tree_ucb = pd.read_csv(dataset)
    if "Unnamed: 0" in tree_ucb.columns:
        tree_ucb.drop(columns=["Unnamed: 0"], inplace=True)
    trials = tree_ucb["trial"].unique()
    master_adk_data = master_adk_data.loc[:, ["org_name", "log10_kcatkm"]]


    actual_kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(master_adk_data["log10_kcatkm"].to_numpy()[:, np.newaxis])
    recalculated = []
    for trial in trials: 
        tree_ucb_subset = tree_ucb[tree_ucb["trial"] == trial].copy()
        start_training_data = master_adk_data[~master_adk_data["org_name"].isin(tree_ucb_subset["candidate_id"])]

        js_results = []
        for i in range(len(tree_ucb_subset)): 
            train_subset_data = master_adk_data[master_adk_data["org_name"].isin(tree_ucb_subset.iloc[0:i]["candidate_id"])]
            accumulated_train_data = pd.concat([start_training_data, train_subset_data])

            pred_kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
                accumulated_train_data["log10_kcatkm"].to_numpy()[:, np.newaxis]
            )

            x_range = np.linspace(
            master_adk_data["log10_kcatkm"].to_numpy()[:, np.newaxis].min() - 0.01,
                master_adk_data["log10_kcatkm"].to_numpy()[:, np.newaxis].max() + 0.01,
                num=600,
            )

            # compute the log-likelihood of each sample
            pred_log_density = pred_kde.score_samples(x_range[:, np.newaxis])
            actual_log_density = actual_kde.score_samples(x_range[:, np.newaxis])

            p = np.exp(actual_log_density)
            q = np.exp(pred_log_density)

            js_dist = jensen_shannon_distance(p, q, x_range)
            js_results.append(js_dist)
        
        tree_ucb_subset["js_dist"] = js_results
        recalculated.append(tree_ucb_subset)

    final = pd.concat(recalculated)
    final.to_csv(output, index=False) 


# %% [markdown]
# #### Plot the intermediate KDE distributions for the ADK dataset

# %%
master_adk_data = pd.read_csv("/Users/uqsporra/git_repos/TreeGazer_paper/data/adk/dataset_adk.tsv", sep='\t')
feature = "kcatkm"
outputs = [f"../results/adk/log10_{feature}_tg_tied_30_trials.csv", 
            f"../results/adk/log10_{feature}_knn_30_trials.csv", 
            f"../results/adk/log10_{feature}_wde_30_trials.csv", 
            f"../results/adk/log10_{feature}_gp_30_trials.csv"]
labels = ["TreeGazer", "KNN", "WDE", "GP"]
for output, label in zip(outputs, labels):
    
    tree_ucb = pd.read_csv(output)

    trials = tree_ucb["trial"].unique()
    master_adk_data = master_adk_data.loc[:, ["org_name", f"log10_{feature}"]]


    actual_kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(master_adk_data[f"log10_{feature}"].to_numpy()[:, np.newaxis])

    tree_ucb_subset = tree_ucb[tree_ucb["trial"] == 0].copy()
    start_training_data = master_adk_data[~master_adk_data["org_name"].isin(tree_ucb_subset["candidate_id"])]

    step = 40
    for i in range(0, len(tree_ucb_subset), step): 
        train_subset_data = master_adk_data[master_adk_data["org_name"].isin(tree_ucb_subset.iloc[0:i]["candidate_id"])]
        accumulated_train_data = pd.concat([start_training_data, train_subset_data])

        pred_kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
            accumulated_train_data[f"log10_{feature}"].to_numpy()[:, np.newaxis]
        )

        x_range = np.linspace(
        master_adk_data[f"log10_{feature}"].to_numpy()[:, np.newaxis].min() - 0.01,
            master_adk_data[f"log10_{feature}"].to_numpy()[:, np.newaxis].max() + 0.01,
            num=600,
        )

        # compute the log-likelihood of each sample
        pred_log_density = pred_kde.score_samples(x_range[:, np.newaxis])
        actual_log_density = actual_kde.score_samples(x_range[:, np.newaxis])

        p = np.exp(actual_log_density)
        q = np.exp(pred_log_density)

        plt.plot(x_range, q, label=f"{i} samples", alpha=0.5)# 10 * count/fracs)

    plt.plot(x_range, p, label="Actual ADK distribution", color="black", linestyle="--")
    plt.title(f"{label} KDE of actual ADK distribution for {feature}")
    plt.xlabel(f"ADK {feature}")
    plt.ylabel("Density")
    plt.grid()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(f"../results/adk/log10_{feature}_{label}_kde.svg", bbox_inches='tight', format="svg")
    plt.show()

# %% [markdown]
# ### Plot the Jensen-Shannon distance over the 30 trials for ADK

# %%

features = ["kcat_km", "km", "kcat"]
titles = ["$\log_{{10}} \mathit{{k_{cat}/K_{m}}}$", "$\log_{{10}} \mathit{{K_{m}}}$", "$\log_{{10}} \mathit{{k_{cat}}}$"]

for feature, title in zip(features, titles):

    knn = pd.read_csv(f"../results/adk/{feature}/log10_{feature}_knn_30_trials.csv")
    knn["algorithm"] = ["KNN (ProtT5 Embeddings)"] * len(knn)

    tree_ucb = pd.read_csv(f"../results/adk/{feature}/log10_{feature}_tg_tied_30_trials.csv")
    tree_ucb["algorithm"] = ["TreeGazer UCB"] * len(tree_ucb)

    wde = pd.read_csv(f"../results/adk/{feature}/log10_{feature}_wde_30_trials.csv")
    wde["algorithm"] = ["Weighted distance estimate"] * len(wde)

    gp_ucb = pd.read_csv(f"../results/adk/{feature}/log10_{feature}_gp_30_trials.csv")
    gp_ucb["algorithm"] = ["GP UCB (ProtT5 Embeddings)"] * len(gp_ucb)

    all_data = pd.concat([knn, tree_ucb, wde, gp_ucb])

    fig = plt.subplots(1,1)
    sns.lineplot(all_data, x=all_data["iterations"] + 34, y='js_dist', hue='algorithm')
    plt.ylabel("Jensen-Shannon Distance")
    plt.xlabel("Training dataset size")
    plt.xlim(35, 175)
    plt.grid()
    plt.title(title)
    #plt.savefig(f"../results/adk/{feature}/log10_{feature}_js_distance.svg", bbox_inches='tight', format="svg")
    plt.plot()





# %% [markdown]
# ### Sequence sampling pattern for the ADK dataset

# %%

master_adk_data = pd.read_csv("/Users/uqsporra/git_repos/TreeGazer_paper/data/adk/dataset_adk.tsv", sep='\t')
feature = "kcat_km"
outputs = [f"../results/adk/{feature}/log10_{feature}_tg_tied_30_trials.csv", 
            f"../results/adk/{feature}/log10_{feature}_knn_30_trials.csv", 
            f"../results/adk/{feature}/log10_{feature}_wde_30_trials.csv", 
            f"../results/adk/{feature}/log10_{feature}_gp_30_trials.csv"]

labels = [ "GP UCB (ProtT5 Embeddings)", "KNN (ProtT5 Embeddings)", "TreeGazer UCB", "Weighted distance estimate",]

tree = dendropy.Tree.get_from_path('../data/adk/adk_ancestors.nwk', schema='newick')

dist_map = np.zeros((len(tree.taxon_namespace), len(tree.taxon_namespace)))
pdc = tree.phylogenetic_distance_matrix()
for i, t1 in enumerate(tree.taxon_namespace[:-1]):
    for j, t2 in enumerate(tree.taxon_namespace[i+1:], i+1):
        d = pdc(t1, t2)
        dist_map[i, j] = d
        dist_map[j, i] = d

dist_df = pd.DataFrame(dist_map, index=["_".join(x.split()) for x in tree.taxon_namespace.labels()], columns=["_".join(x.split()) for x in tree.taxon_namespace.labels()])
condensed = squareform(dist_map)
Z = linkage(condensed, method="complete")
order = leaves_list(Z)
df_reordered = dist_df.iloc[order, order]


cmap = mpl.colors.ListedColormap(["white", "darkblue"])
norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)


for output, label in zip(outputs, labels):
    tree_ucb = pd.read_csv(output)
    tree_ucb_subset = tree_ucb[tree_ucb["trial"] == 0].copy()
    start_training_data = master_adk_data[~master_adk_data["org_name"].isin(tree_ucb_subset["candidate_id"])]

    sampled_map = pd.DataFrame(0, index=df_reordered.index, columns=df_reordered.columns)

    step = 40
    fig, axes = plt.subplots(1, len(range(step, len(tree_ucb_subset), step)) + 1, figsize=(20, 5))
    sns.heatmap(df_reordered, cmap='viridis', ax=axes[0], xticklabels=False, yticklabels=False)
    axes[0].set_title("Phylogenetic distance matrix")
    axes[0].set_ylabel("Taxa")

    
    count = 1
    for i in range(step, len(tree_ucb_subset), step): 
    
        train_subset_data = master_adk_data[master_adk_data["org_name"].isin(tree_ucb_subset.iloc[0:i]["candidate_id"])]
        accumulated_train_data = pd.concat([start_training_data, train_subset_data])
        sampled_map.loc[accumulated_train_data["org_name"], :] = 1
        sns.heatmap(sampled_map, cmap=cmap, norm=norm, cbar=False, ax=axes[count])
        axes[count].set_title(f"Iteration {i}") 
        axes[count].set_ylabel("Taxa")
        axes[count].set_xticks([])
        axes[count].set_yticks([])
        count += 1

    title = f"{feature} sampling pattern - {label}"
    plt.suptitle(title)


    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Unsampled (0)", "Sampled (1)"])

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    #plt.savefig(f"../results/adk/log10_{feature}_{label}_sampling_pattern.png", bbox_inches='tight', dpi=300)
    plt.show()

# %% [markdown]
# ## Lanmodulin

# %% [markdown]
# ### Calculate Jensen-Shannon distance for each iteration

# %%
lan_master_data = pd.read_csv("/Users/sporras/git_repos/TreeGazer_paper/data/lanmodulin/tg_lanmodulin_data.csv")


datasets = [f"/Users/uqsporra/git_repos/TreeGazer_paper/results/lanmodulin/La_gp_30_trials.csv", 
            f"/Users/uqsporra/git_repos/TreeGazer_paper/results/lanmodulin/La_knn_9_trials.csv",
            f"/Users/uqsporra/git_repos/TreeGazer_paper/results/lanmodulin/La_tg_tied_30_trials.csv", 
            f"/Users/uqsporra/git_repos/TreeGazer_paper/results/lanmodulin/La_wde_30_trials.csv",
            ]

outputs = [f"/Users/uqsporra/git_repos/TreeGazer_paper/results/lanmodulin/La_gp_30_trials_test.csv",
           f"/Users/uqsporra/git_repos/TreeGazer_paper/results/lanmodulin/La_knn_9_trials_test.csv",
           f"/Users/uqsporra/git_repos/TreeGazer_paper/results/lanmodulin/La_tg_tied_30_trials_test.csv",
           f"/Users/uqsporra/git_repos/TreeGazer_paper/results/lanmodulin/La_wde_30_trials_test.csv"]

datasets = [
            f"/Users/sporras/git_repos/TreeGazer_paper/results/lanmodulin/La_knn_30_trials.csv",
            
            ]

outputs = [
           f"/Users/sporras/git_repos/TreeGazer_paper/results/lanmodulin/La_knn_30_trials_test.csv",
           ]

for dataset, output in zip(datasets, outputs):
    tree_ucb = pd.read_csv(dataset)
    if "Unnamed: 0" in tree_ucb.columns:
        tree_ucb.drop(columns=["Unnamed: 0"], inplace=True)
    
    trials = tree_ucb["trial"].unique()
    lan_master_data = lan_master_data.loc[:, ["org_name", "La"]]


    actual_kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(lan_master_data["La"].to_numpy()[:, np.newaxis])
    recalculated = []
    for trial in trials: 
        tree_ucb_subset = tree_ucb[tree_ucb["trial"] == trial].copy()
        start_training_data = lan_master_data[~lan_master_data["org_name"].isin(tree_ucb_subset["candidate_id"])]

        js_results = []
        for i in range(len(tree_ucb_subset)): 
            train_subset_data = lan_master_data[lan_master_data["org_name"].isin(tree_ucb_subset.iloc[0:i]["candidate_id"])]
            accumulated_train_data = pd.concat([start_training_data, train_subset_data])

            pred_kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
                accumulated_train_data["La"].to_numpy()[:, np.newaxis]
            )

            x_range = np.linspace(
            lan_master_data["La"].to_numpy()[:, np.newaxis].min() - 0.01,
                lan_master_data["La"].to_numpy()[:, np.newaxis].max() + 0.01,
                num=600,
            )

            # compute the log-likelihood of each sample
            pred_log_density = pred_kde.score_samples(x_range[:, np.newaxis])
            actual_log_density = actual_kde.score_samples(x_range[:, np.newaxis])

            p = np.exp(actual_log_density)
            q = np.exp(pred_log_density)

            js_dist = jensen_shannon_distance(p, q, x_range)
            js_results.append(js_dist)
        
        tree_ucb_subset["js_dist"] = js_results
        recalculated.append(tree_ucb_subset)

    final = pd.concat(recalculated)
    final.to_csv(output, index=False)

# %% [markdown]
# ### Compare JS distance as training data is added 

# %%

starting_number_samples = 123 
knn = pd.read_csv("../results/lanmodulin/La_knn_30_trials_test.csv")
knn["algorithm"] = ["KNN (ProtT5 Embeddings)"] * len(knn)

tree_ucb = pd.read_csv("../results/lanmodulin/La_tg_tied_30_trials_test.csv")
tree_ucb["algorithm"] = ["TreeGazer UCB"] * len(tree_ucb)

wde = pd.read_csv("../results/lanmodulin/La_wde_30_trials_test.csv")
wde["algorithm"] = ["Weighted distance estimate"] * len(wde)

gp_ucb = pd.read_csv("../results/lanmodulin/La_gp_30_trials_test.csv")
gp_ucb["algorithm"] = ["GP UCB (ProtT5 Embeddings)"] * len(gp_ucb)

all_data = pd.concat([knn, tree_ucb, wde, gp_ucb])

fig = plt.subplots(1,1)
start_dataset_size = 123
sns.lineplot(all_data, x=all_data["iterations"] + starting_number_samples, y='js_dist', hue='algorithm')
plt.ylabel("Jensen-Shannon Distance")
plt.xlabel("Training dataset size")
plt.grid()
plt.title(f"Lanmodulin ")
plt.tight_layout()
plt.xlim(123, 650)
# place the legend outside the figure 
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
#plt.title("Lanmodulin selectivity with normalised log distribution coefficient")
plt.savefig(f"../results/lanmodulin/log10_La_js_distance.svg", bbox_inches='tight', dpi=300, format="svg")

plt.plot()



# %% [markdown]
# #### Plot approximated distributions 

# %%
lan_master_data = pd.read_csv("../data/lanmodulin/tg_lanmodulin_data.csv")
lan_master_data = lan_master_data.loc[:, ["org_name", "La"]]
actual_kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(lan_master_data["La"].to_numpy()[:, np.newaxis])

outputs = [f"/Users/sporras/git_repos/TreeGazer_paper/results/lanmodulin/La_gp_30_trials.csv",
           f"/Users/sporras/git_repos/TreeGazer_paper/results/lanmodulin/La_knn_9_trials.csv",
           f"/Users/sporras/git_repos/TreeGazer_paper/results/lanmodulin/La_tg_tied_30_trials.csv",
           f"/Users/sporras/git_repos/TreeGazer_paper/results/lanmodulin/La_wde_30_trials.csv"]

labels = [ "GP UCB (ProtT5 Embeddings)", "KNN (ProtT5 Embeddings)", "TreeGazer UCB", "Weighted distance estimate",]


for output, label in zip(outputs, labels):
    tree_ucb = pd.read_csv(output)
    tree_ucb_subset = tree_ucb[tree_ucb["trial"] == 0].copy()
    start_training_data = lan_master_data[~lan_master_data["org_name"].isin(tree_ucb_subset["candidate_id"])]

    step = 100
    for i in range(0, len(tree_ucb_subset), step): 
        
        train_subset_data = lan_master_data[lan_master_data["org_name"].isin(tree_ucb_subset.iloc[0:i]["candidate_id"])]
        accumulated_train_data = pd.concat([start_training_data, train_subset_data])

        pred_kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
            accumulated_train_data["La"].to_numpy()[:, np.newaxis]
        )

        x_range = np.linspace(
        accumulated_train_data["La"].to_numpy()[:, np.newaxis].min() - 0.01,
            accumulated_train_data["La"].to_numpy()[:, np.newaxis].max() + 0.01,
            num=600,
        )

            # compute the log-likelihood of each sample
        pred_log_density = pred_kde.score_samples(x_range[:, np.newaxis])
        actual_log_density = actual_kde.score_samples(x_range[:, np.newaxis])

        p = np.exp(actual_log_density)
        q = np.exp(pred_log_density)
        plt.plot(x_range, q, label=f"{i} samples", alpha=0.5)# 10 * count/fracs)


    plt.plot(x_range, p, label="Actual La distribution", color="black", linestyle="--")
    plt.title(f"KDE of actual La distribution - {label}")
    plt.xlabel("La selectivity (log distribution coefficient)")
    plt.ylabel("Density")
    plt.grid()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(f"../results/lanmodulin/log10_La_{label}_kde.png", bbox_inches='tight', dpi=300)
    plt.show()


# %% [markdown]
# #### Create ITOL visualisations 

# %%
lan_master_data = pd.read_csv("../data/lanmodulin/tg_lanmodulin_data.csv")
lan_master_data = lan_master_data.loc[:, ["org_name", "La"]]
actual_kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(lan_master_data["La"].to_numpy()[:, np.newaxis])

outputs = [f"../results/lanmodulin/La_gp_30_trials_test.csv",
           f"../results/lanmodulin/La_knn_30_trials_test.csv",
           f"../results/lanmodulin/La_tg_tied_30_trials_test.csv",
           f"../results/lanmodulin/La_wde_30_trials_test.csv"]

labels = [ "GP UCB (ProtT5 Embeddings)", "KNN (ProtT5 Embeddings)", "TreeGazer UCB", "Weighted distance estimate",]


for output, label in zip(outputs, labels):
    tree_ucb = pd.read_csv(output)
    tree_ucb_subset = tree_ucb[tree_ucb["trial"] == 0].copy()
    start_training_data = lan_master_data[~lan_master_data["org_name"].isin(tree_ucb_subset["candidate_id"])]

    step = 100
    for i in range(0, len(tree_ucb_subset), step): 
        
        train_subset_data = lan_master_data[lan_master_data["org_name"].isin(tree_ucb_subset.iloc[0:i]["candidate_id"])]
        accumulated_train_data = pd.concat([start_training_data, train_subset_data])

        write_cat_itol_dataset_symbol(filename=f"lanmodulin_{label}_iteration_{i}.itol",
                                         ids=accumulated_train_data["org_name"].tolist(), data=["sampled"] * len(accumulated_train_data),
                                         dataset_name=f"{label}_iteration_{i}",
                                         legend_title=f"{label}_iteration_{i}",)


# %%

tree = dendropy.Tree.get_from_path('/Users/uqsporra/git_repos/TreeGazer_paper/data/lanmodulin/lanmodulin.aln.treefile', schema='newick')

dist_map = np.zeros((len(tree.taxon_namespace), len(tree.taxon_namespace)))
pdc = tree.phylogenetic_distance_matrix()
for i, t1 in enumerate(tree.taxon_namespace[:-1]):
    for j, t2 in enumerate(tree.taxon_namespace[i+1:], i+1):
        d = pdc(t1, t2)
        dist_map[i, j] = d
        dist_map[j, i] = d

dist_df = pd.DataFrame(dist_map, index=["_".join(x.split()) for x in tree.taxon_namespace.labels()], columns=["_".join(x.split()) for x in tree.taxon_namespace.labels()])
dist_df

# %%
condensed = squareform(dist_map)
Z = linkage(condensed, method="complete")

g = sns.clustermap(dist_df, row_linkage=Z, col_linkage=Z, cmap='viridis')
reordered_rows = g.dendrogram_row.reordered_ind
reordered_cols = g.dendrogram_col.reordered_ind
# Create the new sorted DataFrame
df_sorted = dist_df.iloc[reordered_rows, reordered_cols]


# %%


lan_master_data = pd.read_csv("/Users/uqsporra/uni_OneDrive/phd/kari_project/TreeGazer_data/lanmodulin/tg_lanmodulin_data.csv")
lan_master_data = lan_master_data.loc[:, ["org_name", "La"]]

outputs = [f"../results/lanmodulin/La_gp_9_trials.csv",
           f"../results/lanmodulin/La_knn_9_trials.csv",
           f"../results/lanmodulin/La_tg_tied_6_trials.csv",
           f"../results/lanmodulin/La_wde_La_lanmodulin_9_trials.csv"]

labels = [ "GP UCB (ProtT5 Embeddings)", "KNN (ProtT5 Embeddings)", "TreeGazer UCB", "Weighted distance estimate",]

tree = dendropy.Tree.get_from_path('/Users/uqsporra/git_repos/TreeGazer_paper/data/lanmodulin/lanmodulin.aln.treefile', schema='newick')

dist_map = np.zeros((len(tree.taxon_namespace), len(tree.taxon_namespace)))
pdc = tree.phylogenetic_distance_matrix()
for i, t1 in enumerate(tree.taxon_namespace[:-1]):
    for j, t2 in enumerate(tree.taxon_namespace[i+1:], i+1):
        d = pdc(t1, t2)
        dist_map[i, j] = d
        dist_map[j, i] = d

dist_df = pd.DataFrame(dist_map, index=["_".join(x.split()) for x in tree.taxon_namespace.labels()], columns=["_".join(x.split()) for x in tree.taxon_namespace.labels()])
condensed = squareform(dist_map)
Z = linkage(condensed, method="complete")
order = leaves_list(Z)
df_reordered = dist_df.iloc[order, order]

cmap = mpl.colors.ListedColormap(["white", "darkblue"])
norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

for output, label in zip(outputs, labels):
    tree_ucb = pd.read_csv(output)
    tree_ucb_subset = tree_ucb[tree_ucb["trial"] == 0].copy()
    start_training_data = lan_master_data[~lan_master_data["org_name"].isin(tree_ucb_subset["candidate_id"])]

    sampled_map = pd.DataFrame(0, index=df_reordered.index, columns=df_reordered.columns)

    step = 100
    fig, axes = plt.subplots(1, len(range(step, len(tree_ucb_subset), step)) + 1, figsize=(20, 5))
    sns.heatmap(df_reordered, cmap='viridis', ax=axes[0], xticklabels=False, yticklabels=False)
    axes[0].set_title("Phylogenetic distance matrix")
    axes[0].set_ylabel("Taxa")
    count = 1
    for i in range(step, len(tree_ucb_subset), step): 
        
        train_subset_data = lan_master_data[lan_master_data["org_name"].isin(tree_ucb_subset.iloc[0:i]["candidate_id"])]
        accumulated_train_data = pd.concat([start_training_data, train_subset_data])
        sampled_map.loc[accumulated_train_data["org_name"], :] = 1
        sns.heatmap(sampled_map, cmap=cmap, norm=norm, cbar=False, ax=axes[count])
        axes[count].set_title(f"Iteration {i}") 
        axes[count].set_ylabel("Taxa")
        axes[count].set_xticks([])
        axes[count].set_yticks([])
        count += 1
    
    title = f"Sampling pattern - {label}"
    plt.suptitle(title)

    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, 1])
    
    cbar.set_ticklabels(["Unsampled (0)", "Sampled (1)"])

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(f"../results/lanmodulin/log10_La_{label}_sampling_pattern.png", bbox_inches='tight', format="png", dpi=300)
    plt.show()


