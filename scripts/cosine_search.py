import ete3
import numpy as np
import h5py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from Bio import SeqIO
from typing import Dict
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def get_tree_indices(tree: ete3.Tree):
    """
    sequence_to_idx: dict, idx_to_sequences: dict
    """
    refcnt = 0
    sequence_to_idx = {}
    idx_to_sequence = {}
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            sequence_to_idx[node.name] = refcnt
            idx_to_sequence[refcnt] = node.name
            refcnt += 1

    return sequence_to_idx, idx_to_sequence


def negative_exponential_pdf(x, lam: float = 1.0):
    if x < 0.2:
        return 0.0

    return lam * np.e ** (-1 * lam * x)


def convert_tree_into_matrix(
    phl_tree: ete3.Tree, seq_embeddings: Dict[str, np.ndarray]
):
    sequence_to_idx, idx_to_sequence = get_tree_indices(
        phl_tree
    )  # get the sequence reference dict

    tree_sequences = list(sequence_to_idx.keys())  # get all node names
    # print(f"Tree has {len(tree_sequences)} sequences including internal nodes")
    cophenetic_dist_np = np.zeros(
        (len(tree_sequences), len(tree_sequences))
    )  # create the matrix

    weight_correction = np.zeros((len(tree_sequences), len(tree_sequences)))

    emb_sim = np.zeros((len(tree_sequences), len(tree_sequences)))

    for ref_num, ref_sequence in enumerate(tree_sequences):
        for other_sequence in tree_sequences[ref_num:]:
            # grab tree nodes and calculate phylogenetic distance
            ref_extant_sequence_t = phl_tree & ref_sequence
            other_extant_sequence_t = phl_tree & other_sequence
            dis_node = ref_extant_sequence_t.get_distance(other_extant_sequence_t)

            ref_idx = sequence_to_idx[ref_sequence]
            other_idx = sequence_to_idx[other_sequence]

            cophenetic_dist_np[ref_idx][other_idx] = dis_node
            cophenetic_dist_np[other_idx][ref_idx] = dis_node

            weight = negative_exponential_pdf(dis_node)
            weight_correction[ref_idx][other_idx] = weight
            weight_correction[other_idx][ref_idx] = weight

            similarity = cosine_similarity(
                seq_embeddings[ref_sequence], seq_embeddings[other_sequence]
            )[0][0]
            emb_sim[ref_idx][other_idx] = similarity
            emb_sim[other_idx][ref_idx] = similarity

    return (
        cophenetic_dist_np,
        weight_correction,
        emb_sim,
        sequence_to_idx,
        idx_to_sequence,
    )


def is_ancestor(id):
    if id[0] != "N":
        return False
    try:
        int(id[1:])
    except ValueError:
        return False

    return True


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


ID = 0
VALUE = 1
INDEX = 2
feature = "temperature"
tree_path = "adk_ancestors.nwk"
embedding_path = "adk_ancestors_extants_embeddings.h5"
trials = 30
top_percentile = 90
find_max = True

ext_seqs = list(SeqIO.parse("adk.fasta", "fasta"))
ext_ids = [x.id for x in ext_seqs]

emb_dict = {}
with h5py.File(embedding_path) as file:
    for key, value in file.items():
        emb_dict[key] = np.array(value).reshape(1, -1)

tree = ete3.Tree(tree_path, format=1)

coph_dist, weight_correction, emb_sim, sequence_to_idx, idx_to_sequence = (
    convert_tree_into_matrix(tree, emb_dict)
)

dataset = pd.read_csv("dataset_adk.tsv", sep="\t")
dataset = dataset.sort_values(by=feature, ascending=True)

actual_kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
    dataset[feature].to_numpy()[:, np.newaxis]
)


dataset = pd.read_csv("dataset_adk.tsv", sep="\t")
dataset = dataset.sort_values(by=feature, ascending=find_max)

all_dists = []
all_iterations = []
for random_state in range(trials):
    train, test = train_test_split(dataset, test_size=0.8, random_state=random_state)

    train_data = [
        (name, val, index)
        for name, val, index in zip(train["org_name"], train[feature], train.index)
    ]
    test_data = [
        (name, val, index)
        for name, val, index in zip(test["org_name"], test[feature], test.index)
    ]

    num_iterations = 0
    training_data_ids = set([x[ID] for x in train_data])
    stderr_handle = f"log_{feature}.error"
    stderr = open(stderr_handle, "w")

    dists = []
    iterations = []
    threshold = 1e-5
    last_dist = 0
    while len(test_data) > 0:
        num_iterations += 1

        results = []
        for candidate in test_data:
            cand_score = []
            known_features = []
            for known in train_data:
                candidate_idx = sequence_to_idx[candidate[ID]]
                known_idx = sequence_to_idx[known[ID]]

                cosine = emb_sim[candidate_idx, known_idx]
                # cosine_corr = (
                #     emb_sim[candidate_idx, known_idx]
                #     * weight_correction[candidate_idx, known_idx]
                # )
                cand_score.append(cosine)

                known_features.append(known[VALUE])

            known_features = np.array(known_features)
            percen = np.percentile(known_features, top_percentile)
            training_to_keep = known_features > percen

            top_10_percent = []
            for sim_score, keep in zip(cand_score, training_to_keep):
                if keep:
                    top_10_percent.append(sim_score)
            mean_sim_score = np.mean(top_10_percent)
            results.append((candidate[ID], mean_sim_score, candidate[VALUE]))

        results_sorted = sorted(results, key=lambda x: x[1], reverse=find_max)
        top_candidate = results_sorted[0]

        new_test = []
        for candidate in test_data:
            if candidate[ID] == top_candidate[ID]:
                train_data.append(candidate)  # Add the actual candidate from test_data
                training_data_ids.add(candidate[ID])

            else:
                new_test.append(
                    candidate
                )  # Keep candidates that aren't the best_candidate

        test_data = new_test

    all_dists.append(dists)
    all_iterations.append(iterations)
    stderr.close()


fig, ax = plt.subplots(1, 1)
trial = 1
data = []
for x, y in zip(all_iterations, all_dists):
    plt.plot(x, y, alpha=0.5)
    for round, dist in zip(x, y):
        data.append([round, dist, trial])
    trial += 1

df = pd.DataFrame(data, columns=["iterations", "dists", "trial"])

df.to_csv(f"cosine_treegazer_search_num_trials_{trials}_{feature}.csv", index=False)
plt.xlabel("Iteration")
plt.ylabel("Jensen-Shannon distance")
plt.title(f"cosine TreeGazer {feature}")
plt.savefig(f"cosine_treegazer_search_num_trials_{trials}_{feature}.png", dpi=300)
