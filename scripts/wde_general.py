#!/usr/bin/env -S uv run --script
# /// script
# requires-python= "==3.12"
# dependencies = [
#           "pandas", "numpy", "scikit-learn", "h5py", "ete3"
# ]
# ///

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
import argparse
import ete3


def parse_args():
    """Parse command line arguments for the program."""
    parser = argparse.ArgumentParser(
        description="Run analysis with Bayesian optimization metrics."
    )

    # Add arguments with appropriate defaults matching your original code
    parser.add_argument(
        "--feature",
        type=str,
        required=True,
        help="Feature to analyze",
    )

    parser.add_argument(
        "--trials", type=int, default=5, help="Number of trials to run (default: 5)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=5,
        help="Random seed to start from, will iterate upwards according to number of trials",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Path to the dataset CSV file",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path to the output CSV file",
    )

    parser.add_argument(
        "--test-size",
        default=0.8,
        type=float,
        help="Initial test dataset size (Default: 0.8)",
    )

    parser.add_argument(
        "--tree",
        type=str,
        required=True,
        help="Path to the phylogenetic tree file ",
    )

    parser.add_argument("--leaves-only", action="store_true")

    return parser.parse_args()


def get_tree_indices(tree: ete3.Tree, leaves_only=False):
    """
    sequence_to_idx: dict, idx_to_sequences: dict
    """
    refcnt = 0
    sequence_to_idx = {}
    idx_to_sequence = {}
    for node in tree.traverse("postorder"):
        if leaves_only:
            if node.is_leaf():
                sequence_to_idx[node.name] = refcnt
                idx_to_sequence[refcnt] = node.name
                refcnt += 1
        else:
            sequence_to_idx[node.name] = refcnt
            idx_to_sequence[refcnt] = node.name
            refcnt += 1

    return sequence_to_idx, idx_to_sequence


def convert_tree_into_matrix(phl_tree: ete3.Tree, leaves_only=False):
    sequence_to_idx, idx_to_sequence = get_tree_indices(
        phl_tree, leaves_only
    )  # get the sequence reference dict

    tree_sequences = list(sequence_to_idx.keys())  # get all node names
    # print(f"Tree has {len(tree_sequences)} sequences including internal nodes")
    cophenetic_dist_np = np.zeros(
        (len(tree_sequences), len(tree_sequences))
    )  # create the matrix

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

    return (
        cophenetic_dist_np,
        sequence_to_idx,
        idx_to_sequence,
    )


def construct_training_file(data: list, feature: str, out_handle: str):
    ids = []
    values = []
    for item in data:
        ids.append(item[ID])
        values.append(item[VALUE])

    df = pd.DataFrame({"Entry": ids, feature: values})
    df.to_csv(out_handle, sep="\t", index=False)


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


def main():
    args = parse_args()

    sep = args.dataset.split(".")[-1]
    if sep == ".tsv":
        dataset = pd.read_csv(args.dataset, sep="\t")
    else:
        dataset = pd.read_csv(args.dataset)

    dataset = dataset.sort_values(by=args.feature, ascending=True)

    actual_kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
        dataset[args.feature].to_numpy()[:, np.newaxis]
    )

    tree = ete3.Tree(args.tree, format=1)

    coph_dist, sequence_to_idx, idx_to_sequence = convert_tree_into_matrix(
        tree, args.leaves_only
    )

    out_metrics = open(args.output, "w")
    print(
        "iterations,js_dist,trial,candidate_id,actual_score,predicted_mean,predicted_sd",
        file=out_metrics,
    )

    for random_state in range(args.seed, args.seed + args.trials):
        train, test = train_test_split(
            dataset, test_size=args.test_size, random_state=random_state
        )

        train_data = [
            (name, val)
            for name, val in zip(
                train["org_name"],
                train[args.feature],
            )
        ]
        test_data = [
            (name, val)
            for name, val in zip(
                test["org_name"],
                test[args.feature],
            )
        ]

        num_iterations = 0

        while len(test_data) > 0:
            num_iterations += 1

            train_y = [x[VALUE] for x in train_data]
            preds = []
            for test_node in test_data:
                weighted_score = 0
                all_dists = 0

                for train_node in train_data:
                    branch_dist = coph_dist[
                        sequence_to_idx[test_node[ID]],
                        sequence_to_idx[train_node[ID]],
                    ]

                    weighted_score += (1 / branch_dist) * train_node[VALUE]
                    all_dists += 1 / branch_dist

                weighted_score /= all_dists
                preds.append(weighted_score)
                # print(test_node[ID], weighted_score, test_node[VALUE])

            # need this to estimate the density from, all data included
            preds_and_actual = np.array(preds + train_y)

            preds_and_actual = np.array(train_y)

            pred_kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
                preds_and_actual[:, np.newaxis]
            )

            x_range = np.linspace(
                dataset[args.feature].to_numpy()[:, np.newaxis].min() - 0.01,
                dataset[args.feature].to_numpy()[:, np.newaxis].max() + 0.01,
                num=600,
            )

            # compute the log-likelihood of each sample
            pred_log_density = pred_kde.score_samples(x_range[:, np.newaxis])
            actual_log_density = actual_kde.score_samples(x_range[:, np.newaxis])

            p = np.exp(actual_log_density)
            q = np.exp(pred_log_density)

            js_dist = jensen_shannon_distance(p, q, x_range)

            best_candidate = None
            best_score = -np.inf
            for pred, candidate in zip(preds, test_data):
                if pred > best_score:
                    best_score = pred
                    best_candidate = candidate

            new_test = []
            for candidate in test_data:
                if candidate[ID] == best_candidate[ID]:
                    train_data.append(
                        candidate
                    )  # Add the actual candidate from test_data
                else:
                    new_test.append(
                        candidate
                    )  # Keep candidates that aren't the best_candidate

            test_data = new_test
            print(
                f"{num_iterations},{js_dist},{random_state},{best_candidate[ID]},{best_candidate[VALUE]},{best_score},",
                file=out_metrics,
            )
            out_metrics.flush()

    out_metrics.close()


if __name__ == "__main__":
    main()
