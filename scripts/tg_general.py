#!/usr/bin/env -S uv run --script
# /// script
# requires-python= ">=3.12"
# dependencies = [
#           "pandas", "numpy", "scikit-learn"
# ]
# ///

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import subprocess
import os
from sklearn.neighbors import KernelDensity
import argparse

ID = 0
VALUE = 1
SD = 2
PI = 3
INDEX = 2
MEAN = 1
UCB = 4


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
        "--tree",
        type=str,
        required=True,
        help="Path to the phylogenetic tree file ",
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
        "--metric",
        type=str,
        choices=["PI", "UCB"],
        default="UCB",
        help="Bayesian optimization metric to use (PI or UCB, default: UCB)",
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
        "--jar-path",
        default="treegazer.jar",
        type=str,
        help="Path to JAR file",
    )

    return parser.parse_args()


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


def main(
    feature, tree_path, trials, dataset_path, output_path, bo_metric, test_size, seed
):

    sep = dataset_path.split(".")[-1]
    if sep == ".tsv":
        dataset = pd.read_csv(dataset_path, sep="\t")
    else:
        dataset = pd.read_csv(dataset_path)

    dataset = dataset.sort_values(by=feature, ascending=True)

    actual_kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
        dataset[feature].to_numpy()[:, np.newaxis]
    )

    run_prefix = output_path.split(".")[0]
    out_metrics = open(output_path, "w")
    stderr_handle = f"{run_prefix}.error"
    stderr = open(stderr_handle, "w")
    print(
        "iterations,js_dist,trial,candidate_id,actual_score,predicted_mean,predicted_sd",
        file=out_metrics,
    )

    for random_state in range(seed, seed + trials):
        train, test = train_test_split(
            dataset, test_size=test_size, random_state=random_state
        )

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

        dists = []
        iterations = []
        scores = []
        ids = []
        while len(test_data) > 0:
            num_iterations += 1

            temp_out_handle = f"{run_prefix}_train_iter_{num_iterations}.tsv"

            # remove old training files
            if num_iterations > 1:
                old_out_handle = f"{run_prefix}_train_iter_{num_iterations - 1}.tsv"
                if os.path.exists(old_out_handle):
                    os.remove(old_out_handle)

                old_param_handle = f"{run_prefix}_iter_{num_iterations - 1}.params"
                if os.path.exists(old_param_handle):
                    os.remove(old_param_handle)

                old_pred_handle = f"{run_prefix}_iter_{num_iterations - 1}_preds.out"
                if os.path.exists(old_pred_handle):
                    os.remove(old_pred_handle)

            construct_training_file(train_data, feature, temp_out_handle)

            param_handle = f"{run_prefix}_iter_{num_iterations}.params"
            pred_handle = f"{run_prefix}_iter_{num_iterations}_preds.out"
            # run Treegazer
            learn_cmd = f"java -jar {args.jar_path} -nwk {tree_path} -in {feature}@{temp_out_handle} -params {param_handle} -internal -learn -latent 3 -seed {random_state} -verbose"
            inference_cmd = f"java -jar {args.jar_path} -nwk {tree_path} -in {feature}@{temp_out_handle} -latent 3 -internal -params {param_handle} -seed {random_state} -marg -out {pred_handle} -verbose"

            subprocess.run(learn_cmd, shell=True, stderr=stderr)
            subprocess.run(inference_cmd, shell=True, stderr=stderr)

            # iterate through the candidates from best to worst
            out_tsv = pd.read_csv(f"{pred_handle}.tsv", sep="\t")

            # removes predictions of ancestors, will only include extants
            preds = out_tsv[out_tsv["Entry"].isin(dataset["org_name"])]

            # will only use the training data to estimate the density.
            # preds = out_tsv[out_tsv["Entry"].isin([x[0] for x in train_data])]

            pred_kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
                preds[f"{feature} (Mean)"].to_numpy()[:, np.newaxis]
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
            dists.append(js_dist)
            iterations.append(num_iterations)

            out_tsv = out_tsv[out_tsv["Entry"].isin(dataset["org_name"])]
            out_tsv = out_tsv.dropna(subset=[f"{feature} ({bo_metric})"])
            out_tsv = out_tsv.sort_values(by=f"{feature} ({bo_metric})")
            best_candidate = out_tsv.iloc[-1, :]
            real_score = (
                dataset[dataset["org_name"] == best_candidate.iloc[ID]]
                .loc[:, feature]
                .values[0]
            )
            scores.append(real_score)
            ids.append(best_candidate.iloc[ID])

            new_test = []
            for candidate in test_data:
                if candidate[ID] == best_candidate.iloc[ID]:
                    train_data.append(
                        candidate
                    )  # Add the actual candidate from test_data
                    training_data_ids.add(candidate[ID])

                else:
                    new_test.append(
                        candidate
                    )  # Keep candidates that aren't the best_candidate

            test_data = new_test
            print(
                f"{num_iterations},{js_dist},{random_state},{best_candidate.iloc[ID]},{real_score},{best_candidate.iloc[MEAN]},{best_candidate.iloc[SD]}",
                file=out_metrics,
            )
            out_metrics.flush()

    stderr.close()
    out_metrics.close()


if __name__ == "__main__":
    args = parse_args()

    main(
        feature=args.feature,
        tree_path=args.tree,
        trials=args.trials,
        dataset_path=args.dataset,
        output_path=args.output,
        bo_metric=args.metric,
        test_size=args.test_size,
        seed=args.seed,
    )
