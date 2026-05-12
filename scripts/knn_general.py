#!/usr/bin/env -S uv run --script
# /// script
# requires-python= ">=3.12"
# dependencies = [
#           "pandas", "numpy", "scikit-learn", "h5py"
# ]
# ///

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
import h5py
from sklearn.metrics.pairwise import cosine_distances
from collections import Counter
import argparse
from typing import Tuple


def parse_args():
    """Parse command line arguments for the program."""
    parser = argparse.ArgumentParser(
        description="Run analysis with Bayesian optimization metrics."
    )

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
        "--embeddings",
        required=True,
        type=str,
        help="Path to the embeddings h5py file",
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
        "--k",
        default=3,
        type=int,
        help="Number of K nearest neighbours (Default: 3)",
    )

    return parser.parse_args()


class KNN:
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray, n: int = 3) -> None:
        self.train_x: np.ndarray = train_x
        self.train_y: np.ndarray = train_y
        self.n: int = n

    def predict(
        self, test_x: np.ndarray, is_discrete: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        preds = list()
        preds_sd = list()
        for test_seq in test_x:
            dists = self.calc_cosine(test_seq)
            indices = np.argsort(dists)
            sorted_values = self.train_y[indices]

            top_n = sorted_values[: self.n]

            if is_discrete:
                counter = Counter(top_n)
                preds.append(counter.most_common(1)[0][0])
            else:
                preds.append(np.mean(top_n))
                preds_sd.append(np.std(top_n))

        return np.array(preds), np.array(preds_sd)

    def calc_cosine(self, test_seq: np.ndarray) -> np.ndarray:
        dists = []
        for i in self.train_x:
            d = cosine_distances(test_seq.reshape(1, -1), i.reshape(1, -1))
            # FIX THIS to avoid strange indexing
            d = d[0][0]
            dists.append(d)

        return np.array(dists)


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
SD = 2
PI = 3
INDEX = 2
EMB = 2


def main():
    args = parse_args()

    sep = args.dataset.split(".")[-1]
    if sep == ".tsv":
        dataset = pd.read_csv(args.dataset, sep="\t")
    else:
        dataset = pd.read_csv(args.dataset)

    dataset = dataset.sort_values(by=args.feature, ascending=True)

    with h5py.File(args.embeddings) as file:
        ids = [key.replace("_1", ".1") for key in file.keys()]
        embs = [np.array(key) for key in file.values()]

    emb_df = pd.DataFrame({"org_name": ids, "embedding": embs})
    print(emb_df)

    dataset = dataset.merge(emb_df, on="org_name")

    actual_kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
        dataset[args.feature].to_numpy()[:, np.newaxis]
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
            (name, val, encoding)
            for name, val, encoding in zip(
                train["org_name"],
                train[args.feature],
                train["embedding"],
            )
        ]
        test_data = [
            (name, val, encoding)
            for name, val, encoding in zip(
                test["org_name"],
                test[args.feature],
                test["embedding"],
            )
        ]

        num_iterations = 0

        while len(test_data) > 0:
            num_iterations += 1

            train_x_emb = np.stack([x[EMB] for x in train_data])
            train_y = np.array([x[VALUE] for x in train_data])
            cos_knn = KNN(train_x_emb, train_y, n=args.k)

            test_x_emb = np.stack([x[EMB] for x in test_data])
            test_ids = [x[ID] for x in test_data]
            preds, preds_sd = cos_knn.predict(test_x_emb)

            # need this to estimate the density from, all data included
            preds_and_actual = np.array(list(preds) + list(train_y))

            # only training data
            preds_and_actual = np.array(list(train_y))

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
            best_sd = -np.inf
            for pred, pred_sd, candidate in zip(preds, preds_sd, test_ids):
                if pred > best_score:
                    best_score = pred
                    best_sd = pred_sd
                    best_candidate = candidate

            new_test = []
            # best_candidate_data = None
            for candidate in test_data:
                if candidate[ID] == best_candidate:
                    train_data.append(
                        candidate
                    )  # Add the actual candidate from test_data
                    best_candidate_data = candidate
                else:
                    new_test.append(
                        candidate
                    )  # Keep candidates that aren't the best_candidate

            test_data = new_test
            print(
                f"{num_iterations},{js_dist},{random_state},{best_candidate_data[ID]},{best_candidate_data[VALUE]},{best_score},{best_sd}",
                file=out_metrics,
            )
            out_metrics.flush()

    out_metrics.close()


if __name__ == "__main__":
    main()
