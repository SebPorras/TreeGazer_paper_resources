#!/usr/bin/env -S uv run --script
# /// script
# requires-python= ">=3.12"
# dependencies = [
#           "pandas",
#           "numpy",
#           "scikit-learn",
#           "h5py",
#           "torch",
#           "gpytorch"]
# ///

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
import h5py
import torch
import gpytorch
import argparse


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
        "--metric",
        type=str,
        choices=["PI", "UCB"],
        default="UCB",
        help="Bayesian optimization metric to use (PI or UCB, default: UCB)",
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
        "--lam",
        default=5,
        type=int,
        help="lambda multiplier(Default: 5)",
    )

    return parser.parse_args()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)  # type: ignore[reportArgumentType]


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
SD = 2
PI = 3
INDEX = 2
EMB = 2


def main():

    torch.set_num_threads(5)
    args = parse_args()

    out_metrics = open(args.output, "w")
    print(
        "iterations,js_dist,trial,candidate_id,actual_score,predicted_mean,predicted_sd",
        file=out_metrics,
    )

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

    dataset = dataset.merge(emb_df, on="org_name")

    actual_kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
        dataset[args.feature].to_numpy()[:, np.newaxis]
    )

    for random_state in range(args.seed, args.seed + args.trials):
        train, test = train_test_split(
            dataset, test_size=0.8, random_state=random_state
        )
        print(train, test)

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
        ids = []
        dtype = torch.float32
        while len(test_data) > 0:
            num_iterations += 1

            train_x_emb = torch.from_numpy(np.array([x[EMB] for x in train_data])).to(
                dtype
            )
            train_y = torch.from_numpy(np.array([x[VALUE] for x in train_data])).to(
                dtype
            )

            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dtype)
            model = ExactGPModel(train_x_emb, train_y, likelihood).to(dtype)

            training_iter = 100

            # Find optimal model hyperparameters
            model.train()
            likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.1
            )  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            last_loss = None
            loss_counter = 0
            limit = 3
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(train_x_emb)
                # Calc loss and backprop gradients
                loss = -mll(output, train_y)  # type: ignore
                loss.backward()
                # print(
                #     "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
                #     % (
                #         i + 1,
                #         training_iter,
                #         loss.item(),
                #         model.covar_module.base_kernel.lengthscale.item(),
                #         model.likelihood.noise.item(),
                #     )
                # )
                optimizer.step()

                if last_loss is None:
                    last_loss = loss.item()
                elif loss >= last_loss:
                    loss_counter += 1
                elif loss < last_loss:
                    loss_counter = 0

                if loss_counter == limit:
                    break

                last_loss = loss.item()

            test_x_emb = torch.from_numpy(np.array([x[EMB] for x in test_data])).to(
                dtype
            )
            test_ids = [x[ID] for x in test_data]
            # Get into evaluation (predictive posterior) mode
            model.eval()
            likelihood.eval()

            # Test points are regularly spaced along [0,1]
            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(test_x_emb))
                pred_means = observed_pred.mean
                pred_stds = observed_pred.stddev

            # need this to estimate the density from, all data included
            preds_and_actual = np.array(list(pred_means) + list(train_y))

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
            best_mean = 0
            best_std = 0
            for pred_mean, pred_std, candidate in zip(pred_means, pred_stds, test_ids):
                if args.metric == "UCB":
                    cur_metric = pred_mean + args.lam * pred_std
                else:
                    raise NotImplementedError("PI not implemented")

                if cur_metric > best_score:
                    best_score = cur_metric
                    best_candidate = candidate
                    best_mean = pred_mean
                    best_std = pred_std

            new_test = []
            for candidate in test_data:
                if candidate[ID] == best_candidate:
                    train_data.append(
                        candidate
                    )  # Add the actual candidate from test_data

                    print(
                        f"{num_iterations},{js_dist},{random_state},{candidate[ID]},{candidate[VALUE]},{best_mean},{best_std}",
                        file=out_metrics,
                    )
                    out_metrics.flush()

                else:
                    new_test.append(
                        candidate
                    )  # Keep candidates that aren't the best_candidate

            test_data = new_test

    out_metrics.close()


if __name__ == "__main__":
    main()
