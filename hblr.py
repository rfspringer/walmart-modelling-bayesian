from sklearn.preprocessing import StandardScaler
import pyro
import random
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import pandas as pd
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoNormal
from pyro.optim import Adam
import matplotlib.pyplot as plt
import arviz as az
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from pyro.infer import Predictive
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


# Select relevant columns
worker_regression_cols = [
    'Is Male', 'Time To First Convo (Quart. 1)',
    'Time To First Convo (Quart. 2)',
    'Time To First Convo (Quart. 3)',
    'Time To First Convo (Quart. 4)',
    'Log(Eigen. Centrality)',
    'No. Notes'
]

worksite_regression_cols = [
    'Percent black in ZIP',
    'Percent Latino in ZIP',
    'Percent male',
    'Log(Mean AGI in ZIP)',
    'NDO (Eigen., Pearson)',
       'Mobilizing (Pearson)',
       'Campaign length (Quint. 1)',
       'Campaign length (Quint. 2)',
       'Campaign length (Quint. 3)',
       'Campaign length (Quint. 4)',
       'Campaign length (Quint. 5)',
       # 'Log(Workers signed per worker-week)',
       # 'Log(Workers signed per organizer convers.)',
       'Log(No. organizer convers.)',
       'Log(No. workers contacted)',
       'Log(No. workers discovered)',
        # 'Log(No. workers signed)',
       'Log(No. edges)',
       'Log(Mean degree)',
       # 'Log(Convers. per worker variance)',
       'Log(Centrality variance)',
       'Log(Centrality mean)'
]


def load_data():
    # Load data
    worker_df = pd.read_csv('WorkerRegressionData.csv')
    worksite_df = pd.read_csv('WorksiteNetworkData.csv')

    # Regularize regression cols
    worker_scaler = StandardScaler()
    worksite_scaler = StandardScaler()
    worker_data = worker_scaler.fit_transform(worker_df[worker_regression_cols].to_numpy())
    worksite_data = worksite_scaler.fit_transform(worksite_df[worksite_regression_cols].to_numpy())
    worker_data = np.column_stack((worker_data, worker_df['Worksite ID'].values))

    # Split the data into training and testing sets
    worker_data, held_out_worker_data, outcomes, held_out_outcomes = \
        train_test_split(worker_data, worker_df['Signed'].values, test_size=0.2, random_state=42)

    # Map the values list to their indices
    index_mapping = {worksite_id: index for index, worksite_id in enumerate(worksite_df['Worksite ID'])}
    map_function = np.vectorize(lambda x: index_mapping.get(x, None))
    worksite_indices = map_function(worker_data[:, -1])
    held_out_worksite_indices = map_function(held_out_worker_data[:, -1])

    worker_data = worker_data[:,:-1]
    held_out_worker_data = held_out_worker_data[:,:-1]

    worker_data = torch.tensor(worker_data, dtype=torch.float32)
    worksite_data = torch.tensor(worksite_data, dtype=torch.float32)
    outcomes = torch.tensor(outcomes, dtype=torch.float32)

    held_out_worker_data = torch.tensor(held_out_worker_data, dtype=torch.float32)
    held_out_outcomes = torch.tensor(held_out_outcomes, dtype=torch.float32)

    return (worker_data, worksite_data, worksite_indices, outcomes), \
           (held_out_worker_data, worksite_data, held_out_worksite_indices, held_out_outcomes)


def model(worker_data, worksite_data, worksite_indices, outcomes):
    num_individual_predictors = worker_data.shape[1]
    num_group_predictors = worksite_data.shape[1]

    beta_individual_scale = pyro.sample('beta_individual_scale', dist.HalfCauchy(5))
    beta_group_scale = pyro.sample('beta_group_scale', dist.HalfCauchy(5))

    beta_group = pyro.sample('beta_group', dist.Normal(0, beta_group_scale).expand([num_group_predictors]).to_event(1))

    # Priors
    alpha = pyro.sample('alpha_mean', dist.Normal(0, beta_group_scale))

    with pyro.plate('worksite', len(worksite_data)):
        beta_individual = pyro.sample('beta_individual', dist.Normal(0, beta_individual_scale).expand([num_individual_predictors]).to_event(1))

    logits = alpha + torch.matmul(beta_group, worksite_data.T).sum(dim=-1) \
            + torch.matmul(beta_individual[worksite_indices], worker_data.T).sum(dim=-1)

    with pyro.plate('worker', len(worker_data)):
        y = pyro.sample('y', dist.Bernoulli(logits=logits), obs=outcomes)
    return logits


def vi():
    # Load data
    inputs, held_out_inputs = load_data()
    worker_data, worksite_data, worksite_indices, outcomes = inputs

    pyro.render_model(model, model_args=(*inputs,), filename="model_graph.pdf", render_params=True, render_distributions=True)
    guide = AutoNormal(model)
    pyro.clear_param_store()

    num_iterations = 8000

    scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {"lr": 0.05, "betas": (0.95, 0.999)}, 'gamma': 0.999})
    svi = SVI(model, guide, scheduler, loss=Trace_ELBO())

    # Training loop
    log_likelihoods = []
    losses = []
    val_losses = []
    for step in range(num_iterations):
        loss = svi.step(*inputs)
        losses.append(loss)
        scheduler.step()

        # Evaluate held-out log likelihood during training
        if step % 100 == 0:
            # predictive = Predictive(model, guide=guide, num_samples=100)
            # samples = predictive(*held_out_inputs)
            #log_likelihood = dist.Bernoulli(logits=samples["obs"]).log_prob(held_out_inputs[-1]).mean().item()
            #log_likelihoods.append(log_likelihood)
            print(f"Step {step}/{num_iterations}, Loss: {loss}")
            with torch.no_grad():
                val_loss = svi.evaluate_loss(*held_out_inputs)
                print(f'Validation ELBO: {val_loss}')

    guide_params = guide(*inputs)
    print("guide params: ", guide_params)
    torch.save(guide_params, 'guide_params.pt')

    model_trace = pyro.poutine.trace(model).get_trace(*inputs)
    guide_trace = pyro.poutine.trace(guide).get_trace(*inputs)


    print(model_trace.nodes)
    print(guide_trace.nodes)
    torch.save(log_likelihoods, 'log_likelihoods.pt')

    # sample from posterior predictive
    predictive = Predictive(model, guide=guide, num_samples=5000)
    samples = predictive(*held_out_inputs)
    torch.save(samples, 'samples.pt')

    plot_loss(losses)

    quantiles = guide.quantiles([0.25, 0.5, 0.75])
    torch.save(quantiles, 'quantiles.pt')

    pred_summary = summary(samples)
    torch.save(pred_summary, 'pred_summary.pt')



def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": torch.quantile(v, 0.05, dim=0),
            "95%": torch.quantile(v, 0.95, dim=0),
        }
    return site_stats


def build_predictions_df(pred_summary, worker_data, worksite_data, worksite_indices, outcomes):
    y = pred_summary["obs"]
    predictions = pd.DataFrame({
        "y_mean": y["mean"],
        "y_perc_5": y["5%"],
        "y_perc_95": y["95%"],
        "true_outcome": outcomes,
    })

    for i in range(len(worksite_regression_cols)):
        predictions[worksite_regression_cols[i]] = worksite_data[worksite_indices][:, i]

    for j in range(len(worker_regression_cols)):
        predictions[worker_regression_cols[j]] = worker_data[:,j]

    return predictions


def plot_group_post_pred(worksite_data, worksite_indices, predictions):
    fig, ax = plt.subplots(nrows=1, ncols=len(worksite_regression_cols), figsize=(18, 6), sharey=True)
    fig.suptitle("Posterior predictive distribution with 90% CI", fontsize=16)
    worksite_data_by_prediction = worksite_data[worksite_indices]

    # Plot group features
    for i in range(len(worksite_regression_cols)):
        predictions = predictions.sort_values(by=[worksite_regression_cols[i]])
        # map worksite index to worksite_data rows
        # need to somehow match worksite data to expand to each worker
        feature = predictions[[worksite_regression_cols[i]]].to_numpy().flatten()
        ax[i].plot(feature, predictions["y_mean"], label='Predicted Probability')
        ax[i].fill_between(feature, predictions["y_perc_5"], predictions["y_perc_95"],
                           alpha=0.5, label='90% CI')
        ax[i].plot(feature, predictions["true_outcome"], "o", label='True Outcome')
        ax[i].set(xlabel=worksite_regression_cols[i], ylabel="Probability")
        ax[i].legend()
    plt.show()


def plot_individual_post_pred(worker_data, worksite_indices, predictions):
    for j in range(len(worksite_indices)):
        fig, ax = plt.subplots(nrows=1, ncols=len(worker_regression_cols), figsize=(18, 6), sharey=True)
        fig.suptitle("Posterior predictive distribution with 90% CI", fontsize=16)

        worksite_predictions = predictions[worksite_indices == j]
        worker_data_for_worksite = worker_data[worksite_indices == j]

        # Plot individual features
        for k in range(len(worker_regression_cols)):
            feature = worker_data_for_worksite[:, k]
            #GET WORKSITE ID
            ax[k].plot(feature, worksite_predictions["y_mean"],
                                      label='Predicted Probability')
            ax[k].fill_between(feature, worksite_predictions["y_perc_5"].values, worksite_predictions["y_perc_95"].values,
                                              alpha=0.5, label='90% CI')
            ax[k].plot(feature, worksite_predictions["true_outcome"], "o",
                                      label='True Outcome')
            ax[k].set(xlabel=worker_regression_cols[k], ylabel="Probability", title=f"Worksite {j}")
            ax[k].legend()
        plt.show()


def plot_loss(losses):
    # Plot losses
    plt.figure(figsize=(5, 2))
    plt.plot(losses)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")
    plt.show()


def plot_summary(site_stats):
    # Extract parameter names and statistics
    parameter_names = list(site_stats.keys())
    means = [stats["mean"].item() for stats in site_stats.values()]
    stds = [stats["std"].item() for stats in site_stats.values()]
    quantile_5 = [stats["5%"].item() for stats in site_stats.values()]
    quantile_95 = [stats["95%"].item() for stats in site_stats.values()]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(parameter_names))

    # Plot means with error bars (standard deviation or quantiles)
    ax.bar(x, means, yerr=stds, capsize=5, label='Mean ± Std Dev', alpha=0.7)

    # Alternatively, you can use quantiles
    # ax.bar(x, means, yerr=[means - quantile_5, quantile_95 - means], capsize=5, label='Mean ± 90% CI', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(parameter_names, rotation=45, ha="right")
    ax.set_ylabel("Parameter Value")
    ax.set_title("Parameter Summary")
    ax.legend()

    plt.tight_layout()
    plt.show()


def held_out_log_lik(model, guide, held_out_inputs):
    held_out_worker_data, worksite_data, held_out_worksite_indices, held_out_outcomes = held_out_inputs

    # Sample from the predictive distribution
    predictive = Predictive(model, guide=guide, num_samples=1000)
    samples = predictive(*held_out_inputs)
    torch.save(samples, 'held_out_samples.pt')

    # Compute log likelihoods
    log_likelihoods = []
    for i in range(len(held_out_worker_data)):
        # Modify this line to access the logits or probability directly from the model
        logits = model(*held_out_inputs)  # Replace with the actual attribute or calculation
        log_likelihood = dist.Bernoulli(logits=logits).log_prob(held_out_outcomes[i]).mean()
        log_likelihoods.append(log_likelihood.item())

    # Plot log likelihoods on held-out data
    plt.hist(log_likelihoods, bins=20, density=True, alpha=0.75, label="Log Likelihoods")
    plt.xlabel("Log Likelihood")
    plt.ylabel("Density")
    plt.title("Log Likelihood on Held-Out Data")
    plt.legend()
    plt.show()

    return log_likelihoods


def posterior_predictive_check(held_out_inputs):
    simulated_data = torch.load('held_out_samples.pt')

    # Convert the simulated data to NumPy arrays
    simulated_data_np = simulated_data["obs"].numpy()
    observed_data_np = held_out_inputs[-1].numpy()

    # Plot KDEs for simulated and observed data
    sns.kdeplot(simulated_data_np.flatten(), label='Simulated', fill=True)
    sns.kdeplot(observed_data_np.flatten(), label='Observed', fill=True)
    plt.legend()
    plt.show()


def post_pred():
    #NOW TESTING ON BASELINE- CHANGE BACK BEFORE TESTING AGAIN
    inputs, held_out = load_data()
    held_out_worker_data, worksite_data, held_out_worksite_indices, held_out_outcomes = held_out

    guide_params = torch.load('guide_params.pt')
    guide = AutoNormal(pyro.poutine.block(model, hide=['y']))

    # update the guide parameters with the saved values
    for param_name, param_value in guide_params.items():
        pyro.param(param_name, param_value)

    predictive = Predictive(model, guide=guide, num_samples=3000)
    posterior_predictive_samples = predictive(held_out_worker_data, worksite_data, held_out_worksite_indices, torch.zeros(len(held_out_outcomes)))

    # extract predictions and observed data
    y_posterior_predictive = posterior_predictive_samples['y'].mean(dim=0)
    y_observed = held_out_outcomes

    # convert continuous probabilities to binary values using a threshold (e.g., 0.5)
    y_pred_binary = (y_posterior_predictive >= 0.5).numpy()

    # compute and print metrics
    accuracy = accuracy_score(y_observed, y_pred_binary)
    auc = roc_auc_score(y_observed, y_posterior_predictive.numpy())

    print(f'Accuracy: {accuracy:.4f}')
    print(f'AUC: {auc:.4f}')
    print(classification_report(y_observed, y_pred_binary))

    # plot ppc
    sns.kdeplot(y_posterior_predictive, label='Simulated', fill=True)
    sns.kdeplot(y_observed.flatten(), label='Observed', fill=True)
    plt.legend()
    plt.title('Posterior Predictive Check')
    plt.show()


if __name__ == "__main__":
    vi()
    post_pred()

