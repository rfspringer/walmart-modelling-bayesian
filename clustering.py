import numpy as np
import random
import scipy
import matplotlib.pyplot as plt
import math
import torch

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


def sample_assignments(assignments, proportions, means, data, params):
    new_assignments = assignments.copy()

    for i in range(params['n']):
        # Calculate posterior probabilities
        posterior_probs = np.zeros(params['K'])
        for k in range(params['K']):
            prior = proportions[k]
            f = lambda x: scipy.stats.norm.logpdf(x, loc=means[k], scale=params['sigma'])
            likelihood = f(data[i]).sum()
            posterior_probs[k] = prior + likelihood

        posterior_probs -= np.max(posterior_probs)
        posterior_probs = np.exp(posterior_probs)

        # normalize
        assignment_probs = posterior_probs / np.sum(posterior_probs)

        # sample
        new_assignment = np.random.choice(params['K'], p=assignment_probs)
        new_assignments[i] = new_assignment
    return new_assignments


def sample_means(new_assignments, means, data, params):
    cluster_means = means.copy()
    new_means = means.copy()

    for k in range(params['K']):
        sigma = params['sigma']
        eta = params['eta']

        # gather data points assigned to component k
        data_indices = np.where(new_assignments == k)[0]
        component_data = data[data_indices]

        # update the mean
        if len(component_data) > 0:
            center = np.mean(component_data, axis=0)
            cluster_means[k] = center

        num_assignments = len(component_data)
        mean = ((num_assignments / (sigma * sigma)) / (num_assignments / (sigma * sigma) + 1 / (eta * eta))) * cluster_means[k]
        var = 1 / (num_assignments / (sigma * sigma) + 1 / (eta * eta))

        new_means[k] = np.random.normal(mean, math.sqrt(var))
    return new_means


def sample_proportions(new_assignments, params):
    alpha = np.full(params['K'], params['alpha_0'])
    for k in range(params['K']):
        alpha[k] += np.sum(new_assignments == k)
    new_proportions = np.random.dirichlet(alpha)
    return new_proportions


def held_out_log_likelihood(new_data, means, proportions, params):
    predictive_log_prob = 0
    probs = []
    for i in range(len(new_data)):
        log_local_likelihood = 0
        for k in range(params['K']):
            f = lambda x: scipy.stats.norm.logpdf(x, loc=means[k], scale=params['sigma'])
            log_local_cluster_likelihood = np.log(proportions[k]) + f(new_data[i])
            log_local_likelihood = np.logaddexp(log_local_likelihood, log_local_cluster_likelihood)
        probs.append(log_local_likelihood + log_prior(proportions, means, params))
        predictive_log_prob = np.logaddexp(predictive_log_prob, log_local_likelihood + log_prior(proportions, means, params))
        # predictive_log_prob += log_local_likelihood
    probs = np.array(probs)
    max_val = np.max(probs)
    shifted_arr = probs - max_val
    result = max_val + np.log(np.sum(np.exp(shifted_arr)))
    return result



def log_joint(assignments, proportions, means, data, params):    # log joint, held out log likelihood, then comparing to audio files to see
    log_likelihood = 0
    # log prob of data
    for i in range(params['n']):
        f = lambda x: scipy.stats.norm.logpdf(x, loc=means[assignments[i]], scale=params['sigma'])
        log_likelihood += f(data[i]).sum()

    #log prob of assignments given proportions
    # for each z assigned to a cluster, theta likelihood
    for k in range(params['K']):
        log_likelihood += np.log(proportions[k]) * np.sum(assignments == k)
    return log_likelihood + log_prior(proportions, means, params)


def log_prior(proportions, means, params):
    log_prob = 0
    # log prob of means
    # d zero mean gaussians with var of eta^2
    for k in range(params['K']):
        f = lambda x: scipy.stats.norm.logpdf(x, loc=0, scale=params['eta'])
        log_prob += f(means[k]).sum()

    # log prob of proportions
    alpha = np.full(params['K'], params['alpha_0'])
    log_prob += scipy.stats.dirichlet.logpdf(proportions, alpha)
    return log_prob


def gibbs(data, params):
    # Randomly initialize assignments, means, and proportions
    assignments = np.random.randint(0, params['K'], params['n'])
    means = [random.choice(data) for _ in range(params['K'])]
    proportions = np.random.dirichlet(np.ones(params['K']))
    log_joints = []

    for iteration in range(params['max_iterations']):
        new_assignments = sample_assignments(assignments, proportions, means, data, params)
        new_means = sample_means(new_assignments, means, data, params)
        new_proportions = sample_proportions(new_assignments, params)

        # Update parameters
        assignments = np.array(new_assignments)
        means = np.array(new_means)
        proportions = np.array(new_proportions)
        log_joints.append(log_joint(assignments, proportions, means, data, params))

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: \nAssignments = {assignments}, \nMeans = {means}, \nProportions = {proportions}\n")

    return assignments, means, proportions, log_joints


def plot_feature_scatter(data, means, feature_idx, ax, cluster_colors):
    if feature_idx == len(means[0]) - 1:
        feature_idx = feature_idx - 1

    num_clusters = len(means)

    for cluster_idx in range(num_clusters):
        cluster_data = data[assignments == cluster_idx]
        cluster_mean = means[cluster_idx]

        ax.scatter(cluster_data[:, feature_idx], cluster_data[:, feature_idx + 1], c=cluster_colors[cluster_idx],
                   label=f'Cluster {cluster_idx} Data Points')
        ax.scatter(cluster_mean[feature_idx], cluster_mean[feature_idx + 1], c=cluster_colors[cluster_idx], marker='x',
                   s=100, label=f'Cluster {cluster_idx} Mean')

    # scatter plot for cluster means
    ax.set_xlabel(worker_regression_cols[feature_idx])
    ax.set_ylabel(worker_regression_cols[feature_idx + 1])
    #ax.legend()


def plot_means_across_features(data, means, cluster_colors):
    num_features = len(means[0])
    num_rows = 2
    num_cols = (num_features + num_rows - 2) // num_rows  # calculate the number of columns dynamically
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

    for feature_idx in range(0, num_features - 1):
        row_idx = feature_idx // num_cols
        col_idx = feature_idx % num_cols
        plot_feature_scatter(data, means, feature_idx, axes[row_idx, col_idx], cluster_colors)

    plt.tight_layout()
    plt.show()


def plot_assignments_by_workplace_feature(workplace_data, assignments, feature_idx, ax, cluster_colors):
    num_clusters = len(np.unique(assignments))

    for cluster_idx in range(num_clusters):
        cluster_data = workplace_data[assignments == cluster_idx]
        ax.scatter(cluster_data[:, feature_idx], np.full_like(cluster_data[:, feature_idx], cluster_idx),
                   c=cluster_colors[cluster_idx], label=f'Cluster {cluster_idx} Assignments')

    ax.set_xlabel(worksite_regression_cols[feature_idx])
    ax.set_yticks(range(num_clusters))
    ax.set_yticklabels([f'Cluster {i}' for i in range(num_clusters)])
    ax.legend()
    plt.show()


if __name__ == "__main__":
    guide_params = torch.load('guide_params.pt')
    data = guide_params['beta_individual'].detach().numpy()[:100]  # out of 120, hold out 20
    data_test = guide_params['beta_individual'].detach().numpy()[100:]

    params = {'sigma': 0.00001, 'alpha_0': 1, 'eta': 0.1, 'K': 2, 'n': len(data), 'seed': 34, 'max_iterations': 400,
              'embedding_len': len(data[0])}
    likelihoods = []
    sigmas = [0.01, 0.001, 0.0001, 0.00001]

    for k in range(0, 4):
        # Define the parameters
        params['K'] = 2
        params['sigma'] = sigmas[k]
        assignments, means, proportions, log_joints = gibbs(data, params)

        plt.plot(log_joints)
        plt.xlabel('Iteration')
        plt.ylabel('Log Joint Probability')
        plt.title('Convergence Monitoring')
        plt.show()

        log_likelihood = held_out_log_likelihood(data_test, means, proportions, params)
        likelihoods.append(log_likelihood)
        print("likelihoods: ", likelihoods)

        cluster_colors = plt.cm.viridis(np.linspace(0, 1, k))
        plot_means_across_features(data, means, cluster_colors)

        worksite_data = pd.read_csv('WorksiteNetworkData.csv')[worksite_regression_cols].to_numpy()
        for i in range(len(worksite_regression_cols)):
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_assignments_by_workplace_feature(worksite_data, assignments, i, ax, cluster_colors)

# plot assignments vs workplace features

