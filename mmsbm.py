import stan
import pandas as pd
from sklearn.preprocessing import StandardScaler
import arviz as az
import matplotlib.pyplot as plt
import pickle
import numpy as np


stan_code = """
data {
  // Define data input
  int<lower=1> n;           // Number of worksites
  int<lower=1> p_group;     // Number of worksite-level input features
  int<lower=1> p_individual; // Number of individual-level input features
  int<lower=1> num_workers;           // Number of workers

  matrix[n, p_group] x_group;                // Worksite-level feature vectors
  matrix[num_workers, p_individual] x_individual; // Worker-level feature vectors (pad to max_mi)
  int<lower=0, upper=1> y[num_workers];          // Binary response variable
  int<lower=1> mi[n];                         // Number of workers in each worksite
}

parameters {
  // Define model parameters
  vector<lower=0>[p_group] tau_group;
  vector<lower=0>[p_individual] tau_individual;
  
  real<lower=0> delta_group;           // Hyperprior for group-level standard deviations
  real<lower=0> delta_individual;   // Hyperprior for individual-level standard deviations

  matrix[p_individual, n] beta_individual; // Coefficients for individual-level features
  matrix[p_group, n] beta_group;           // Coefficients for group-level features
}

model {
  // Hyperpriors
  tau_group ~ cauchy(0, 5);
  tau_individual ~ cauchy(0,5);
  
  delta_group ~ gamma(tau_group, 1);     // problems being 0
  delta_individual ~ gamma(tau_individual, 1);   //problems being inf
  
  
  // Priors
  for (i in 1:n) {
    //beta_group[:, i] ~ normal(0, delta_group);
    //beta_individual[:, i] ~ normal(0, delta_individual);
    beta_group[:, i] ~ normal(0, 1);
    beta_individual[:, i] ~ normal(0, 1);
  }


  // Hierarchical logistic regression likelihood
  for (i in 1:n) {
    real eta_ij;
    for (j in 1:mi[i]) {
      int index;
      index = i + j - 1;
      eta_ij = dot_product(x_individual[index, :], beta_individual[:, i])
                   + dot_product(x_group[i, :], beta_group[:, i]);
      y[index] ~ bernoulli_logit(eta_ij);
      //log_lik[index] = bernoulli_logit_lpmf(y[index] | eta_ij);
    }
  }
  // target += sum(log_lik)
}

generated quantities {
  vector[num_workers] log_lik_generated;
  vector[num_workers] y_generated;

  // Generate log likelihood for LOO calculation
  for (i in 1:num_workers) {
    real eta_ij_generated;
    eta_ij_generated = dot_product(x_individual[i, :], beta_individual[:, 1])
                       + dot_product(x_group[1, :], beta_group[:, 1]);
    log_lik_generated[i] = bernoulli_logit_lpmf(y[i] | eta_ij_generated);
    y_generated[i] = bernoulli_logit_rng(eta_ij_generated);
  }
  
    // Add observed_data group
  vector[num_workers] observed_data;
  for (i in 1:num_workers) {
    observed_data[i] = y[i];
  }
}

"""

worker_df = pd.read_csv('WorkerRegressionData.csv')
worksite_df = pd.read_csv('WorksiteNetworkData.csv')

worker_regression_cols = ['Is Male', 'Time To First Convo (Quart. 1)',
       'Time To First Convo (Quart. 2)', 'Time To First Convo (Quart. 3)',
       'Time To First Convo (Quart. 4)',
       'Log(Eigen. Centrality)', 'No. Notes']   # some of these should probably not be normal eventually- ex no. notes poisson

worksite_regression_cols = [
       # 'Percent black in ZIP',
       # 'Percent Latino in ZIP', 'Percent male',
       # 'Log(Mean AGI in ZIP)',
       'NDO (Eigen., Pearson)',
       # 'Mobilizing (Pearson)',
       # 'Campaign length (Quint. 1)', 'Campaign length (Quint. 2)',
       # 'Campaign length (Quint. 3)', 'Campaign length (Quint. 4)',
       # 'Campaign length (Quint. 5)',
       # 'Log(Workers signed per worker-week)',
       # 'Log(Workers signed per organizer convers.)',
       'Log(No. organizer convers.)',
       #  'Log(No. workers contacted)',
       # 'Log(No. workers discovered)', 'Log(No. workers signed)',
       # 'Log(No. edges)',
       'Log(Mean degree)',
       # 'Log(Convers. per worker variance)',
       # 'Log(Centrality variance)',
       # 'Log(Centrality mean)'
                            ]

# for normalization
scaler = StandardScaler()

worker_data = scaler.fit_transform(worker_df[worker_regression_cols].to_numpy())
worksite_data = scaler.fit_transform(worksite_df[worksite_regression_cols].to_numpy())
worksite_counts = worker_df['Worksite ID'].value_counts()[worker_df['Worksite ID'].unique()].values
print("worksite counts")
print(worksite_counts)
print(len(worksite_counts))

outcomes = worker_df['Signed'].to_numpy()

for index, value in enumerate(worker_df['Worksite ID'].unique()):
    if value not in worksite_df['Worksite ID'].unique():
        print('value', value)
        print('index', index)

print(worker_data)
print(worksite_data)
print(outcomes)

num_workers = len(worker_data)
num_worksites = len(worksite_data)
num_worker_predictors = len(worker_data[0])
num_worksite_predictors = len(worksite_data[0])

data_dict = {
    # 'N': num_workers,
    'p_individual': num_worker_predictors,
    'n': num_worksites,
    'p_group': num_worksite_predictors,
    'mi': worksite_counts,
    'x_individual': worker_data,
    'x_group': worksite_data,
    'y': outcomes,
    'num_workers': num_workers
}


#compile model
model = stan.build(stan_code, data_dict)

print("fitting now!")
stan_fit = model.sample(num_chains=4, num_samples=500)

# Print the summary of the results
print(stan_fit)


# df = az.summary(stan_fit, var_names=["beta_group", "beta_individual", "delta_individual", "delta_group", "tau_group", "tau_individual"]).reset_index() # used to include delta group and delta individual too
# df.to_csv('stan_results.csv', index=False)

idata = az.from_pystan(stan_fit, log_likelihood=["log_lik_generated"], posterior_predictive=["y_generated"], observed_data=["y"])

# Save the InferenceData object to a file (optional)
#az.to_netcdf(idata, "idata.nc", chunks={"dim1": 100, "dim2": 100})

# Assuming you have the fitted model stored in 'stan_fit'
# and the InferenceData object stored in 'idata'

# 1. Visual Inspection of Trace Plots
az.plot_trace(stan_fit)
plt.show()

# 2. Summary Statistics
print(stan_fit)

# 3. Posterior Predictive Checks (PPC)
az.plot_ppc(idata, data_pairs={"y": "y_generated"})
plt.show()

# 4. Autocorrelation Plots
az.plot_autocorr(stan_fit)
plt.show()

# 5. Gelman-Rubin Statistic (R-hat)
print(az.rhat(stan_fit))

# 6. Effective Sample Size (ESS)
print(az.ess(stan_fit))

# 7. LOO (Leave-One-Out) Cross-Validation
loo = az.loo(stan_fit, pointwise=True)
az.plot_loo_pit(loo)
plt.show()

# 8. Other Diagnostic Plots
az.plot_posterior(stan_fit, kind='kde')
plt.show()


