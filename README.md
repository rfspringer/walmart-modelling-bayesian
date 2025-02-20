# Labor Movement Analysis: Predicting Union Membership Card Signing

## Overview

This project analyzes workplace organizing dynamics using data from the 2010-2015 OUR Walmart organizing campaign. The study employs a Bayesian hierarchical model to understand what factors influence workers' decisions to sign union membership cards, with a focus on both individual and workplace-level characteristics.

## Key Findings

- Workplace-level features are stronger predictors of card signing than individual attributes
- Key positive correlations with card signing:
  - Network-driven organizing (NDO)
  - Number of organizer conversations
  - High average workplace centrality
- Key negative correlations:
  - Mid-length campaigns
  - Large worksites
  - High Latino population in workplace ZIP code
- Large worksites showed distinct dynamics where organizer attention to individual workers was particularly important

## Data

The analysis uses two primary datasets:
- `WorkerRegressionData.csv`: Individual worker data including:
  - Gender
  - Time to first conversation quartile
  - Eigenvector centrality
  - Number of organizer notes
  - Signing status (outcome variable)

- `WorksiteNetworkData.csv`: Workplace-level data including:
  - Demographic information (ZIP code level)
  - Network metrics
  - Campaign characteristics
  - Organizing strategy measures

## Model

### Hierarchical Bayesian Model

The project implements a Bayesian hierarchical logistic regression model using Pyro:

```python
def model(worker_data, worksite_data, worksite_indices, outcomes):
    num_individual_predictors = worker_data.shape[1]
    num_group_predictors = worksite_data.shape[1]

    # Scale parameters
    beta_individual_scale = pyro.sample('beta_individual_scale', dist.HalfCauchy(5))
    beta_group_scale = pyro.sample('beta_group_scale', dist.HalfCauchy(5))

    # Group-level coefficients
    beta_group = pyro.sample('beta_group', 
        dist.Normal(0, beta_group_scale).expand([num_group_predictors]).to_event(1))

    # Mean intercept
    alpha = pyro.sample('alpha_mean', dist.Normal(0, beta_group_scale))

    # Individual-level coefficients per worksite
    with pyro.plate('worksite', len(worksite_data)):
        beta_individual = pyro.sample('beta_individual', 
            dist.Normal(0, beta_individual_scale).expand([num_individual_predictors]).to_event(1))

    # Combined logits
    logits = alpha + torch.matmul(beta_group, worksite_data.T).sum(dim=-1) \
            + torch.matmul(beta_individual[worksite_indices], worker_data.T).sum(dim=-1)

    # Observations
    with pyro.plate('worker', len(worker_data)):
        y = pyro.sample('y', dist.Bernoulli(logits=logits), obs=outcomes)
```

### Gaussian Mixture Model Analysis

After the hierarchical regression, a Gaussian Mixture Model (GMM) was employed to identify clusters in workplace-specific coefficients. The GMM analysis revealed important patterns in organizing dynamics:

1. **Clustering Implementation**:
   - Used a Gibbs sampler for parameter inference
   - Optimal parameters found through experimentation:
     - σ = 0.0001 (variance parameter)
     - α₀ = 1 (Dirichlet concentration)
     - η = 0.1 (prior scale)
     - K = 2 (number of clusters)
   - Convergence achieved after 400 iterations

2. **Key Findings from GMM**:
   - Identified two distinct clusters of workplaces
   - Main differentiating factors:
     - Number of workers contacted
     - Number of workers discovered
   - Larger worksites showed different organizing dynamics:
     - Stronger correlation between organizer notes and card signing
     - Earlier contact timing more crucial for success
     - Individual attention from organizers more important

3. **Cluster Characteristics**:
   - First cluster: Smaller worksites with more uniform organizing patterns
   - Second cluster: Larger worksites requiring more targeted organizer attention
   - Network metrics (centrality, edge count) distributed similarly across clusters

4. **Implications for Organizing**:
   - Different organizing strategies needed based on workplace size
   - Large workplaces require more focused individual outreach
   - Timing of initial contact more critical in larger workplaces

## Results

The model achieved:
- Accuracy: 0.9356
- AUC: 0.7688

## Post-Processing Analysis

The project includes additional analysis tools:
- Feature importance analysis across workplace sizes
- Visualization tools for posterior predictive checks
- Comparative analysis of cluster-specific organizing patterns

## Future Work

Potential areas for expansion:
1. Additional measurements of workplace characteristics
2. More complex parameterization beyond mean-field
3. Analysis of organizing dynamics across different workplace scales
4. Investigation of temporal aspects of organizing campaigns
5. Extension of GMM analysis to include more workplace features
6. Development of cluster-specific organizing recommendations

## Files

- `regression.py`: Main model implementation and training
- `clustering.py`: Post-processing analysis and GMM clustering
- `WorkerRegressionData.csv`: Individual worker data
- `WorksiteNetworkData.csv`: Workplace-level data

Course project for STAT 6701, Columbia University
