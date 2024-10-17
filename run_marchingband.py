from dsarf import DSARF
from dsarf import compute_NRMSE
from concurrent.futures import ProcessPoolExecutor
from .marchingband_data.run_sim import generate_training_data
import numpy as np
import torch
import json


def ind_run(N, J, T, seed, factor_dim, L): 
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    GLOBAL_MSG = "LAUGH" * N
    total_data = generate_training_data(GLOBAL_MSG, J, T, 0)
    data = total_data[0]
    
    new_T = T // N  
    conctat_data = data.reshape(N, new_T, J * D)

    # Create and train the model
    dsarf = DSARF(D, factor_dim=factor_dim, L=L, S=6)
    model_train = dsarf.fit(conctat_data, epoch_num=500)

    # Extract posterior summary
    posterior_summary = model_train.q_S
    
    return posterior_summary

def save_results(results, seed, factor_dim, L):
    # Save results as a JSON file
    filename = f'run_{seed}_{factor_dim}_{L}.json'
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == '__main__':
    D = 2
    N = 10
    J = 64
    T = 200
    seeds = range(120, 130)
    factor_dims = range(10)
    lags = [list(range(1, i + 1)) for i in range(1, 11)]

    # Use parallel processing to speed up the task
    with ProcessPoolExecutor() as executor:
        futures = []
        for seed in seeds:
            for L in lags:
                for factor_dim in factor_dims:
                    futures.append(executor.submit(ind_run, N, J, T, seed, factor_dim, L))
        
        # Collect and save results as they finish
        for i, future in enumerate(futures):
            seed = seeds[i // (len(factor_dims) * len(lags))]
            factor_dim = factor_dims[(i // len(lags)) % len(factor_dims)]
            L = lags[i % len(lags)]
            results = future.result()
            save_results(results, seed, factor_dim, L)              