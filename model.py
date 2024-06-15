# %%
import numpy as np
from load_data import *
from Linear_prediction import *
from Linear_prediction2 import *

#%%
#Define the logistic CA model

#calculate suitability
def calculate_suitability(i,j, coefficients, intercept, proxy_dict):
    z_ij = linear_predictor(i, j, coefficients, intercept, proxy_dict)
    probability = 1 / (1 + np.exp(-z_ij))
    return probability

def neighborhood_density(array, i, j, neighborhood_size=5):
    """
    Calculate the density of 1s around a specific cell in a numpy array,
    excluding the cell itself from the calculation.

    :param array: numpy array
    :param i: row index of the cell
    :param j: column index of the cell
    :param neighborhood_size: size of the neighborhood (must be an odd number)
    :return: density of 1s in the neighborhood
    """
    # Ensure the neighborhood_size is an odd number
    if neighborhood_size % 2 == 0:
        raise ValueError("neighborhood_size must be an odd number")

    # Calculate the half size
    half_size = neighborhood_size // 2

    # Calculate the start and end indices for rows and columns
    row_start = max(0, i - half_size)
    row_end = min(array.shape[0], i + half_size + 1)
    col_start = max(0, j - half_size)
    col_end = min(array.shape[1], j + half_size + 1)

    # Extract the neighborhood
    neighborhood = array[row_start:row_end, col_start:col_end]

    # Calculate the number of 1s and exclude the central cell if it's in the range
    if 0 <= i - row_start < neighborhood.shape[0] and 0 <= j - col_start < neighborhood.shape[1]:
        original_value = neighborhood[i - row_start, j - col_start]
        neighborhood[i - row_start, j - col_start] = 0
        num_ones = np.sum(neighborhood == 1)
        neighborhood[i - row_start, j - col_start] = original_value
        total_cells = neighborhood.size - 1
    else:
        num_ones = np.sum(neighborhood == 1)
        total_cells = neighborhood.size

    # Calculate the density of 1s
    density = num_ones / total_cells if total_cells > 0 else 0

    return density


def linear_predictor(i, j, coefficients, intercept, proxy_dict):
    # Start with the intercept
    linear_pred = intercept
    
    # proxy keys as input
    proxy_keys = proxy_dict.keys()

    # Calculate the linear predictor
    for idx, key in enumerate(proxy_keys):
        if idx < len(coefficients):
            linear_pred += coefficients[idx] * proxy_dict[key][i, j]
        else:
            raise IndexError(f"Index {idx} out of range for coefficients list.")
    
    return linear_pred

def stochastic_perturbation():
    alpha = 1
    lambda_val = np.random.uniform(1e-10, 1)
    return 1 + (-np.log(lambda_val)) ** alpha

def land_constraint(i, j, land_use_dict):
    """
    Check land use constraints. Return 0 if the land use natural or the land use water has a value of 1 at the given location.
    Return 1 otherwise.
    
    Parameters:
    i (int): Row index of the cell.
    j (int): Column index of the cell.
    land_use_dict (dict): Dictionary containing land use arrays.
    
    Returns:
    int: 0 if the land use natural or the land use water has a value of 1, 1 otherwise.
    """
    for land_use, array in land_use_dict.items():
        if array[i, j] == 1:
            return 0
    return 1

def development_probability(i, j, array, proxy_dict, land_use_dict, coefficients, intercept):
    pg_ij = calculate_suitability(i,j, coefficients, intercept, proxy_dict)
    omega_ij = neighborhood_density(array, i, j)
    ra_ij = stochastic_perturbation()
    land_ij = land_constraint(i,j, land_use_dict)
    prob_development = pg_ij * omega_ij * ra_ij * land_ij
    return prob_development

def calculate_overall_accuracy(predicted_array, actual_array):
    # Calculate the number of correctly classified cells
    correctly_classified = np.sum(predicted_array == actual_array)
    # Calculate the total number of cells
    total_cells = predicted_array.size
    # Calculate overall accuracy
    overall_accuracy = correctly_classified / total_cells
    return overall_accuracy

def calculate_figure_of_merit(predicted_array, actual_array):
    # Identify changes between actual and predicted arrays
    change_predicted = (predicted_array != actual_array).astype(int)
    change_actual = (actual_array != np.roll(actual_array, shift=1, axis=0)).astype(int) | \
                    (actual_array != np.roll(actual_array, shift=1, axis=1)).astype(int)
    
    # Calculate areas
    A = np.sum((change_predicted == 1) & (change_actual == 1))  # Correctly predicted change
    B = np.sum((change_predicted == 0) & (change_actual == 1))  # Observed change but predicted non-change
    C = np.sum((change_predicted == 1) & (change_actual == 0))  # Predicted change but no change occurred
    D = np.sum((change_predicted == 0) & (change_actual == 0))  # Correctly predicted non-change
    
    # Calculate figure of merit
    if A + B + C + D == 0:
        figure_of_merit = 0
    else:
        figure_of_merit = A / (A + B + C + D)
    
    return figure_of_merit

def run_model(data_dict, proxy_dict, land_use_dict, first_year, last_year, coefficients, intercept):
    first_value = 'data_' + str(first_year)
    last_value = 'data_' + str(last_year)

    # Get arrays from start and end years
    start_array = data_dict[first_value]
    end_array = data_dict[last_value]

    necessary_development = np.sum(end_array) - np.sum(start_array)

    rows, columns = start_array.shape

    evaluation_list = []

    # Iterate over all cells in the start_array
    for i in range(rows):
        for j in range(columns):
            # Check the conditions
            if feature_dict['distance_to_Tiananmen_Square'][i, j] > 0 and start_array[i, j] == 0:
                # Evaluate the cell using development_probability function
                evaluation_score = development_probability(i, j, start_array, proxy_dict, land_use_dict, coefficients, intercept)

                # Store the score along with its coordinates
                evaluation_list.append((evaluation_score, (i, j)))

    # Sort the evaluations based on the scores in descending order
    evaluation_list.sort(reverse=True, key=lambda x: x[0])

    # Determine the number of cells to develop
    num_cells_to_develop = int(necessary_development)

    # Change the necessary amount of cells from 0 to 1 based on the highest evaluation scores
    predicted_array = start_array.copy()
    for idx in range(num_cells_to_develop):
        score, (i, j) = evaluation_list[idx]
        predicted_array[i, j] = 1

    oa = calculate_overall_accuracy(predicted_array, end_array)
    fom = calculate_figure_of_merit(predicted_array, end_array)

    return predicted_array, oa, fom


def run_model_with_intermediate_years(data_dict, proxy_dict, land_use_dict, first_year, last_year, coefficients, intercept):
    current_year = first_year
    next_year = current_year + 1
    
    # Initialize the start array
    current_array = data_dict['data_' + str(current_year)].copy()

    while next_year <= last_year:
        print(f"Processing year: {current_year} to {next_year}")

        # Get the target array for the next year
        target_array = data_dict['data_' + str(next_year)]

        necessary_development = np.sum(target_array) - np.sum(current_array)

        rows, columns = current_array.shape

        evaluation_list = []

        # Iterate over all cells in the current_array
        for i in range(rows):
            for j in range(columns):
                # Check the conditions
                if proxy_dict['distance_to_Tiananmen_Square'][i, j] > 0 and current_array[i, j] == 0:
                    # Evaluate the cell using development_probability function
                    evaluation_score = development_probability(i, j, current_array, proxy_dict, land_use_dict, coefficients, intercept)

                    # Store the score along with its coordinates
                    evaluation_list.append((evaluation_score, (i, j)))

        # Sort the evaluations based on the scores in descending order
        evaluation_list.sort(reverse=True, key=lambda x: x[0])

        # Determine the number of cells to develop
        num_cells_to_develop = int(necessary_development)

        # Change the necessary amount of cells from 0 to 1 based on the highest evaluation scores
        for idx in range(num_cells_to_develop):
            score, (i, j) = evaluation_list[idx]
            current_array[i, j] = 1

        # Move to the next year
        current_year = next_year
        next_year += 1

    # The final predicted array is the updated current_array
    predicted_array = current_array

    # Calculate evaluation metrics based on the last year
    end_array = data_dict['data_' + str(last_year)]
    oa = calculate_overall_accuracy(predicted_array, end_array)
    fom = calculate_figure_of_merit(predicted_array, end_array)

    return predicted_array, oa, fom
#%%

shapefiles = ['natural', 'waterways']
data_dict, feature_dict, landuse_dict = generate_data(True, shapefiles, normalization_method='z-score')

first_year = 1984
last_year = 2013

coefficients, intercept = find_coef_and_intercept(data_dict, feature_dict, first_year, last_year)
#coefficients, intercept = find_coef_and_intercept2(data_dict, feature_dict, first_year, last_year)
#coefficients = 
#predicted_array, oa, fom = run_model(data_dict, feature_dict, landuse_dict, first_year, last_year, coefficients, intercept)
predicted_array, oa, fom = run_model_with_intermediate_years(data_dict, feature_dict, landuse_dict, first_year, last_year, coefficients, intercept)
# %%
print(oa, fom)
