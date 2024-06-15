import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
from load_data import *
from model import *

# Scenario 1: Unrestricted deforestation
def land_constraint_unrestricted(i, j, land_use_dict):
    if land_use_dict['waterways'][i, j] == 1:
        return 0
    return 1

# Scenario 2: Some policy, a certain percentage is protected
def land_constraint_policy_driven(i, j, land_use_dict, protection_percentage):
    if land_use_dict['natural'][i, j] == 1:
        if np.random.random() < protection_percentage:
            return 0
    if land_use_dict['waterways'][i, j] == 1:
        return 0
    return 1

# Scenario 3: Allow deforestation, but also allow non-urbanized cells to be reforested
def become_forest_probability(i, j, array, threshold=0.3):
    neighborhood = array[max(0, i-1):min(array.shape[0], i+2), max(0, j-1):min(array.shape[1], j+2)]
    num_forest_neighbors = np.sum(neighborhood == 1)
    if np.random.random() < threshold * num_forest_neighbors:
        return 1
    return 0

def land_constraint_partial_protection_with_growth(i, j, land_use_dict, protection_percentage, growth_threshold=0.3):
    if land_use_dict['natural'][i, j] == 1:
        if np.random.random() < protection_percentage:
            return 0 #there are some protection policies
    if land_use_dict['waterways'][i, j] == 1:
        return 0 #can't urbanize water
    if land_use_dict['natural'][i, j] == 0 and land_use_dict['waterways'][i, j] == 0: #allow non-urbanized cells that aren't water to become forest again
        if become_forest_probability(i, j, land_use_dict['natural'], growth_threshold):
            land_use_dict['natural'][i, j] = 1
            return 0
    return 1

# Code to run a certain scenario, based on the original model
def run_scenario(data_dict, proxy_dict, land_use_dict, first_year, last_year, coefficients, intercept, land_constraint_func, *args):
    first_value = 'data_' + str(first_year)
    last_value = 'data_' + str(last_year)

    start_array = data_dict[first_value]
    end_array = data_dict[last_value]

    necessary_development = np.sum(end_array) - np.sum(start_array)

    rows, columns = start_array.shape

    evaluation_list = []

    for i in range(rows):
        for j in range(columns):
            if proxy_dict['distance_to_Tiananmen_Square'][i, j] > 0 and start_array[i, j] == 0:
                evaluation_score = development_probability(i, j, start_array, proxy_dict, land_use_dict, coefficients, intercept)
                evaluation_list.append((evaluation_score, (i, j)))

        evaluation_list.sort(reverse=True, key=lambda x: x[0])

    # Copy start array to predicted array
    predicted_array = start_array.copy()

    # Track the number of developed cells
    developed_cells = 0

    # Develop cells based on evaluation and land constraint
    for score, (i, j) in evaluation_list:
        if developed_cells >= necessary_development:
            break  # Stop if the required number of cells is developed

        if land_constraint_func(i, j, land_use_dict, *args): #landuse_constraint_func returns 1 if cell can be developed, 0 if it can't
            predicted_array[i, j] = 1  # Urbanize the cell
            developed_cells += 1

    oa = calculate_overall_accuracy(predicted_array, end_array)
    fom = calculate_figure_of_merit(predicted_array, end_array)

    return predicted_array, oa, fom

# For evaluation and visualization
def evaluate_deforestation_rate(start_array, predicted_array, land_use_dict):
    forest_cells_start = np.sum((start_array == 0) & (land_use_dict['natural'] == 1))
    forest_cells_end = np.sum((predicted_array == 0) & (land_use_dict['natural'] == 1))
    if forest_cells_start == 0:
        return 0
    deforestation_rate = (forest_cells_start - forest_cells_end) / forest_cells_start
    return deforestation_rate

def visualize_results(start_array, predicted_array, land_use_dict, title):
    color_map_start = np.zeros((start_array.shape[0], start_array.shape[1], 3), dtype=np.uint8)
    color_map_predicted = np.zeros((predicted_array.shape[0], predicted_array.shape[1], 3), dtype=np.uint8)

    # Define colors
    urban_color = [169, 169, 169]  # Gray
    forest_color = [34, 139, 34]  # Green
    water_color = [0, 0, 255]  # Blue

    # Populate the color map for start_array
    for i in range(start_array.shape[0]):
        for j in range(start_array.shape[1]):
            if start_array[i, j] == 1:
                color_map_start[i, j] = urban_color
            elif land_use_dict['waterways'][i, j] == 1:
                color_map_start[i, j] = water_color
            elif land_use_dict['natural'][i, j] == 1:
                color_map_start[i, j] = forest_color

    # Populate the color map for predicted_array
    for i in range(predicted_array.shape[0]):
        for j in range(predicted_array.shape[1]):
            if predicted_array[i, j] == 1:
                color_map_predicted[i, j] = urban_color
            elif land_use_dict['waterways'][i, j] == 1:
                color_map_predicted[i, j] = water_color
            elif land_use_dict['natural'][i, j] == 1:
                color_map_predicted[i, j] = forest_color

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(color_map_start, interpolation='nearest')
    axs[0].set_title('Start Year')
    axs[1].imshow(color_map_predicted, interpolation='nearest')
    axs[1].set_title('Predicted End Year')
    fig.suptitle(title)
    plt.show()

# Running all scenarios
def run_all_scenarios():
    shapefiles = ['natural', 'waterways']
    data_dict, feature_dict, landuse_dict = generate_data(True, shapefiles, normalization_method='min-max')

    first_year = 1984
    last_year = 2013

    coefficients, intercept = find_coef_and_intercept(data_dict, feature_dict, first_year, last_year)

    # Scenario 1: Unrestricted Urbanization into Forests
    predicted_array, oa, fom = run_scenario(data_dict, feature_dict, landuse_dict, first_year, last_year, coefficients, intercept, land_constraint_unrestricted)
    deforestation_rate = evaluate_deforestation_rate(data_dict['data_' + str(first_year)], predicted_array, landuse_dict)
    visualize_results(data_dict['data_' + str(first_year)], predicted_array, landuse_dict, f'Unrestricted Urbanization\nOA: {oa:.4f}, FoM: {fom:.4f}, Deforestation Rate: {deforestation_rate:.4f}')

    # Scenario 2: Policy-Driven Forest Protection
    protection_percentages = [0.25, 0.50, 1.00]
    for protection_percentage in protection_percentages:
        predicted_array, oa, fom = run_scenario(data_dict, feature_dict, landuse_dict, first_year, last_year, coefficients, intercept, land_constraint_policy_driven, protection_percentage)
        deforestation_rate = evaluate_deforestation_rate(data_dict['data_' + str(first_year)], predicted_array, landuse_dict)
        visualize_results(data_dict['data_' + str(first_year)], predicted_array, landuse_dict, f'Policy-Driven Forest Protection ({int(protection_percentage * 100)}%)\nOA: {oa:.4f}, FoM: {fom:.4f}, Deforestation Rate: {deforestation_rate:.4f}')

    # Scenario 3: Partial Forest Protection with Development Constraints and Growth
    for protection_percentage in protection_percentages:
        predicted_array, oa, fom = run_scenario(data_dict, feature_dict, landuse_dict, first_year, last_year, coefficients, intercept, land_constraint_partial_protection_with_growth, protection_percentage)
        deforestation_rate = evaluate_deforestation_rate(data_dict['data_' + str(first_year)], predicted_array, landuse_dict)
        visualize_results(data_dict['data_' + str(first_year)], predicted_array, landuse_dict, f'Partial Forest Protection with Growth ({int(protection_percentage * 100)}%)\nOA: {oa:.4f}, FoM: {fom:.4f}, Deforestation Rate: {deforestation_rate:.4f}')

run_all_scenarios()
