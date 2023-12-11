# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 23:16:37 2023

@author: Ke
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def regress_av_model(file_path):
    def calculate_acceleration(speed, delta_t):
        return np.diff(speed) / delta_t

    # Load the data
    data = pd.read_csv(file_path)

    # Initialize variables
    delta_t = 0.1  # seconds
    regression_results = {}

    # Regression analysis for each vehicle
    for i in range(2, 6):
        data[f'DeltaV{i}'] = data[f'Speed{i-1}'] - data[f'Speed{i}']
        dv = data[f'Speed{i}'].values
        vf = data[f'Speed{i}'].values
        acc_f = calculate_acceleration(vf, delta_t)
        ivs = data[f'IVS{i-1}'].values[:-1]

        X = np.column_stack((ivs, data[f'DeltaV{i}'][:-1], vf[:-1]))
        y = acc_f
        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        n = len(y)
        p = X.shape[1]
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

        regression_results[f'AV{i}'] = {
            'Coefficients': model.coef_,
            'Intercept': model.intercept_,
            'R2': r2,
            'Adjusted R2': adjusted_r2
        }

    return regression_results

def regress_av_model_with_delay(file_path):

    def calculate_acceleration(speed, delta_t):
        return np.diff(speed) / delta_t

    # Load the data
    data = pd.read_csv(file_path)

    # Initialize variables
    delta_t = 0.1  # seconds

    regression_results_all_delay = {}
    for delay in range(0, 51):
        regression_results_one_delay = {}
        # Regression analysis for each vehicle
        for i in range(2, 6):
            dv = data[f'Speed{i - 1}'] - data[f'Speed{i}']
            data[f'DeltaV{i}'] = dv
            vf = data[f'Speed{i}'].values
            acc_f = calculate_acceleration(vf, delta_t)
            vf = vf[:-1]
            dv = dv[:-1]
            ivs = data[f'IVS{i - 1}'].values[:-1]
            X = np.column_stack((ivs, dv, vf))
            y = acc_f

            # dv_delay = dv[delay:]
            # vf_delay = vf[delay:]
            acc_f_delay = acc_f[delay:]
            if delay == 0:
                dv_delay = dv
                vf_delay = vf
                ivs_delay = ivs
            else:
                dv_delay = dv[:(-1)*delay]
                vf_delay = vf[:(-1)*delay]
                ivs_delay = ivs[:(-1)*delay]
            X_delay = np.column_stack((ivs_delay, dv_delay, vf_delay))
            y_delay = acc_f_delay

            model = LinearRegression()
            model.fit(X_delay, y_delay)
            y_pred = model.predict(X_delay)
            r2 = r2_score(y_delay, y_pred)
            n = len(y_delay)
            p = X_delay.shape[1]
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            regression_results_one_delay[f'AV{i}'] = {
                'Coefficients': model.coef_,
                'Intercept': model.intercept_,
                'R2': r2,
                'Adjusted R2': adjusted_r2
            }
        regression_results_all_delay[delay] = regression_results_one_delay

    return regression_results_all_delay

def calculate_string_stability(regression_results):
    string_stability_results = {}
    for i in range(2, 6):
        f_d, f_delta_v, f_v = regression_results[f'AV{i}']['Coefficients']
        string_stability = 0.5 - f_delta_v / f_v - f_d / (f_v ** 2)
        string_stability_results[f'AV{i}'] = string_stability

    return string_stability_results

def calculate_string_stability_with_delay(regression_results_delay):
    string_stability_result1 = {}
    string_stability_result2 = {}
    for i in range(2, 6):
        f_d, f_delta_v, f_v = regression_results[f'AV{i}']['Coefficients']
        string_stability1 =  f_v ** 2 - 2*f_d - 2*f_v*f_delta_v
        string_stability2 = 1 + 2*tau*(f_v-f_delta_v) + tau** 2*f_d
        string_stability_results1[f'AV{i}'] = string_stability1
        string_stability_results2[f'AV{i}'] = string_stability2
    return string_stability_results1, string_stability_results2

def simulate_avs(regression_coefficients, total_time=200, time_step=0.1):
    # Constants and initial conditions
    lead_speed_initial = 20  # m/s
    lead_speed_final = 25  # m/s
    speed_change_time = 100  # seconds
    initial_distance = 40  # meters
    initial_speed = 20  # m/s
    num_vehicles = 5  # Including the lead vehicle

    # Time array
    times = np.arange(0, total_time, time_step)

    # Initialize arrays for speed and position
    speeds = np.zeros((num_vehicles, len(times)))
    positions = np.zeros((num_vehicles, len(times)))

    # Set initial conditions
    speeds[:, 0] = initial_speed
    positions[:, 0] = -np.arange(0, num_vehicles * initial_distance, initial_distance)

    # Simulation loop
    for t in range(1, len(times)):
        # Update lead vehicle speed and position
        if times[t] > speed_change_time:
            speeds[0, t] = lead_speed_final
        else:
            speeds[0, t] = lead_speed_initial

        # Calculate position for the lead vehicle using kinematic equation
        # if times[t] <= speed_change_time:
        #     positions[0, t] = positions[0, t-1] + speeds[0, t] * time_step
        # else:
        #     acceleration_lead = (lead_speed_final - lead_speed_initial) / time_step
        #     positions[0, t] = positions[0, t-1] + speeds[0, t-1] * time_step + 0.5 * acceleration_lead * time_step**2
        positions[0, t] = positions[0, t - 1] + speeds[0, t] * time_step
        # Update positions and speeds of following vehicles
        for i in range(1, num_vehicles):
            # Current speed and position
            current_speed = speeds[i, t-1]
            current_position = positions[i, t-1]

            # Lead vehicle's speed and position
            lead_speed = speeds[i-1, t]
            lead_position = positions[i-1, t]

            # Calculate the IVS and DeltaV
            ivs = lead_position - current_position
            delta_v = lead_speed - current_speed

            # Using the regression model to update the speed
            acceleration = (regression_coefficients[0] * ivs + 
                            regression_coefficients[1] * delta_v + 
                            regression_coefficients[2] * current_speed + 
                            regression_coefficients[3])
            new_speed = current_speed + acceleration * time_step
            new_position = current_position + new_speed * time_step + 0.5 * acceleration * time_step**2

            # Update the arrays
            speeds[i, t] = new_speed
            positions[i, t] = new_position

    return times, speeds, positions

def simulate_avs_delay_difpara(regression_coefficients, delay_list, total_time=200, time_step=0.1):
    # Constants and initial conditions
    lead_speed_initial = 20  # m/s
    lead_speed_final = 25  # m/s
    speed_change_time = 100  # seconds
    initial_distance = 40  # meters
    initial_speed = 20  # m/s
    num_vehicles = 5  # Including the lead vehicle
    delay_list = [0] + delay_list # first vehicle 0

    # Time array
    times = np.arange(0, total_time, time_step)

    # Initialize arrays for speed and position
    speeds = np.zeros((num_vehicles, len(times)))
    positions = np.zeros((num_vehicles, len(times)))

    # Set initial conditions
    speeds[:, 0] = initial_speed
    positions[:, 0] = -np.arange(0, num_vehicles * initial_distance, initial_distance)

    # Simulation loop
    for t in range(1, len(times)):
        # Update lead vehicle speed and position
        if times[t] > speed_change_time:
            speeds[0, t] = lead_speed_final
        else:
            speeds[0, t] = lead_speed_initial

        positions[0, t] = positions[0, t - 1] + speeds[0, t] * time_step
        # Update positions and speeds of following vehicles
        for i in range(1, num_vehicles):
            # Current speed and position
            current_speed = speeds[i, t-1]
            current_position = positions[i, t-1]
            current_speed_delay = speeds[i, t-1-delay_list[i]]
            current_position_delay = positions[i, t-1-delay_list[i]]

            # Lead vehicle's speed and position
            lead_speed_delay = speeds[i-1, t-delay_list[i]]
            lead_position_delay = positions[i-1, t-delay_list[i]]

            # Calculate the IVS, delta_v
            ivs_delay = lead_position_delay - current_position_delay
            delta_v_delay = lead_speed_delay - current_speed_delay

            # Using the regression model to update the speed
            acceleration = (regression_coefficients[i][0] * ivs_delay +
                            regression_coefficients[i][1] * delta_v_delay +
                            regression_coefficients[i][2] * current_speed_delay +
                            regression_coefficients[i][3])
            new_speed = current_speed + acceleration * time_step
            new_position = current_position + new_speed * time_step + 0.5 * acceleration * time_step**2

            # Update the arrays
            speeds[i, t] = new_speed
            positions[i, t] = new_position

    return times, speeds, positions

def plot_av_speeds(times, speeds):
    line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1))]
    colors = ['black', 'green', 'red', 'cyan', 'blue']

    plt.figure(figsize=(12, 8))
    for i in range(speeds.shape[0]):
        # plt.plot(times, speeds[i], label=f'Vehicle {i} - Base AV5' if i > 0 else 'Lead Vehicle',
        plt.plot(times, speeds[i], label=f'Vehicle {i}' if i > 0 else 'Lead Vehicle',
                 linestyle=line_styles[i % len(line_styles)], color=colors[i % len(colors)], linewidth=2.5)

    # Setting font sizes
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Speed (m/s)', fontsize=16)
    plt.title('Speed of Vehicles Over Time', fontsize=16)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.show()

file_path = 'ASta_test.csv'
# regression_results = regress_av_model(file_path)
# vehicle_models = [regression_results[f'AV5']]
# stability_results = calculate_string_stability(regression_results)
# av_model = vehicle_models[0]
# av_coefficients = list(av_model['Coefficients'])
# av_intercept = av_model['Intercept']
# av_parameters = av_coefficients + [av_intercept]
# times, speeds, positions = simulate_avs(av_parameters)
# plot_av_speeds(times, speeds)

regression_results_all_delay = regress_av_model_with_delay(file_path)
delay_list = list(regression_results_all_delay.keys())
veh_list = list(regression_results_all_delay[delay_list[0]].keys())
# regression_results_all_delay_dict = {}
# for delay in delay_list:
#     one_delay_regression = {}
#     for veh in veh_list:
#         one_delay_regression[veh] = regression_results_all_delay[delay][veh]['Adjusted R2']
#     regression_results_all_delay_dict[delay] = one_delay_regression
# regression_coefficients = []
# for veh in veh_list:
#
# regression_results = regression_results_all_delay[14]
# vehicle_models = [regression_results[f'AV5']]
# stability_results = calculate_string_stability(regression_results)
delay_regression = {'AV2': 12, 'AV3': 19, 'AV4': 13, 'AV5': 14}
regression_cofficient = []
regression_cofficient.append([])
# for veh in veh_list:
#     dict_temp = regression_results_all_delay[delay_regression[veh]][veh]
#     one_regression_cofficient = list(dict_temp['Coefficients']) + [dict_temp['Intercept']]
#     regression_cofficient.append(one_regression_cofficient)
veh = 'AV5'
for i in range(len(veh_list)):
    dict_temp = regression_results_all_delay[delay_regression[veh]][veh]
    one_regression_cofficient = list(dict_temp['Coefficients']) + [dict_temp['Intercept']]
    regression_cofficient.append(one_regression_cofficient)
times, speeds, positions = simulate_avs_delay_difpara(regression_cofficient, list(delay_regression.values()))

# origin_data = pd.read_csv(file_path)
# times = np.array(origin_data['Time'])
# speeds_df = origin_data[['Speed1', 'Speed2', 'Speed3', 'Speed4', 'Speed5']]
# speeds = speeds_df.to_numpy().T


plot_av_speeds(times, speeds)


badebug = 1