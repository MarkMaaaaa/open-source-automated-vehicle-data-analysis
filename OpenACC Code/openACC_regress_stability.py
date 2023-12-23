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

class analysis():

    def __init__(self):
        self.data_file_path = None
        self.data = None
        self.delta_t = 0.1
        self.simulate_total_time = 200
        self.simulate_time_step = 0.1
        self.num_vehicles = 5  # Including the lead vehicle

    def load_data(self):
        self.data = pd.read_csv(self.data_file_path)

        return

    def calculate_acceleration(self, speed):

        return np.diff(speed) / self.delta_t

    def linear_regress(self, X, y):
        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        n = len(y)
        p = X.shape[1]
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

        return {
            'Coefficients': model.coef_,
            'Intercept': model.intercept_,
            'R2': r2,
            'Adjusted R2': adjusted_r2
        }

    def deter_av_model(self):
        av_model = {}

        # Regression analysis for each vehicle
        for i in range(2, 6):
            self.data[f'DeltaV{i}'] = self.data[f'Speed{i - 1}'] - self.data[f'Speed{i}']
            dv = self.data[f'Speed{i}'].values
            vf = self.data[f'Speed{i}'].values
            acc_f = self.calculate_acceleration(vf)
            ivs = self.data[f'IVS{i - 1}'].values[:-1]

            X = np.column_stack((ivs, self.data[f'DeltaV{i}'][:-1], vf[:-1]))
            y = acc_f

            av_model[f'AV{i}'] = self.linear_regress(X, y)

        return av_model

    def regress_with_delay(self):
        regression_results_all_delay = {}
        for delay in range(0, 51):
            regression_results_one_delay = {}
            # Regression analysis for each vehicle
            for i in range(2, 6):
                dv = self.data[f'Speed{i - 1}'] - self.data[f'Speed{i}']
                self.data[f'DeltaV{i}'] = dv
                vf = self.data[f'Speed{i}'].values
                acc_f = self.calculate_acceleration(vf)
                vf = vf[:-1]
                dv = dv[:-1]
                ivs = self.data[f'IVS{i - 1}'].values[:-1]
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
                    dv_delay = dv[:(-1) * delay]
                    vf_delay = vf[:(-1) * delay]
                    ivs_delay = ivs[:(-1) * delay]
                X_delay = np.column_stack((ivs_delay, dv_delay, vf_delay))
                y_delay = acc_f_delay

                regression_results_one_delay[f'AV{i}'] = self.linear_regress(X_delay, y_delay)
                regression_results_one_delay[f'AV{i}']['Tau'] = delay * self.delta_t

            regression_results_all_delay[delay] = regression_results_one_delay

        return regression_results_all_delay

    def deter_av_model_with_delay(self):
        regression_results_all_delay = self.regress_with_delay()
        delay_list = list(regression_results_all_delay.keys())
        veh_list = list(regression_results_all_delay[delay_list[0]].keys())
        av_model_with_delay = {}
        optimal_delay = {}
        for veh in veh_list:
            av_model_with_delay[veh] = regression_results_all_delay[delay_list[0]][veh]
            optimal_delay[veh] = delay_list[0]
        for delay in delay_list[1:]:
            for veh in veh_list:
                if regression_results_all_delay[delay][veh]['Adjusted R2'] > av_model_with_delay[veh]['Adjusted R2']:
                    av_model_with_delay[veh] = regression_results_all_delay[delay][veh]
                    optimal_delay[veh] = delay

        return av_model_with_delay

    def calculate_string_stability(self, av_model):
        string_stability_results = {}
        for i in range(2, 6):
            f_d, f_delta_v, f_v = av_model[f'AV{i}']['Coefficients']
            string_stability = 0.5 - f_delta_v / f_v - f_d / (f_v ** 2)
            string_stability_results[f'AV{i}'] = string_stability

        return string_stability_results

    def calculate_string_stability_with_delay(self, av_model_with_delay):
        string_stability_results1 = {}
        string_stability_results2 = {}
        for i in range(2, 6):
            f_d, f_delta_v, f_v = av_model_with_delay[f'AV{i}']['Coefficients']
            tau = av_model_with_delay[f'AV{i}']['Tau']
            string_stability1 = f_v ** 2 - 2 * f_d - 2 * f_v * f_delta_v
            string_stability2 = 1 + 2 * tau * (f_v - f_delta_v) + tau ** 2 * f_d
            string_stability_results1[f'AV{i}'] = string_stability1
            string_stability_results2[f'AV{i}'] = string_stability2

        return string_stability_results1, string_stability_results2

    def generate_lead_speed_profile(self):
        lead_speed_initial = 20  # m/s
        lead_speed_final = 25  # m/s
        speed_change_time = 100  # seconds

        # Time array
        times = np.arange(0, self.simulate_total_time, self.simulate_time_step)
        # Initialize arrays for speed
        speeds = np.zeros((1, len(times)))
        speeds[:, 0] = lead_speed_initial
        for t in range(1, len(times)):
            # Update lead vehicle speed and position
            if times[t] > speed_change_time:
                speeds[0, t] = lead_speed_final
            else:
                speeds[0, t] = lead_speed_initial

        return speeds

    def generate_simulate_ini_info_from_acc_data(self):
        selected_time_period = [1060, 1200]
        selected_df = self.data[(self.data['Time'] >= selected_time_period[0]) & (self.data['Time'] < selected_time_period[1])]
        lead_speed_profile = selected_df['Speed1'].values
        follow_ini_distance = [selected_df['IVS1'].values[0],
                               selected_df['IVS2'].values[0],
                               selected_df['IVS3'].values[0],
                               selected_df['IVS4'].values[0]]
        follow_ini_speed = [selected_df['Speed2'].values[0],
                            selected_df['Speed3'].values[0],
                            selected_df['Speed4'].values[0],
                            selected_df['Speed5'].values[0]]
        debug = 1

        return lead_speed_profile, follow_ini_distance, follow_ini_speed

    def determine_simulate_historic_info_from_acc_data(self):
        selected_time_period = [960, 1060]
        selected_df = self.data[(self.data['Time'] >= selected_time_period[0]) & (self.data['Time'] < selected_time_period[1])]
        # redefine the index of the dataframe
        indexList = range(selected_df.shape[0])
        selected_df = selected_df.set_index(pd.Index(indexList))
        # Time array
        times = np.arange(selected_time_period[0], selected_time_period[1], self.simulate_time_step)
        # Initialize arrays for speed and position
        speeds = selected_df[['Speed1', 'Speed2', 'Speed3', 'Speed4', 'Speed5']].values.transpose()
        positions = np.zeros((self.num_vehicles, len(times)))
        for t in range(len(times)-1, 0-1, -1):
            # calculate lead vehicle
            if t == len(times)-1:
                positions[0, t] = 0
            else:
                positions[0, t] = positions[0, t+1] - speeds[0, t] * self.simulate_time_step
            # calculate following vehicle
            for i in range(2, 6):
                positions[i-1, t] = positions[i-2, t] - selected_df[f'IVS{i-1}'].values[t]

        debug = 1

        return speeds, positions

    def extract_coefficients_from_av_model(self, av_model):
        coefficients = {}
        for i in range(2, 6):
            regression_veh = av_model[f'AV{i}']
            coefficients0 = list(regression_veh['Coefficients'])
            intercept = regression_veh['Intercept']
            parameters = coefficients0 + [intercept]
            coefficients[f'AV{i}'] = parameters

        return coefficients

    def extract_Tau(self, av_model):
        Tau_dict = {}
        for i in range(2, 6):
            regression_veh = av_model[f'AV{i}']
            if 'Tau' in regression_veh:
                Tau = regression_veh['Tau']
            else:
                Tau = 0
            Tau_dict[f'AV{i}'] = Tau

        return Tau_dict

    def kinematic_model_1(self, current_speed, acceleration, current_position):
        new_speed = current_speed + acceleration * self.simulate_time_step
        if new_speed <= 0:
            new_speed = 0
        new_position = current_position + new_speed * self.simulate_time_step + 0.5 * acceleration * self.simulate_time_step ** 2

        return new_speed, new_position

    def kinematic_model_2(self, current_speed, acceleration, current_position):
        new_speed = None
        new_position = None

        return new_speed, new_position

    def simulate_avs(self,
                     av_model,
                     lead_speed_profile=None,
                     follow_ini_distance=None,
                     follow_ini_speed=None,
                     historic_speeds=None,
                     historic_positions=None):
        # speed units: m/s
        # distance(position) units: m

        # Time array
        times = np.arange(0, self.simulate_total_time, self.simulate_time_step)
        # Initialize arrays for speed and position
        speeds = np.zeros((self.num_vehicles, len(times)))
        positions = np.zeros((self.num_vehicles, len(times)))

        # extract coefficients and delays from av_model
        coefficients = self.extract_coefficients_from_av_model(av_model)
        delays = self.extract_Tau(av_model)

        if lead_speed_profile is None:
            lead_speed_profile = self.generate_lead_speed_profile()
        if follow_ini_distance is None:
            follow_ini_distance = [40] * (self.num_vehicles - 1)
        if follow_ini_speed is None:
            follow_ini_speed = [20] * (self.num_vehicles - 1)
        if historic_speeds is None:
            historic_speeds = np.full((self.num_vehicles, len(times)), 20)
        if historic_positions is None:
            historic_positions = np.full((self.num_vehicles, len(times)), 0)

        # Set initial conditions
        speeds[0] = lead_speed_profile
        speeds[1:, 0] = follow_ini_speed
        ini_distance = [0] + follow_ini_distance
        positions[:, 0] = [-1 * sum(ini_distance[:i + 1]) for i in range(len(ini_distance))]

        # Simulation loop
        # for lead vehicles
        for t in range(1, len(times)):
            # Update lead vehicle position
            positions[0, t] = positions[0, t - 1] + speeds[0, t] * self.simulate_time_step
        # for follow vehicles
        for t in range(1, len(times)):
            # use info in t-1 to update t=info in t
            # Update positions and speeds of following vehicles
            for i in range(1, self.num_vehicles):
                delay = int(delays[f'AV{i+1}'] / self.delta_t)
                # Current speed and position
                current_speed = speeds[i, t-1]
                current_position = positions[i, t-1]
                if t-1-delay <= 0:
                    current_speed_delay = historic_speeds[i, t-1-delay]
                    current_position_delay = historic_positions[i, t-1-delay]
                else:
                    current_speed_delay = speeds[i, t-1-delay]
                    current_position_delay = positions[i, t-1-delay]
                # Lead vehicle's speed and position
                if t-1-delay <= 0:
                    lead_speed_delay = historic_speeds[i-1, t-1-delay]
                    lead_position_delay = historic_positions[i-1, t-1-delay]
                else:
                    lead_speed_delay = speeds[i-1, t-1-delay]
                    lead_position_delay = positions[i-1, t-1-delay]
                # Calculate the IVS, delta_v
                ivs_delay = lead_position_delay - current_position_delay
                delta_v_delay = lead_speed_delay - current_speed_delay

                # Using the regression model to update the speed
                acceleration = (coefficients[f'AV{i+1}'][0] * ivs_delay +
                                coefficients[f'AV{i+1}'][1] * delta_v_delay +
                                coefficients[f'AV{i+1}'][2] * current_speed_delay +
                                coefficients[f'AV{i+1}'][3])
                new_speed, new_position = self.kinematic_model_1(current_speed, acceleration, current_position)

                # Update the arrays
                speeds[i, t] = new_speed
                positions[i, t] = new_position
        debug = 1

        return times, speeds, positions

    def plot_av_speeds(self, times, speeds, pic_name, av_model_index):
        line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1))]
        colors = ['black', 'green', 'red', 'cyan', 'blue']

        plt.figure(figsize=(12, 8))
        for i in range(speeds.shape[0]):
            plt.plot(times, speeds[i], label=f'Vehicle {i} - Base' + av_model_index[i-1] if i > 0 else 'Lead Vehicle',
                     linestyle=line_styles[i % len(line_styles)], color=colors[i % len(colors)], linewidth=2.5)
            # plt.plot(times, speeds[i], label=f'Vehicle {i}' if i > 0 else 'Lead Vehicle',
            #          linestyle=line_styles[i % len(line_styles)], color=colors[i % len(colors)], linewidth=2.5)

        # Setting font sizes
        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel('Speed (m/s)', fontsize=16)
        plt.title('Speed of Vehicles Over Time', fontsize=16)
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.savefig(pic_name)
        # plt.show()



if __name__ == '__main__':

    m_analysis = analysis()
    m_analysis.data_file_path = 'ASta_test.csv'
    m_analysis.load_data()
    av_model = m_analysis.deter_av_model()
    string_stability_results = m_analysis.calculate_string_stability(av_model)

    av_model_with_delay = m_analysis.deter_av_model_with_delay()
    string_stability_results1, string_stability_results2 = m_analysis.calculate_string_stability_with_delay(av_model_with_delay)


    # simulate with self determined leading vehicle speeds and following vehicles distance
    # simulate different av delay model
    times, speeds, positions = m_analysis.simulate_avs(av_model_with_delay)
    av_model_index = list(av_model_with_delay.keys())
    m_analysis.plot_av_speeds(times, speeds, 'simulate different AV delay model.png', av_model_index)
    # simulate same av delay model
    for i in range(2, 6):
        av_model_index = []
        av_model_with_delay_ = av_model_with_delay.copy()
        for key in av_model_with_delay.keys():
            av_model_with_delay_[key] = av_model_with_delay[f'AV{i}']
            av_model_index.append(f'AV{i}')
        times, speeds, positions = m_analysis.simulate_avs(av_model_with_delay_)
        m_analysis.plot_av_speeds(times, speeds, f'simulate AV {i} delay model.png', av_model_index)

    #
    # # simulate with actual leading vehicle speeds and following vehicles distance
    # m_analysis.simulate_total_time = 140
    # lead_speed_profile, follow_ini_distance, follow_ini_speed = m_analysis.generate_simulate_ini_info_from_acc_data()
    # historic_speeds, historic_positions = m_analysis.determine_simulate_historic_info_from_acc_data()
    # # simulate different av delay model
    # times, speeds, positions = m_analysis.simulate_avs(av_model_with_delay,
    #                                                    lead_speed_profile=lead_speed_profile,
    #                                                    follow_ini_distance=follow_ini_distance,
    #                                                    follow_ini_speed=follow_ini_speed,
    #                                                    historic_speeds=historic_speeds,
    #                                                    historic_positions=historic_positions
    #                                                    )
    # av_model_index = list(av_model_with_delay.keys())
    # m_analysis.plot_av_speeds(times, speeds, 'simulate different AV delay model with actual lead veh.png', av_model_index)
    # # simulate same av delay model
    # for i in range(2, 6):
    #     av_model_index = []
    #     av_model_with_delay_ = av_model_with_delay.copy()
    #     for key in av_model_with_delay.keys():
    #         av_model_with_delay_[key] = av_model_with_delay[f'AV{i}']
    #         av_model_index.append(f'AV{i}')
    #     times, speeds, positions = m_analysis.simulate_avs(av_model_with_delay_,
    #                                                        lead_speed_profile=lead_speed_profile,
    #                                                        follow_ini_distance=follow_ini_distance,
    #                                                        follow_ini_speed=follow_ini_speed,
    #                                                        historic_speeds=historic_speeds,
    #                                                        historic_positions=historic_positions
    #                                                        )
    #     m_analysis.plot_av_speeds(times, speeds, f'simulate AV {i} delay model with actual lead veh.png', av_model_index)
