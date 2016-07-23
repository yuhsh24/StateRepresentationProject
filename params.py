# -*- coding: gb2312 -*-
# __author__ = 'fuhaobo'

import datetime as dt

# common config
input_filename = 'stock_whole_data.txt'
p_type = 'clf'  # 'reg' or 'clf'
is_use_fourier_denoising = True
top_fourier_energy = 1500
feature_mode_list = ['series_only']
test_start_date = dt.date(2015, 1, 1)
is_output_probability = True
train_data_length = 5760  # in terms of days
prediction_steps = 1
lag_list = [10]


# for SOM
som_num_row = 6
som_num_col = 6
som_embedding_dim = 5
is_use_one_hot = False

# for NN
max_num_train_iterations = 1

# for LSTM
epochs_per_cycle = 10
cycles = 100


# for q-learning
num_train_times_q_learning = 201
start_money = 1000000
initial_q_value = 0.0
is_use_date_as_state = False
is_consider_cost_fee = True
is_update_q_for_val_and_test = True
is_show_pics = True

# q_train_start_date = dt.date(2001, 1, 1)
# q_val_start_date = dt.date(2009, 1, 1)
q_test_start_date = dt.date(2010, 1, 1)
# q_test_end_date = dt.date(2010, 12, 31)

position_grind = 0.25 # should be divided by 1 #
unit_size = 0.25  # buy and sell portion #

stamp_tax = 0.001  # only for sell
commission_tax = 0.0005  # for both sell and buy
transfer_tax = 0.001  # only for shanghai stock market (both sell and buy)

exploration_epsilon = 0.2
learning_rate = 0.1 #
discount = 1.0

selected_signals = ["cp_greater_ma20", "op_break_high", "op_break_low", "tv_break_high", "tv_break_low",
                    "ma5_greater_ma60", "vma5_greater_vma22", "macd_signal_1", "macd_signal_2", "9k_9d"]

#selected_signals = ["macd_signal_1", "macd_signal_2"]