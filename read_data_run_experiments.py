# -*- coding: gb2312 -*-
# __author__ = 'fuhaobo'

import pandas as pd
import datetime as dt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import gaussian_process
from scipy.fftpack import dct, idct
import params
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def read_data_into_dict_df(input_filename):
    metric_name_list = ['op', 'hp', 'lp', 'cp', 'tv', 'ta', 'index']
    stock_code_list = ['000001', '000002', '000016', '000300', '399001', '399002', '399005', '399006', '399106',
                       '399550']
    data_dict = dict()
    for metric in metric_name_list:
        for stock in stock_code_list:
            data_dict[metric + '_' + stock] = []
    # parse the original stock data
    for line in open(input_filename):
        items = line.strip().split()
        # trade_date stock_code	stock_name open_price highest_price	lowest_price close_price turnover_volume
        # turnover_amount
        if items[1] in stock_code_list:
            for index in range(0, len(metric_name_list)-1):
                data_dict[metric_name_list[index] + '_' + items[1]].append(items[index + 3])
            data_dict[metric_name_list[-1] + '_' + items[1]].append(items[0])

    for metric in metric_name_list:
        if metric != 'index':
            for stock in stock_code_list:
                data_dict[metric + '_' + stock] = pd.Series([float(i) for i in data_dict[metric + '_' + stock]],
                                                            index=[dt.datetime.strptime(x, '%Y/%m/%d').date()
                                                                   for x in data_dict['index' + '_' + stock]])
                # make sure each series is sorted by the dates
                data_dict[metric + '_' + stock].sort_index(inplace=True)

    for stock in stock_code_list:
        data_dict.pop('index' + '_' + stock, None)

    df_result = pd.DataFrame(data_dict)
    return df_result


# produce the train val test data for a specific stock (we only consider the prediction of up/downs of the close price)
# the data can be constructed by the following three means
# (1) only the time series data of the close price
# (2) data that belongs to the current stock, which includes open price, turn amount, etc.
# (3) all data, containing information from other stocks
# param whole_data: a pandas data frame
# param test_start_date
# param prediction_step: notice this may not be consecutive days
# param stock_code
# param num_lags
# param mode: 'series_only', 'stock_only', 'all'
# param is_use_day_of_week: boolean
def produce_train_val_test(original_whole_data, train_data_length, test_start_date, prediction_step, stock_code,
                           num_lags, mode, is_use_day_of_week, clf_reg_mode):
    whole_data = original_whole_data.copy()
    train = []
    train_data_length = dt.timedelta(days=train_data_length)
    val = []
    train_val = []
    test = []
    test_date_start_index = find_date_index(test_start_date,
                                            original_whole_data['cp' + '_' + stock_code].dropna().index)

    if params.is_use_fourier_denoising:
        x = whole_data['cp' + '_' + stock_code].dropna()[:test_date_start_index]
        # print(x[-1])
        # print(test_start_date)
        # print(whole_data['cp' + '_' + stock_code].dropna()[test_start_date])
        # print(whole_data['cp' + '_' + stock_code].dropna()[test_date_start_index])
        # print()

        y = dct(x, norm='ortho')
        window = np.zeros(len(x))
        window[:min(params.top_fourier_energy, len(x))] = 1
        denoised_x = idct(y * window, norm='ortho')

        # write back
        dates = whole_data['cp' + '_' + stock_code].dropna()[:test_date_start_index].index
        index = 0
        for date in dates:
            whole_data['cp' + '_' + stock_code][date] = denoised_x[index]
            index += 1

    if mode == 'series_only':
        ts = whole_data['cp' + '_' + stock_code].dropna()
        ts_shift = ts.shift()
        ts_diff = ts - ts_shift
        ts_diff.dropna(inplace=True)

        # delete out_of_date data
        ts_diff = ts_diff[test_start_date - train_data_length:]

        # calculate the validation date
        val_index = int(len(ts_diff[:test_date_start_index]) * 0.75)
        val_start_date = ts_diff.index[val_index]

        for index in range(num_lags, len(ts_diff)):
            temp_array = []
            for diff in ts_diff[index-num_lags:index]:
                temp_array.append(diff)

            if clf_reg_mode == 'clf':
                temp_array.append(int(ts_diff[index] > 0))
            else:  # clf_reg_mode == 'reg'
                temp_array.append(ts_diff[index])

            if ts_diff.index[index] < val_start_date:
                train.append(temp_array)
                train_val.append(temp_array)
            if val_start_date <= ts_diff.index[index] < test_start_date:
                val.append(temp_array)
                train_val.append(temp_array)
            if test_start_date <= ts_diff.index[index] and len(test) < prediction_step:
                test.append(temp_array)

        # correct the label for the first test data point
        if params.is_use_fourier_denoising:
            if clf_reg_mode == 'clf':
                test[0][-1] = int(original_whole_data['cp' + '_' + stock_code].dropna()[test_date_start_index] -
                                  original_whole_data['cp' + '_' + stock_code].dropna()[test_date_start_index - 1] > 0)
            else:  # clf_reg_mode == 'reg'
                test[0][-1] = original_whole_data['cp' + '_' + stock_code].dropna()[test_date_start_index] - \
                              original_whole_data['cp' + '_' + stock_code].dropna()[test_date_start_index - 1]

        return np.array(train), np.array(val), np.array(train_val), np.array(test)

    if mode == 'stock_only':
        metric_name_list = ['op', 'hp', 'lp', 'cp', 'tv', 'ta']
        dict_metric = dict()
        for metric in metric_name_list:
            ts = whole_data[metric + '_' + stock_code].dropna()
            ts_shift = ts.shift()
            ts_diff = ts - ts_shift
            dict_metric[metric + '_' + stock_code] = ts_diff.dropna()[test_start_date - train_data_length:]

        # calculate the validation date
        val_index = int(len(dict_metric['cp' + '_' + stock_code][:test_date_start_index]) * 0.75)
        val_start_date = dict_metric['cp' + '_' + stock_code].index[val_index]

        for index in range(num_lags, len(dict_metric['cp' + '_' + stock_code])):
            temp_array = []
            for metric in metric_name_list:
                ts_diff = dict_metric[metric + '_' + stock_code]
                for diff in ts_diff[index - num_lags:index]:
                    temp_array.append(diff)
            if is_use_day_of_week:
                # add the feature of the day of the week
                day_of_week = 7 * [0]
                which_day = dict_metric['cp' + '_' + stock_code].index[index].weekday()
                day_of_week[which_day] = 1
                temp_array += day_of_week

            if clf_reg_mode == 'clf':
                temp_array.append(int(dict_metric['cp' + '_' + stock_code][index] > 0))
            else:  # clf_reg_mode == 'reg'
                temp_array.append(dict_metric['cp' + '_' + stock_code][index])

            if dict_metric['cp' + '_' + stock_code].index[index] < val_start_date:
                train.append(temp_array)
                train_val.append(temp_array)
            if val_start_date <= dict_metric['cp' + '_' + stock_code].index[index] < test_start_date:
                val.append(temp_array)
                train_val.append(temp_array)
            if test_start_date <= dict_metric['cp' + '_' + stock_code].index[index] and len(test) < prediction_step:
                test.append(temp_array)

        # correct the label for the first test data point
        if params.is_use_fourier_denoising:
            if clf_reg_mode == 'clf':
                test[0][-1] = int(original_whole_data['cp' + '_' + stock_code].dropna()[test_date_start_index] -
                                  original_whole_data['cp' + '_' + stock_code].dropna()[test_date_start_index - 1] > 0)
            else:  # clf_reg_mode == 'reg'
                test[0][-1] = original_whole_data['cp' + '_' + stock_code].dropna()[test_date_start_index] - \
                              original_whole_data['cp' + '_' + stock_code].dropna()[test_date_start_index - 1]

        return np.array(train), np.array(val), np.array(train_val), np.array(test)

    if mode == 'all':
        metric_name_list = ['op', 'hp', 'lp', 'cp', 'tv', 'ta']
        stock_code_list = ['000001', '000002', '000016', '000300', '399001', '399002', '399005', '399006', '399106',
                           '399550']
        # this is important!
        whole_data.dropna(inplace=True)

        dict_metric = dict()
        # notice the time alignment !!!
        for metric in metric_name_list:
            for each_stock in stock_code_list:
                ts = whole_data[metric + '_' + each_stock]
                ts_shift = ts.shift()
                ts_diff = ts - ts_shift
                dict_metric[metric + '_' + each_stock] = ts_diff.dropna()[test_start_date - train_data_length:]

        # calculate the validation date
        val_index = int(len(dict_metric['cp' + '_' + stock_code][:test_date_start_index]) * 0.75)
        val_start_date = dict_metric['cp' + '_' + stock_code].index[val_index]

        for index in range(num_lags, len(dict_metric['cp' + '_' + stock_code])):
            temp_array = []
            for metric in metric_name_list:
                for each_stock in stock_code_list:
                    ts_diff = dict_metric[metric + '_' + each_stock]
                    for diff in ts_diff[index - num_lags:index]:
                        temp_array.append(diff)
            if is_use_day_of_week:
                # add the feature of the day of the week
                day_of_week = 7 * [0]
                which_day = dict_metric['cp' + '_' + stock_code].index[index].weekday()
                day_of_week[which_day] = 1
                temp_array += day_of_week

            if clf_reg_mode == 'clf':
                temp_array.append(int(dict_metric['cp' + '_' + stock_code][index] > 0))
            else:  # clf_reg_mode == 'reg'
                temp_array.append(dict_metric['cp' + '_' + stock_code][index])

            if dict_metric['cp' + '_' + stock_code].index[index] < val_start_date:
                train.append(temp_array)
                train_val.append(temp_array)
            if val_start_date <= dict_metric['cp' + '_' + stock_code].index[index] < test_start_date:
                val.append(temp_array)
                train_val.append(temp_array)
            if test_start_date <= dict_metric['cp' + '_' + stock_code].index[index] and len(test) < prediction_step:
                test.append(temp_array)

        # correct the label for the first test data point
        if params.is_use_fourier_denoising:
            if clf_reg_mode == 'clf':
                test[0][-1] = int(original_whole_data['cp' + '_' + stock_code].dropna()[test_date_start_index] -
                                  original_whole_data['cp' + '_' + stock_code].dropna()[test_date_start_index - 1] > 0)
            else:  # clf_reg_mode == 'reg'
                test[0][-1] = original_whole_data['cp' + '_' + stock_code].dropna()[test_date_start_index] - \
                              original_whole_data['cp' + '_' + stock_code].dropna()[test_date_start_index - 1]

        return np.array(train), np.array(val), np.array(train_val), np.array(test)


def return_a_reg(reg_name):
    if reg_name == 'RR':
        reg = Ridge()
        reg.set_params(alpha=1)
        return reg
    if reg_name == 'KRR':
        reg = KernelRidge()
        reg.set_params(kernel='rbf', alpha=1)
        return reg
    if reg_name == 'SVR':
        reg = SVR()
        reg.set_params(kernel='rbf', C=10)
        return reg
    if reg_name == 'KNR':
        reg = KNeighborsRegressor()
        reg.set_params(n_neighbors=10)
        return reg
    if reg_name == 'GP':
        reg = gaussian_process.GaussianProcess()
        reg.set_params(regr='constant', corr='squared_exponential', theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
        return reg
    if reg_name == 'DT':
        reg = tree.DecisionTreeRegressor()
        reg.set_params(max_depth=8, min_samples_leaf=5, random_state=0)
        return reg
    elif reg_name == 'RF':
        reg = RandomForestRegressor()
        reg.set_params(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=0)
        return reg
    elif reg_name == 'GBDT':
        reg = GradientBoostingRegressor()
        reg.set_params(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=0)
        return reg


def return_a_clf(clf_name, mode):
    if clf_name == "LR":
        clf = LogisticRegression()
        if mode == 'series_only':
            clf.set_params(C=50, penalty="l2")
        if mode == 'stock_only':
            clf.set_params(C=7.5, penalty="l2")
        if mode == 'all':
            clf.set_params(C=10, penalty="l2")
        return clf
    elif clf_name == 'SVC':
        clf = SVC()
        if mode == 'series_only':
            clf.set_params(C=50, kernel='rbf', probability=True)
        if mode == 'stock_only':
            clf.set_params(C=7.5, kernel='rbf', probability=True)
        if mode == 'all':
            clf.set_params(C=10, kernel='rbf', probability=True)
        return clf
    elif clf_name == 'DT':
        clf = tree.DecisionTreeClassifier()
        if mode == 'series_only':
            clf.set_params(criterion='gini', max_depth=15, min_samples_leaf=5, random_state=0)
        if mode == 'stock_only':
            clf.set_params(criterion='gini', max_depth=10, min_samples_leaf=10, random_state=0)
        if mode == 'all':
            clf.set_params(criterion='gini', max_depth=5, min_samples_leaf=10, random_state=0)
        return clf
    elif clf_name == 'RF':
        clf = RandomForestClassifier()
        if mode == 'series_only':
            clf.set_params(criterion='gini', n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=0)
        if mode == 'stock_only':
            clf.set_params(criterion='gini', n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=0)
        if mode == 'all':
            clf.set_params(criterion='gini', n_estimators=100, max_depth=4, min_samples_leaf=10, random_state=0)
        return clf
    elif clf_name == 'GBDT':
        clf = GradientBoostingClassifier()
        if mode == 'series_only':
            clf.set_params(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=0)
        if mode == 'stock_only':
            clf.set_params(n_estimators=100, max_depth=4, min_samples_leaf=10, random_state=0)
        if mode == 'all':
            clf.set_params(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=0)
        return clf
    elif clf_name == 'AB':
        clf = AdaBoostClassifier()
        if clf == 'series_only':
            model.set_params(n_estimators=50, learning_rate=0.8)
        if clf == 'stock_only':
            clf.set_params(n_estimators=40, learning_rate=0.8)
        if mode == 'all':
            clf.set_params(n_estimators=40, learning_rate=0.8)
        return clf
    elif clf_name == 'NB':
        clf = GaussianNB()
        return clf
    else:
        print("Error! There is no implements of '%s' classifier" % clf_name)
        print("choose the right classifier name.\n"
              "classifier list:'LR', 'SVC', 'DT', 'RF', 'GBDT', 'AB'")


def find_date_index(date, date_list):
    target_index = -1
    for index in range(0, len(date_list)):
        if date_list[index] >= date:
            target_index = index
            break
    return target_index


# param en_result is a list of a list, each small list is a
def return_prob_array_for_ensemble(en_result):  # en_result is a 2_dimensional list
    result_array = np.array(en_result)
    row_dim, col_dim = result_array.shape
    prob_array = col_dim * [0]

    for index in range(0, col_dim):
        temp_array = result_array[:, index]
        prob_array[index] = np.mean(temp_array)

    return prob_array


if __name__ == '__main__':
    df_whole_data = read_data_into_dict_df(params.input_filename)
    prediction_steps = params.prediction_steps
    p_type = params.p_type
    test_start_date = params.test_start_date
    train_data_length = params.train_data_length

    stock_code_list = ['000001', '000002', '000016', '000300', '399001', '399002', '399005', '399006', '399106',
                       '399550']
    for stock in stock_code_list:
        print('stock ' + stock + ' start date: ' + str(df_whole_data['cp' + '_' + stock].dropna().index[0]))

    for stock_code in ['000001', '000002', '000016', '000300', '399001', '399002', '399005', '399006', '399106',
                       '399550']:
        for lag in params.lag_list:
            for feature_mode in params.feature_mode_list:
                if feature_mode == 'all':
                    df_whole_data.dropna(inplace=True)

                test_date_start_index = find_date_index(test_start_date,
                                                        df_whole_data['cp' + '_' + stock_code].dropna().index)
                if test_date_start_index == -1:
                    print("test date wrong")
                    exit(-1)

                if params.is_use_fourier_denoising:
                    result_file = open('./common_models_result_with_fourier/stock' + stock_code + '-lag' + str(lag) +
                                       '-' + feature_mode + '-p' + str(prediction_steps) + '-' + p_type + '-tl' +
                                       str(train_data_length) + '-tfe' + str(params.top_fourier_energy) + '.txt', 'w')
                    print("Experiments Start: " + 'stock' + stock_code + '-lag' + str(lag) + '-' + feature_mode + '-p' +
                          str(prediction_steps) + '-' + p_type + '-tl' + str(train_data_length) + '-tfe' +
                          str(params.top_fourier_energy))
                else:
                    result_file = open('./common_models_result_with_fourier/stock' + stock_code + '-lag' + str(lag) +
                                       '-' + feature_mode + '-p' + str(prediction_steps) + '-' + p_type + '-tl' +
                                       str(train_data_length) + '.txt', 'w')
                    print("Experiments Start: " + 'stock' + stock_code + '-lag' + str(lag) + '-' + feature_mode + '-p' +
                          str(prediction_steps) + '-' + p_type + '-tl' + str(train_data_length))

                if p_type == 'clf':
                    result_file.write(
                        'prediction_date' + '\t' + 'LR' + '\t' + 'SVC' + '\t' + 'DT' + '\t' + 'RF' + '\t' + 'GBDT' +
                        '\t' + 'AB' + '\t' + 'NB' + '\t' + 'Ensemble' + '\t' + 'True' + '\n')
                else:  # p_type == 'reg'
                    result_file.write(
                        'prediction_date' + '\t' + 'RR' + '\t' + 'KRR' + '\t' + 'SVR' + '\t' + 'KNR' + '\t' +
                        'DT' + '\t' + 'RF' + '\t' + 'GBDT' + '\t' + 'Ensemble' + '\t' + 'True' + '\n')

                for date_index in range(test_date_start_index,
                                        len(df_whole_data['cp' + '_' + stock_code].dropna()),
                                        prediction_steps):
                    test_date = df_whole_data['cp' + '_' + stock_code].dropna().index[date_index]
                    train_data, val_data, train_val_data, test_data = produce_train_val_test(original_whole_data=
                                                                                             df_whole_data,
                                                                                             train_data_length=
                                                                                             train_data_length,
                                                                                             test_start_date=test_date,
                                                                                             prediction_step=
                                                                                             prediction_steps,
                                                                                             stock_code=stock_code,
                                                                                             num_lags=lag,
                                                                                             mode=feature_mode,
                                                                                             is_use_day_of_week=True,
                                                                                             clf_reg_mode=p_type)
                    normalizer = preprocessing.StandardScaler().fit(train_val_data[:, 0:-1])
                    train_val_input_scaled = normalizer.transform(train_val_data[:, 0:-1])
                    test_input_scaled = normalizer.transform(test_data[:, 0:-1])
                    # print('train_data shape:' + str(train_data.shape))
                    # print('val_data shape:' + str(val_data.shape))
                    # print('train_val_data shape:' + str(train_val_data.shape))
                    # print('test_data shape:' + str(test_data.shape))
                    result_dic = dict()
                    ensemble_result = []
                    result_dic['date'] = \
                        df_whole_data['cp' + '_' + stock_code].dropna().index[date_index:date_index + prediction_steps]

                    # print(date_index)
                    # print(date_index + prediction_steps)
                    # print(df_whole_data['cp' + '_' + stock_code].dropna().index)

                    clf_name_list = ['LR', 'SVC', 'DT', 'RF', 'GBDT', 'AB', 'NB']
                    # reg_name_list = ['RR', 'KRR', 'SVR', 'KNR', 'GP', 'DT', 'RF', 'GBDT']
                    reg_name_list = ['RR', 'KRR', 'SVR', 'KNR', 'DT', 'RF', 'GBDT']

                    if p_type == 'clf':
                        model_name_list = clf_name_list
                    else:  # p_type == 'reg'
                        model_name_list = reg_name_list

                    for name in model_name_list:
                        if p_type == 'clf':
                            model = return_a_clf(name, feature_mode)
                            model.fit(train_val_input_scaled, np.array(train_val_data[:, -1], dtype=int))
                        else:  # p_type == 'reg'
                            model = return_a_reg(name)
                            model.fit(train_val_input_scaled, train_val_data[:, -1])

                        if params.is_output_probability and p_type == 'clf':
                            # the last column for the '1' prob
                            test_result = model.predict_proba(test_input_scaled)[:, -1]

                        else:
                            test_result = model.predict(test_input_scaled)
                        ensemble_result.append(test_result.tolist())
                        result_dic[name] = test_result
                        # if p_type == 'clf':
                        # print(name + '\t' + 'classification accuracy' + '\t' + str(accuracy_score(test_data[:, -1],
                        #                                                                          test_result)))
                        # else:  # p_type == 'reg'
                        # print(name + '\t' + 'mean squared error' + '\t' + str(mean_squared_error(test_data[:, -1],
                        #                                                                         test_result)))
                        # print(name + '\t' + str(accuracy_score(test_data[:, -1], test_result)))

                        # plot negative samples
                        # x_list = train_val_data[[train_val_data[:, -1] == 0]][:,0]
                        # y_list = train_val_data[[train_val_data[:, -1] == 0]][:,1]
                        # plt.scatter(x_list, y_list, c='b')

                        # plot positive samples
                        # x_list = train_val_data[[train_val_data[:, -1] == 1]][:,0]
                        # y_list = train_val_data[[train_val_data[:, -1] == 1]][:,1]
                        # plt.scatter(x_list, y_list, c='r')
                        # plt.show()

                    if p_type == 'clf':
                        if params.is_output_probability:
                            ensemble_result = return_prob_array_for_ensemble(ensemble_result)
                        else:
                            ensemble_result = 1 * np.array([x >= int((len(model_name_list) + 1) / 2) for
                                                            x in np.sum(np.array(ensemble_result), axis=0)])
                    else:  # p_type == 'reg'
                        ensemble_result = np.mean(np.array(ensemble_result), axis=0)

                    # ensemble_result = 1 * np.array([x >= int((len(clf_name_list)+1)/2) for x in ensemble_result])
                    result_dic['Ensemble'] = ensemble_result
                    if p_type == 'clf':
                        result_dic['True'] = np.array(test_data[:, -1], dtype=int)
                    else:  # p_type == 'reg'
                        result_dic['True'] = np.array(test_data[:, -1])

                    # print('Ensemble' + '\t' + str(accuracy_score(test_data[:, -1], ensemble_result)))

                    # output to file
                    temp_str = ''
                    for row_index in range(0, len(result_dic['True'])):
                        for column_name in ['date'] + model_name_list + ['Ensemble', 'True']:
                            temp_str += str(result_dic[column_name][row_index]) + '\t'
                        temp_str = temp_str.strip()
                        temp_str += '\n'
                    result_file.write(temp_str)
                result_file.close()
