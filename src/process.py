import os
import itertools
import json
import numpy as np
import pandas as pd
import math
from utils import save, load, makedir_exist_ok
from config import cfg
import matplotlib.pyplot as plt
from collections import defaultdict

result_path = './output/result'
vis_path = './output/vis'
num_experiments = 1
exp = [str(x) for x in list(range(num_experiments))]
fontsize = 16
save_format = 'png'


def make_controls(data_names, model_names, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + data_names + model_names + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(file):
    if file == 'all':
        control_name = [[['1'], ['1'], ['none'], ['-1'], ['none'], ['none']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls = make_controls(data_names, model_names, control_name)
        data_names = [['CIFAR100']]
        model_names = [['wresnet28x8']]
        cifar100_controls = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls + cifar100_controls
    elif file == 't':
        control_name = [[['1'], ['1'], ['none'], ['250', '4000'], ['none'], ['none']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls = make_controls(data_names, model_names, control_name)
        control_name = [[['1'], ['1'], ['none'], ['2500', '10000'], ['none'], ['none']]]
        data_names = [['CIFAR100']]
        model_names = [['wresnet28x8']]
        cifar100_controls = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls + cifar100_controls
    elif file == 'ts':
        control_name = [[['1'], ['1'], ['iid'], ['250', '4000'], ['r'], ['0.95']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls = make_controls(data_names, model_names, control_name)
        control_name = [[['1'], ['1'], ['iid'], ['2500', '10000'], ['r'], ['0.95']]]
        data_names = [['CIFAR100']]
        model_names = [['wresnet28x8']]
        cifar100_controls = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls + cifar100_controls
    elif file == 'silo':
        control_name = [[['10'], ['1'], ['iid', 'non-iid-2'], ['250', '4000'], ['r'], ['0.95']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls = make_controls(data_names, model_names, control_name)
        control_name = [[['10'], ['1'], ['iid'], ['2500', '10000'], ['r'], ['0.95']]]
        data_names = [['CIFAR100']]
        model_names = [['wresnet28x8']]
        cifar100_controls = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls + cifar100_controls
    elif file == 'device':
        control_name = [[['100'], ['0.1'], ['iid', 'non-iid-2'], ['250', '4000'], ['r'], ['0.95']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls = make_controls(data_names, model_names, control_name)
        control_name = [[['100'], ['0.1'], ['iid', 'non-iid-2'], ['2500', '10000'], ['r'], ['0.95']]]
        data_names = [['CIFAR100']]
        model_names = [['wresnet28x8']]
        cifar100_controls = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls + cifar100_controls
    else:
        raise ValueError('Not valid file')
    return controls


def main():
    # all_control_list = make_control_list('all')
    # t_control_list = make_control_list('t')
    ts_control_list = make_control_list('ts')
    silo_control_list = make_control_list('silo')
    device_control_list = make_control_list('device')
    # controls = all_control_list + t_control_list + ts_control_list + silo_control_list + device_control_list
    controls = ts_control_list + silo_control_list + device_control_list
    processed_result_exp, processed_result_history = process_result(controls)
    with open('{}/processed_result_exp.json'.format(result_path), 'w') as fp:
        json.dump(processed_result_exp, fp, indent=2)
    save(processed_result_exp, os.path.join(result_path, 'processed_result_exp.pt'))
    save(processed_result_history, os.path.join(result_path, 'processed_result_history.pt'))
    extracted_processed_result_exp = {}
    extracted_processed_result_history = {}
    extract_processed_result(extracted_processed_result_exp, processed_result_exp, [])
    extract_processed_result(extracted_processed_result_history, processed_result_history, [])
    df_exp = make_df_exp(extracted_processed_result_exp)
    df_history = make_df_history(extracted_processed_result_history)
    # make_vis(df_history)
    make_lc(df_history)
    return


def process_result(controls):
    processed_result_exp, processed_result_history = {}, {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), model_tag, processed_result_exp, processed_result_history)
    summarize_result(processed_result_exp)
    summarize_result(processed_result_history)
    return processed_result_exp, processed_result_history


def extract_result(control, model_tag, processed_result_exp, processed_result_history):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            for k in base_result['logger']['test'].mean:
                metric_name = k.split('/')[1]
                if metric_name not in processed_result_exp:
                    processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                    processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                    processed_result_exp[metric_name]['exp'][exp_idx] = base_result['logger']['test'].mean[k]
                processed_result_history[metric_name]['history'][exp_idx] = base_result['logger']['train'].history[k]
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        if control[1] not in processed_result_exp:
            processed_result_exp[control[1]] = {}
            processed_result_history[control[1]] = {}
        extract_result([control[0]] + control[2:], model_tag, processed_result_exp[control[1]],
                       processed_result_history[control[1]])
    return


def summarize_result(processed_result):
    if 'exp' in processed_result:
        pivot = 'exp'
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0).item()
        processed_result['std'] = np.std(processed_result[pivot], axis=0).item()
        processed_result['max'] = np.max(processed_result[pivot], axis=0).item()
        processed_result['min'] = np.min(processed_result[pivot], axis=0).item()
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0).item()
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0).item()
        processed_result[pivot] = processed_result[pivot].tolist()
    elif 'history' in processed_result:
        pivot = 'history'
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0)
        processed_result['std'] = np.std(processed_result[pivot], axis=0)
        processed_result['max'] = np.max(processed_result[pivot], axis=0)
        processed_result['min'] = np.min(processed_result[pivot], axis=0)
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0)
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0)
        processed_result[pivot] = processed_result[pivot].tolist()
    else:
        for k, v in processed_result.items():
            summarize_result(v)
        return
    return


def extract_processed_result(extracted_processed_result, processed_result, control):
    if 'exp' in processed_result or 'history' in processed_result:
        exp_name = '_'.join(control[:-1])
        metric_name = control[-1]
        if exp_name not in extracted_processed_result:
            extracted_processed_result[exp_name] = defaultdict()
        extracted_processed_result[exp_name]['{}_mean'.format(metric_name)] = processed_result['mean']
        extracted_processed_result[exp_name]['{}_std'.format(metric_name)] = processed_result['std']
    else:
        for k, v in processed_result.items():
            extract_processed_result(extracted_processed_result, v, control + [k])
    return


def make_df_exp(extracted_processed_result_exp):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_exp:
        control = exp_name.split('_')
        data_name, model_name, num_clients, active_rate, data_split_mode, num_supervised, client_data_mode, threshold = control
        index_name = ['_'.join([active_rate, data_split_mode, num_supervised, client_data_mode, threshold])]
        df_name = '_'.join([data_name, model_name, num_clients])
        df[df_name].append(pd.DataFrame(data=extracted_processed_result_exp[exp_name], index=index_name))
    startrow = 0
    writer = pd.ExcelWriter('{}/result_exp.xlsx'.format(result_path), engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return df


def make_df_history(extracted_processed_result_history):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_history:
        control = exp_name.split('_')
        data_name, model_name, num_clients, active_rate, data_split_mode, num_supervised, client_data_mode, threshold = control
        index_name = ['_'.join([active_rate, data_split_mode, num_supervised, client_data_mode, threshold])]
        for k in extracted_processed_result_history[exp_name]:
            df_name = '_'.join([data_name, model_name, num_clients, k])
            df[df_name].append(
                pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
    startrow = 0
    writer = pd.ExcelWriter('{}/result_history.xlsx'.format(result_path), engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return df


def make_vis(df):
    color = {'Joint': 'red', 'Alone': 'orange', 'GAL-b': 'dodgerblue', 'GAL-s': 'green'}
    linestyle = {'Joint': '-', 'Alone': '--', 'GAL-b': ':', 'GAL-s': '-.'}
    marker = {'Joint': {'1': 'o', '10': 's', '100': 'D'}, 'Alone': {'1': 'v', '10': '^', '100': '>'},
              'GAL-b': {'1': 'p', '10': 'd', '100': 'h'}, 'GAL-s': {'1': 'X', '10': '*', '100': 'x'}}
    loc = {'Loss': 'upper right', 'Accuracy': 'lower right', 'RMSE': 'upper right',
           'Gradient assisted learning rates': 'upper right', 'Assistance weights': 'upper right'}
    color_ap = ['red', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange']
    linestyle_ap = ['-', '--', ':', '-.', '-', '--', ':', '-.']
    marker_ap = ['o', 's', 'v', '^', 'p', 'd', 'X', '*']
    fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
    capsize = 5
    save_format = 'png'
    fig = {}
    for df_name in df:
        print(df_name)
        data_name, model_name, num_users, global_epoch, metric_name, stat = df_name.split('_')
        if num_users == '1':
            continue
        if stat == 'std':
            continue
        df_name_std = '_'.join([data_name, model_name, num_users, global_epoch, metric_name, 'std'])
        baseline_df_name = '_'.join([data_name, model_name, '1', global_epoch, metric_name, stat])
        baseline_df_name_std = '_'.join([data_name, model_name, '1', global_epoch, metric_name, 'std'])
        if metric_name in ['Loss', 'Accuracy', 'RMSE']:
            x = np.arange(0, int(global_epoch) + 1)
        elif metric_name in ['Assist-Rate']:
            x = np.arange(1, int(global_epoch) + 1)
            metric_name = 'Gradient assisted learning rates'
        elif metric_name in ['Assist-Parameters']:
            x = np.arange(1, int(global_epoch) + 1)
            metric_name = 'Assistance weights'
        else:
            raise ValueError('Not valid metric name')
        if global_epoch == '10':
            markevery = 1
            xticks = np.arange(0, int(global_epoch) + 1, step=markevery)
        elif global_epoch == '50':
            markevery = 10
            xticks = np.arange(int(global_epoch) + 1, step=markevery)
            xticks[0] = 1
        else:
            raise ValueError('Not valid global epoch')
        if baseline_df_name in df:
            for ((index, row), (_, row_std)) in zip(df[baseline_df_name].iterrows(),
                                                    df[baseline_df_name_std].iterrows()):
                local_epoch, assist_mode = index.split('_')
                if assist_mode == 'none':
                    assist_mode = 'Joint'
                else:
                    raise ValueError('Not valid assist_mode')
                label_name = '{}'.format(assist_mode)
                y = row.to_numpy()
                yerr = row_std.to_numpy()
                fig[df_name] = plt.figure(df_name)
                if metric_name == 'Gradient assisted learning rates':
                    plt.plot(x, y, color=color[assist_mode],
                             linestyle=linestyle[assist_mode], label=label_name,
                             marker=marker[assist_mode][local_epoch], markevery=markevery)
                else:
                    plt.errorbar(x, y, yerr=yerr, capsize=capsize, color=color[assist_mode],
                                 linestyle=linestyle[assist_mode], label=label_name,
                                 marker=marker[assist_mode][local_epoch], markevery=markevery)
                plt.legend(loc=loc[metric_name], fontsize=fontsize['legend'])
                plt.xlabel('Assistance rounds', fontsize=fontsize['label'])
                plt.ylabel(metric_name, fontsize=fontsize['label'])
                plt.xticks(xticks, fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
        for ((index, row), (_, row_std)) in zip(df[df_name].iterrows(), df[df_name_std].iterrows()):
            local_epoch, assist_mode = index.split('_')
            if metric_name == 'Assistance weights':
                for i in reversed(range(int(num_users))):
                    label_name = 'm = {}'.format(i + 1)
                    y = row.to_numpy().reshape(int(global_epoch), -1)[:, i]
                    yerr = row_std.to_numpy().reshape(int(global_epoch), -1)[:, i]
                    fig[df_name] = plt.figure(df_name)
                    plt.plot(x, y, color=color_ap[i], linestyle=linestyle_ap[i],
                             label=label_name, marker=marker_ap[i], markevery=markevery)
                    plt.legend(loc=loc[metric_name], fontsize=fontsize['legend'])
                    plt.xlabel('Assistance rounds', fontsize=fontsize['label'])
                    plt.ylabel(metric_name, fontsize=fontsize['label'])
                    plt.xticks(xticks, fontsize=fontsize['ticks'])
                    plt.yticks(fontsize=fontsize['ticks'])
            else:
                if assist_mode == 'none':
                    assist_mode = 'Alone'
                elif assist_mode == 'bag':
                    assist_mode = 'GAL-b'
                elif assist_mode == 'stack':
                    assist_mode = 'GAL-s'
                else:
                    raise ValueError('Not valid assist_mode')
                label_name = 'M={}, {}'.format(num_users, assist_mode)
                y = row.to_numpy()
                yerr = row_std.to_numpy()
                fig[df_name] = plt.figure(df_name)
                if metric_name == 'Gradient assisted learning rates':
                    plt.errorbar(x, y, color=color[assist_mode],
                                 linestyle=linestyle[assist_mode], label=label_name,
                                 marker=marker[assist_mode][local_epoch], markevery=markevery)
                else:
                    plt.errorbar(x, y, yerr=yerr / math.sqrt(num_experiments), capsize=capsize,
                                 color=color[assist_mode],
                                 linestyle=linestyle[assist_mode], label=label_name,
                                 marker=marker[assist_mode][local_epoch], markevery=markevery)
                plt.legend(loc=loc[metric_name], fontsize=fontsize['legend'])
                plt.xlabel('Assistance rounds', fontsize=fontsize['label'])
                plt.ylabel(metric_name, fontsize=fontsize['label'])
                plt.xticks(xticks, fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
        plt.grid()
        fig_path = '{}/{}.{}'.format(vis_path, df_name, save_format)
        makedir_exist_ok(vis_path)
        plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(df_name)
    return


def make_lc(df):
    metric_name_dict = {'CIFAR10': 'Accuracy', 'CIFAR100': 'Accuracy'}
    loc_dict = {'CIFAR10': 'lower right', 'CIFAR100': 'lower right'}
    data_split_mode_dict = {'iid': 'IID', 'non-iid-2': 'Non-IID'}
    fig = {}
    for df_name in df:
        data_name, model_name, num_clients, metric_name, stat = df_name.split('_')
        if stat == 'std':
            continue
        metric_name = metric_name_dict[data_name]
        for ((index, row), (_, row_std)) in zip(df[df_name].iterrows(), df[df_name].iterrows()):
            active_rate, data_split_mode, num_supervised, client_data_mode, threshold = index.split('_')
            data_split_mode_ = data_split_mode_dict[data_split_mode]
            label_name = '{}, $C={}, N_S={}$'.format(data_split_mode_, active_rate, num_supervised)
            y = row.to_numpy()[1:151]
            x = np.arange(len(y))
            fig[df_name] = plt.figure(df_name)
            plt.plot(x, y, label=label_name)
            plt.legend(loc=loc_dict[data_name], fontsize=fontsize)
            plt.xlabel('Communication Rounds', fontsize=fontsize)
            plt.ylabel(metric_name, fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        plt.grid()
        fig_path = '{}/{}.{}'.format(vis_path, fig_name, save_format)
        makedir_exist_ok(vis_path)
        plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()
