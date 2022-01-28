import os
import itertools
import json
import numpy as np
import pandas as pd
from utils import save, load, makedir_exist_ok
import matplotlib.pyplot as plt
from collections import defaultdict

result_path = './output/result'
save_format = 'png'
vis_path = './output/vis/{}'.format(save_format)
num_experiments = 4
exp = [str(x) for x in list(range(num_experiments))]


def make_controls(data_names, model_names, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + data_names + model_names + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(file):
    if file == 'fs':
        control_name = [[['fs']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls = make_controls(data_names, model_names, control_name)
        data_names = [['SVHN']]
        model_names = [['wresnet28x2']]
        svhn_controls = make_controls(data_names, model_names, control_name)
        data_names = [['CIFAR100']]
        model_names = [['wresnet28x8']]
        cifar100_controls = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls + svhn_controls + cifar100_controls
    elif file == 'ps':
        control_name = [[['250', '4000']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls = make_controls(data_names, model_names, control_name)
        control_name = [[['250', '1000']]]
        data_names = [['SVHN']]
        model_names = [['wresnet28x2']]
        svhn_controls = make_controls(data_names, model_names, control_name)
        control_name = [[['2500', '10000']]]
        data_names = [['CIFAR100']]
        model_names = [['wresnet28x8']]
        cifar100_controls = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls + svhn_controls + cifar100_controls
    elif file == 'cd':
        control_name = [[['250', '4000'], ['fix-mix'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['5'], ['0.5'], ['1']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls = make_controls(data_names, model_names, control_name)
        control_name = [[['250', '1000'], ['fix-mix'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['5'], ['0.5'], ['1']]]
        data_names = [['SVHN']]
        model_names = [['wresnet28x2']]
        svhn_controls = make_controls(data_names, model_names, control_name)
        control_name = [[['2500', '10000'], ['fix-mix'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['5'], ['0.5'],
                         ['1']]]
        data_names = [['CIFAR100']]
        model_names = [['wresnet28x8']]
        cifar100_controls = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls + svhn_controls + cifar100_controls
    elif file == 'ub':
        control_name = [
            [['250', '4000'], ['fix-mix'], ['100'], ['0.1'], ['non-iid-d-0.1', 'non-iid-d-0.3'], ['5'], ['0.5'], ['1']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls = make_controls(data_names, model_names, control_name)
        control_name = [
            [['250', '1000'], ['fix-mix'], ['100'], ['0.1'], ['non-iid-d-0.1', 'non-iid-d-0.3'], ['5'], ['0.5'], ['1']]]
        data_names = [['SVHN']]
        model_names = [['wresnet28x2']]
        svhn_controls = make_controls(data_names, model_names, control_name)
        control_name = [[['2500', '10000'], ['fix-mix'], ['100'], ['0.1'], ['non-iid-d-0.1', 'non-iid-d-0.3'], ['5'],
                         ['0.5'], ['1']]]
        data_names = [['CIFAR100']]
        model_names = [['wresnet28x8']]
        cifar100_controls = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls + svhn_controls + cifar100_controls
    elif file == 'loss':
        control_name = [[['4000'], ['fix'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['5'], ['0.5'], ['1']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls
    elif file == 'local-epoch':
        control_name = [[['4000'], ['fix-mix'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['1'], ['0.5'], ['1']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls
    elif file == 'gm':
        control_name = [[['4000'], ['fix-mix'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['5'], ['0'], ['1']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls
    elif file == 'sbn':
        control_name = [[['250', '4000'], ['fix-mix'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['5'], ['0.5'], ['0']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls
    elif file == 'alternate':
        control_name = [[['4000'], ['fix-batch'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['5'], ['0.5'],
                         ['1']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls_1 = make_controls(data_names, model_names, control_name)
        control_name = [[['4000'], ['fix', 'fix-batch'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['5'], ['0.5'],
                         ['1'], ['0']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls_2 = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls_1 + cifar10_controls_2
    elif file == 'fl':
        control_name = [
            [['fs'], ['sup'], ['100'], ['0.1'], ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'], ['5'],
             ['0.5'], ['1']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls = make_controls(data_names, model_names, control_name)
        control_name = [
            [['fs'], ['sup'], ['100'], ['0.1'], ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'], ['5'],
             ['0.5'], ['1']]]
        data_names = [['SVHN']]
        model_names = [['wresnet28x2']]
        svhn_controls = make_controls(data_names, model_names, control_name)
        control_name = [
            [['fs'], ['sup'], ['100'], ['0.1'], ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'], ['5'],
             ['0.5'], ['1']]]
        data_names = [['CIFAR100']]
        model_names = [['wresnet28x8']]
        cifar100_controls = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls + svhn_controls + cifar100_controls
    elif file == 'fsgd':
        control_name = [[['4000'], ['fix-fsgd'], ['100'], ['0.1'], ['iid', 'non-iid-l-2'], ['0'], ['0'], ['1']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls
    elif file == 'frgd':
        control_name = [
            [['250', '4000'], ['fix-frgd'], ['100'], ['0.1'], ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'],
             ['5'], ['0.5'], ['1'], ['0']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls = make_controls(data_names, model_names, control_name)
        control_name = [
            [['250', '1000'], ['fix-frgd'], ['100'], ['0.1'], ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'],
             ['5'], ['0.5'], ['1'], ['0']]]
        data_names = [['SVHN']]
        model_names = [['wresnet28x2']]
        svhn_controls = make_controls(data_names, model_names, control_name)
        control_name = [[['2500', '10000'], ['fix-frgd'], ['100'], ['0.1'],
                         ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'], ['5'], ['0.5'], ['1'], ['0']]]
        data_names = [['CIFAR100']]
        model_names = [['wresnet28x8']]
        cifar100_controls = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls + svhn_controls + cifar100_controls
    elif file == 'fmatch':
        control_name = [[['250', '4000'], ['fix-fmatch'], ['100'], ['0.1'],
                         ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'], ['5'], ['0.5'], ['1'], ['0']]]
        data_names = [['CIFAR10']]
        model_names = [['wresnet28x2']]
        cifar10_controls = make_controls(data_names, model_names, control_name)
        control_name = [[['250', '1000'], ['fix-fmatch'], ['100'], ['0.1'],
                         ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'], ['5'], ['0.5'], ['1'], ['0']]]
        data_names = [['SVHN']]
        model_names = [['wresnet28x2']]
        svhn_controls = make_controls(data_names, model_names, control_name)
        control_name = [[['2500', '10000'], ['fix-fmatch'], ['100'], ['0.1'],
                         ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3', 'non-iid-l-2'], ['5'], ['0.5'], ['1'], ['0']]]
        data_names = [['CIFAR100']]
        model_names = [['wresnet28x8']]
        cifar100_controls = make_controls(data_names, model_names, control_name)
        controls = cifar10_controls + svhn_controls + cifar100_controls
    else:
        raise ValueError('Not valid file')
    return controls


def main():
    files = ['fs', 'ps', 'cd', 'ub', 'loss', 'local-epoch', 'gm', 'sbn', 'alternate', 'fl', 'fsgd', 'frgd', 'fmatch']
    controls = []
    for file in files:
        controls += make_control_list(file)
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
    make_vis(df_exp, df_history)
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
        filter_length = []
        for i in range(len(processed_result[pivot])):
            x = processed_result[pivot][i]
            if len(processed_result[pivot][i]) in [400, 800]:
                filter_length.append(x)
            elif len(processed_result[pivot][i]) == 801:
                filter_length.append(x[:800])
            else:
                filter_length.append(x + [x[-1]] * (800 - len(x)))
        processed_result[pivot] = filter_length
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


def write_xlsx(path, df, startrow=0):
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return


def make_df_exp(extracted_processed_result_exp):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_exp:
        control = exp_name.split('_')
        if len(control) == 3:
            data_name, model_name, num_supervised = control
            index_name = ['1']
            df_name = '_'.join([data_name, model_name, num_supervised])
        elif len(control) == 10:
            data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, \
            local_epoch, gm, sbn = control
            index_name = ['_'.join([local_epoch, gm])]
            df_name = '_'.join(
                [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, sbn])
        elif len(control) == 11:
            data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, \
            local_epoch, gm, sbn, ft = control
            index_name = ['_'.join([local_epoch, gm])]
            df_name = '_'.join(
                [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, sbn,
                 ft])
        else:
            raise ValueError('Not valid control')
        df[df_name].append(pd.DataFrame(data=extracted_processed_result_exp[exp_name], index=index_name))
    write_xlsx('{}/result_exp.xlsx'.format(result_path), df)
    return df


def make_df_history(extracted_processed_result_history):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_history:
        control = exp_name.split('_')
        if len(control) == 3:
            data_name, model_name, num_supervised = control
            index_name = ['1']
            for k in extracted_processed_result_history[exp_name]:
                df_name = '_'.join([data_name, model_name, num_supervised, k])
                df[df_name].append(
                    pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
        elif len(control) == 10:
            data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, \
            local_epoch, gm, sbn = control
            index_name = ['_'.join([local_epoch, gm])]
            for k in extracted_processed_result_history[exp_name]:
                df_name = '_'.join(
                    [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode,
                     sbn, k])
                df[df_name].append(
                    pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
        elif len(control) == 11:
            data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, \
            local_epoch, gm, sbn, ft = control
            index_name = ['_'.join([local_epoch, gm])]
            for k in extracted_processed_result_history[exp_name]:
                df_name = '_'.join(
                    [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode,
                     sbn, ft, k])
                df[df_name].append(
                    pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
        else:
            raise ValueError('Not valid control')
    write_xlsx('{}/result_history.xlsx'.format(result_path), df)
    return df


def make_vis(df_exp, df_history):
    data_split_mode_dict = {'iid': 'IID', 'non-iid-l-2': 'Non-IID, $K=2$',
                            'non-iid-d-0.1': 'Non-IID, $\operatorname{Dir}(0.1)$',
                            'non-iid-d-0.3': 'Non-IID, $\operatorname{Dir}(0.3)$', 'fix-fsgd': 'FedSGD + FixMatch',
                            'fix-batch': 'FedAvg + FixMatch', 'fs': 'Fully Supervised', 'ps': 'Partially Supervised'}
    color = {'5_0.5': 'red', '1_0.5': 'orange', '5_0': 'dodgerblue', '5_0.9': 'blue', '5_0.5_nomixup': 'green',
             '5_0_nomixup': 'green', 'iid': 'red', 'non-iid-l-2': 'orange', 'non-iid-d-0.1': 'dodgerblue',
             'non-iid-d-0.3': 'green', 'fix-fsgd': 'red', 'fix-batch': 'blue',
             'fs': 'black', 'ps': 'orange'}
    linestyle = {'5_0.5': '-', '1_0.5': '--', '5_0': ':', '5_0.5_nomixup': '-.', '5_0_nomixup': '-.',
                 '5_0.9': (0, (1, 5)), 'iid': '-', 'non-iid-l-2': '--', 'non-iid-d-0.1': '-.', 'non-iid-d-0.3': ':',
                 'fix-fsgd': '--', 'fix-batch': ':', 'fs': '-', 'ps': '-.'}
    loc_dict = {'Accuracy': 'lower right', 'Loss': 'upper right'}
    fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
    fig = {}
    reorder_fig = []
    for df_name in df_history:
        df_name_list = df_name.split('_')
        if len(df_name_list) == 5:
            data_name, model_name, num_supervised, metric_name, stat = df_name.split('_')
            if stat == 'std':
                continue
            df_name_std = '_'.join([data_name, model_name, num_supervised, metric_name, 'std'])
            fig_name = '_'.join([data_name, model_name, num_supervised, metric_name])
            fig[fig_name] = plt.figure(fig_name)
            for ((index, row), (_, row_std)) in zip(df_history[df_name].iterrows(), df_history[df_name_std].iterrows()):
                y = row.to_numpy()
                yerr = row_std.to_numpy()
                x = np.arange(len(y))
                plt.plot(x, y, color='r', linestyle='-')
                plt.fill_between(x, (y - yerr), (y + yerr), color='r', alpha=.1)
                plt.xlabel('Communication Rounds', fontsize=fontsize['label'])
                plt.ylabel(metric_name, fontsize=fontsize['label'])
                plt.xticks(fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
        elif len(df_name_list) == 10:
            data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, sbn, \
            metric_name, stat = df_name.split('_')
            if stat == 'std':
                continue
            df_name_std = '_'.join(
                [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, sbn,
                 metric_name, 'std'])
            for ((index, row), (_, row_std)) in zip(df_history[df_name].iterrows(), df_history[df_name_std].iterrows()):
                y = row.to_numpy()
                yerr = row_std.to_numpy()
                x = np.arange(len(y))
                if index == '5_0.5' and loss_mode == 'fix-mix':
                    fig_name = '_'.join(
                        [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, sbn,
                         metric_name])
                    reorder_fig.append(fig_name)
                    label_name = '{}'.format(data_split_mode_dict[data_split_mode])
                    style = data_split_mode
                    fig[fig_name] = plt.figure(fig_name)
                    plt.plot(x, y, color=color[style], linestyle=linestyle[style], label=label_name)
                    plt.fill_between(x, (y - yerr), (y + yerr), color=color[style], alpha=.1)
                    plt.legend(loc=loc_dict[metric_name], fontsize=fontsize['legend'])
                    plt.xlabel('Communication Rounds', fontsize=fontsize['label'])
                    plt.ylabel(metric_name, fontsize=fontsize['label'])
                    plt.xticks(fontsize=fontsize['ticks'])
                    plt.yticks(fontsize=fontsize['ticks'])
                if data_split_mode in ['iid', 'non-iid-l-2'] and loss_mode not in ['fix-batch', 'fix-fsgd', 'fix-frgd']:
                    fig_name = '_'.join(
                        [data_name, model_name, num_supervised, num_clients, active_rate, data_split_mode, sbn,
                         metric_name])
                    reorder_fig.append(fig_name)
                    fig[fig_name] = plt.figure(fig_name)
                    local_epoch, gm = index.split('_')
                    if loss_mode == 'fix':
                        label_name = '$E={}$, $\\beta_g={}$, No mixup'.format(local_epoch, gm)
                        style = '{}_nomixup'.format(index)
                    else:
                        label_name = '$E={}$, $\\beta_g={}$'.format(local_epoch, gm)
                        style = index
                    plt.plot(x, y, color=color[style], linestyle=linestyle[style], label=label_name)
                    plt.fill_between(x, (y - yerr), (y + yerr), color=color[style], alpha=.1)
                    plt.legend(loc=loc_dict[metric_name], fontsize=fontsize['legend'])
                    plt.xlabel('Communication Rounds', fontsize=fontsize['label'])
                    plt.ylabel(metric_name, fontsize=fontsize['label'])
                    plt.xticks(fontsize=fontsize['ticks'])
                    plt.yticks(fontsize=fontsize['ticks'])
                if data_split_mode in ['iid', 'non-iid-l-2'] and loss_mode == 'fix-fsgd':
                    fix_batch_df_name = '_'.join(
                        [data_name, model_name, num_supervised, 'fix-batch', num_clients, active_rate, data_split_mode,
                         sbn, '0', metric_name, 'mean'])
                    fix_batch_df_name_std = '_'.join(
                        [data_name, model_name, num_supervised, 'fix-batch', num_clients, active_rate, data_split_mode,
                         sbn, '0', metric_name, 'std'])
                    fix_batch_y = list(df_history[fix_batch_df_name].iterrows())[0][1]
                    fix_batch_y_yerr = list(df_history[fix_batch_df_name_std].iterrows())[0][1]
                    fs_df_name = '_'.join([data_name, model_name, 'fs'])
                    fs_df_name_std = '_'.join([data_name, model_name, 'fs'])
                    fs_y = list(df_exp[fs_df_name].iterrows())[0][1]['{}_mean'.format(metric_name)]
                    fs_y_yerr = list(df_exp[fs_df_name_std].iterrows())[0][1]['{}_std'.format(metric_name)]
                    ps_df_name = '_'.join([data_name, model_name, num_supervised])
                    ps_df_name_std = '_'.join([data_name, model_name, num_supervised])
                    ps_y = list(df_exp[ps_df_name].iterrows())[0][1]['{}_mean'.format(metric_name)]
                    ps_y_yerr = list(df_exp[ps_df_name_std].iterrows())[0][1]['{}_std'.format(metric_name)]
                    fig_name = '_'.join(
                        [data_name, model_name, num_supervised, num_clients, active_rate, data_split_mode, sbn,
                         metric_name, 'fsgd'])
                    reorder_fig.append(fig_name)
                    fig[fig_name] = plt.figure(fig_name)
                    label_name = '{}'.format(data_split_mode_dict['fix-fsgd'])
                    style = 'fix-fsgd'
                    plt.plot(x, y, color=color[style], linestyle=linestyle[style], label=label_name)
                    plt.fill_between(x, (y - yerr), (y + yerr), color=color[style], alpha=.1)
                    label_name = '{}'.format(data_split_mode_dict['fix-batch'])
                    style = 'fix-batch'
                    plt.plot(x, fix_batch_y, color=color[style], linestyle=linestyle[style], label=label_name)
                    plt.fill_between(x, (fix_batch_y - fix_batch_y_yerr), (fix_batch_y + fix_batch_y_yerr),
                                     color=color[style], alpha=.1)
                    label_name = '{}'.format(data_split_mode_dict['fs'])
                    style = 'fs'
                    plt.plot(x, np.repeat(fs_y, len(x)), color=color[style], linestyle=linestyle[style],
                             label=label_name)
                    plt.fill_between(x, np.repeat(fs_y - fs_y_yerr, len(x)), np.repeat(fs_y + fs_y_yerr, len(x)),
                                     color=color[style], alpha=.1)
                    label_name = '{}'.format(data_split_mode_dict['ps'])
                    style = 'ps'
                    plt.plot(x, np.repeat(ps_y, len(x)), color=color[style], linestyle=linestyle[style],
                             label=label_name)
                    plt.fill_between(x, np.repeat(ps_y - ps_y_yerr, len(x)), np.repeat(ps_y + ps_y_yerr, len(x)),
                                     color=color[style], alpha=.1)
                    plt.legend(loc=loc_dict[metric_name], fontsize=fontsize['legend'])
                    plt.xlabel('Communication Rounds', fontsize=fontsize['label'])
                    plt.ylabel(metric_name, fontsize=fontsize['label'])
                    plt.xticks(fontsize=fontsize['ticks'])
                    plt.yticks(fontsize=fontsize['ticks'])
    for fig_name in reorder_fig:
        fig_name_list = fig_name.split('_')
        data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, sbn, metric_name = fig_name_list[:8]
        plt.figure(fig_name)
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(fig_name_list) == 9:
            if len(handles) == 4:
                handles = [handles[2], handles[3], handles[0], handles[1]]
                labels = [labels[2], labels[3], labels[0], labels[1]]
                plt.legend(handles, labels, loc=loc_dict[metric_name], fontsize=fontsize['legend'])
        else:
            if len(handles) == 4:
                handles = [handles[0], handles[3], handles[2], handles[1]]
                labels = [labels[0], labels[3], labels[2], labels[1]]
                plt.legend(handles, labels, loc=loc_dict[metric_name], fontsize=fontsize['legend'])
            if len(handles) == 5:
                handles = [handles[0], handles[4], handles[2], handles[3], handles[1]]
                labels = [labels[0], labels[4], labels[2], labels[3], labels[1]]
                plt.legend(handles, labels, loc=loc_dict[metric_name], fontsize=fontsize['legend'])
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
