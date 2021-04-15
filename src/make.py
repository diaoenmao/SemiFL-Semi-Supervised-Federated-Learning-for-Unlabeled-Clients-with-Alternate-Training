import argparse
import itertools

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--init_seed', default=0, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
parser.add_argument('--data', default=None, type=str)
parser.add_argument('--model', default=None, type=str)
parser.add_argument('--file', default=None, type=str)
args = vars(parser.parse_args())


def make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments, resume_mode,
                  control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = script_name + data_names + model_names + init_seeds + world_size + num_experiments + resume_mode + \
               control_names
    controls = list(itertools.product(*controls))
    return controls


def main():
    run = args['run']
    num_gpus = args['num_gpus']
    world_size = args['world_size']
    round = args['round']
    experiment_step = args['experiment_step']
    init_seed = args['init_seed']
    num_experiments = args['num_experiments']
    resume_mode = args['resume_mode']
    data = args['data']
    model = args['model']
    file = args['file']
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in list(range(0, num_gpus, world_size))]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    world_size = [[world_size]]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    if file == 'all':
        filename = '{}_{}_{}'.format(run, file, model)
        script_name = [['{}_classifier.py'.format(run)]]
        data_names = [[data]]
        model_names = [[model]]
        control_name = [[['1'], ['1'], ['none'], ['-1'], ['none'], ['none'], ['none']]]
        controls = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                 resume_mode, control_name)
    elif file == 't':
        filename = '{}_{}_{}'.format(run, file, model)
        script_name = [['{}_classifier.py'.format(run)]]
        data_names = [[data]]
        model_names = [[model]]
        control_name = [[['1'], ['1'], ['none'], ['40', '250', '4000'], ['none'], ['none'], ['none']]]
        controls = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                 resume_mode, control_name)
    elif file == 'ts':
        filename = '{}_{}_{}'.format(run, file, model)
        script_name = [['{}_classifier_ts.py'.format(run)]]
        data_names = [[data]]
        model_names = [[model]]
        control_name = [[['1'], ['1'], ['none'], ['40', '250', '4000'], [data], [model], ['0', '0.8']]]
        controls = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                 resume_mode, control_name)
    elif file == 'fed':
        filename = '{}_{}_{}'.format(run, file, model)
        script_name = [['{}_classifier_fed.py'.format(run)]]
        data_names = [['data']]
        model_names = [[model]]
        control_name = [[['100'], ['0.1'], ['iid'], ['40', '250', '4000'], [data], [model], ['0', '0.8']]]
        controls = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                 resume_mode, control_name)
    else:
        raise ValueError('Not valid file')
    s = '#!/bin/bash\n'
    k = 0
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --data_name {} --model_name {} --init_seed {} ' \
                '--world_size {} --num_experiments {} --resume_mode {} --control_name {}&\n'.format(
            gpu_ids[k % len(gpu_ids)], *controls[i])
        if k % round == round - 1:
            s = s[:-2] + '\nwait\n'
        k = k + 1
    if s[-5:-1] != 'wait':
        s = s + 'wait\n'
    print(s)
    run_file = open('./{}.sh'.format(filename), 'w')
    run_file.write(s)
    run_file.close()
    return


if __name__ == '__main__':
    main()
