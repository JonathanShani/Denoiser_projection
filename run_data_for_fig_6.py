

import sys
import os
import run_MoM
import numpy as np

def create_flags(args):
    special_cases = {'bool_vars': ['projected'],
                     'str_vars': ['alg_option', 'name_exp'],
                     'float_vars': ['lr', 'sigma']}

    default_vals = {'gpu_num': 0,
                    'lr': 0.01,
                    'sigma': 0.1,
                    'N': [],
                    'alg_option': 'BFGS',
                    'iterations_number': 400,
                    'projected': False,
                    'name_exp': 'tmp_name'}



    dict_flags = dict([convert_type(x.split('='), special_cases) for x in args])
    dict_flags = {**default_vals, **dict_flags}
    flags = Struct(**dict_flags)

    return flags

def convert_type(flag, special_cases):
    if flag[0] in special_cases['bool_vars']:
        flag[1] = flag[1] == 'True'
    elif flag[0] in special_cases['float_vars']:
        flag[1] = float(flag[1])
    elif flag[0] not in special_cases['str_vars']:
        flag[1] = int(flag[1])
    return flag

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def main():
    given_seed = 26723683
    flags = create_flags(sys.argv[1:])

    flags.image_path = 'lena.png'

    os.mkdir(flags.name_exp)
    global_name_exp = flags.name_exp + '/'


    all_log_sigma = np.linspace(-3,3,20)

    for test in range(100):
        for curr_log_sigma in all_log_sigma:
            flags.sigma = np.exp(curr_log_sigma)
            str_name = str(round(curr_log_sigma, 2)).replace('.', '_').replace('-', 'm')
            flags.name_exp = global_name_exp + 'test_num_' + str(test) + '_noise_' + str_name + '_proj_MOM'
            run_MoM.run_MoM(flags, given_seed+test)


if __name__ == "__main__":
    main()