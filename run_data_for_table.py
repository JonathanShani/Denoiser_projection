
import sys
import os
import run_MoM
import run_EM

def create_flags(args):
    special_cases = {'bool_vars': ['projected'],
                     'str_vars': ['alg_option', 'name_exp'],
                     'float_vars': ['lr', 'sigma']}

    default_vals = {'gpu_num': 0,
                    'lr': 0.01,
                    'sigma': 0.1,
                    'N': [],
                    'alg_option': 'BFGS',
                    'iterations_number': 100,
                    'projected': False,
                    'name_exp': 'tmp_name'}

    dict_flags = dict([convert_type(x.split('='), special_cases) for x in args])
    dict_flags = {**default_vals, **dict_flags}
    flags = Struct(**dict_flags)

    assert flags.alg_option in ['GD', 'BFGS'], "wrong optimization algorithm"

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


    all_sigma = [0.125, 0.25, 0.5]
    all_N = [100, 10000]
    scales = [(128,32), (128,16), (64,32)]

    for t in range(20):
        for sigma in all_sigma:
            for N in all_N:
                for scale in scales:
                    flags.sigma = sigma
                    flags.N = N
                    flags.high_res = scale[0]
                    flags.low_res = scale[1]
                    flags.name_exp = global_name_exp + str(scale[0]) + '_' + str(scale[1]) + '_sigma_' + \
                        str(sigma).replace('.', '_') + '_N_' + str(N) + '_t_' + str(t) +'_proj_MOM'
                    run_MoM.run_MoM(flags, given_seed)

                    flags.name_exp = global_name_exp + str(scale[0]) + '_' + str(scale[1]) + '_sigma_' + \
                        str(sigma).replace('.', '_') + '_N_' + str(N) + '_t_' + str(t) +'_proj_EM'
                    run_EM.run_EM(flags, given_seed)



if __name__ == "__main__":
    main()