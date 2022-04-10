
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
                    'iterations_number': 400,
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

    images_dir = 'CBSD68-dataset-master/CBSD68/original_png/'
    all_files = os.listdir(images_dir)

    os.mkdir(flags.name_exp)
    global_name_exp = flags.name_exp + '/'

    scales = [(128,8), (128,16), (128,32), (128,64)]

    for scale in scales:
        flags.high_res = scale[0]
        flags.low_res = scale[1]

        for file_name in all_files:
            flags.image_path = images_dir + file_name
            str_name = str(file_name).replace('.', '_').replace('-', 'm')

            flags.sigma = 0.125
            flags.N = []
            flags.iterations_number = 400

            flags.projected = True
            flags.name_exp = global_name_exp + 'proj_mom_' + str(scale[0]) + '_' + str(scale[1]) + '_' + str_name
            run_MoM.run_MoM(flags, given_seed)

            flags.projected = False
            flags.name_exp = global_name_exp + 'not_proj_mom_' + str(scale[0]) + '_' + str(scale[1]) + '_' + str_name
            run_MoM.run_MoM(flags, given_seed)

            flags.sigma = 0.125
            flags.N = 10000
            flags.iterations_number = 100

            flags.projected = True
            flags.name_exp = global_name_exp + 'proj_em_' + str(scale[0]) + '_' + str(scale[1]) + '_' + str_name
            run_EM.run_EM(flags, given_seed)

            flags.projected = False
            flags.name_exp = global_name_exp + 'not_proj_em_' + str(scale[0]) + '_' + str(scale[1]) + '_' + str_name
            run_EM.run_EM(flags, given_seed)


if __name__ == "__main__":
    main()