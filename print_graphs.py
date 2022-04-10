
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



def main():

    # create figure #1
    os.mkdir('fig_1_output')
    L_h = 128
    L_l = 64
    shift_ind = np.floor(L_h * np.random.rand(4, 2))
    images = ['lena.png', 'cameraman.png', 'fruits.png', 'cryo_em_img_1.png']
    for ind, im in enumerate(images):
        img = Image.open(im)
        img = img.convert('L').resize((L_h, L_h), Image.ANTIALIAS)
        img = np.array(img.getdata()).reshape(img.size[0], img.size[1]) / 255

        shift_img = np.roll(img, (int(shift_ind[ind][0]), int(shift_ind[ind][1])), axis=(1, 0))

        base_mat = np.zeros((int(L_h/L_l), int(L_h/L_l)))
        base_mat[0, 0] = 1
        sample_mat = np.kron(np.ones((L_l, L_l)), base_mat)
        sampled_img = np.reshape(shift_img[sample_mat == 1], (L_l, L_l))

        low_noise = sampled_img + 0.125 * np.random.randn(L_l, L_l)
        high_noise = sampled_img + 0.5 * np.random.randn(L_l, L_l)

        plt.imsave('fig_1_output/image_' + str(im) +'_original.png', img, cmap='gray')
        plt.imsave('fig_1_output/image_' + str(im) +'_sampled.png', sampled_img, cmap='gray')
        plt.imsave('fig_1_output/image_' + str(im) +'_low_noise.png', low_noise, cmap='gray')
        plt.imsave('fig_1_output/image_' + str(im) +'_high_noise.png', high_noise, cmap='gray')


    # create figure #2
    os.mkdir('fig_2_output')
    for method in ['mom', 'em']:
        for file_name in ['lena.png', 'cameraman.png', 'fruits.png', 'cryo_em_img_1.png']:
            for version in ['proj', 'not_proj']:
                str_name = str(file_name).replace('.', '_').replace('-', 'm')
                res = np.load('visual_output/' + version + '_' + method + '_128_64_' + str_name +
                              '/final_res.npy', allow_pickle=True).item()
                plt.imsave('fig_2_output/' +  version + '_' + method + '_128_64_' + str_name + '.png',
                           res['curr_shifted_image'], cmap='gray')

    # create figure #3 and #4
    l_list = [64, 32, 16, 8]
    all_results_em_proj = [[],[],[],[]]
    all_results_em_not_proj = [[],[],[],[]]
    all_results_mom_proj = [[],[],[],[]]
    all_results_mom_not_proj = [[],[],[],[]]
    os.mkdir('fig_3_output')
    for i, l in enumerate(l_list):
        for t in range(68):
            res = np.load('dataset_all_scales/proj_mom_128_' + str(l) + '_00' + str(t).zfill(2) +
                          '_png/eval_vec.npy', allow_pickle=True).item()
            all_results_mom_proj[i].append(res['error'])
            res = np.load('dataset_all_scales/not_proj_mom_128_' + str(l) + '_00' + str(t).zfill(2) +
                          '_png/eval_vec.npy', allow_pickle=True).item()
            all_results_mom_not_proj[i].append(res['error'])
            res = np.load('dataset_all_scales/proj_em_128_' + str(l) + '_00' + str(t).zfill(2) +
                          '_png/eval_vec.npy', allow_pickle=True).item()
            all_results_em_proj[i].append(res['error'])
            res = np.load('dataset_all_scales/not_proj_em_128_' + str(l) + '_00' + str(t).zfill(2) +
                          '_png/eval_vec.npy', allow_pickle=True).item()
            all_results_em_not_proj[i].append(res['error'])
            if l == 64:
                res = np.load('dataset_all_scales/proj_em_128_' + str(l) + '_00' + str(t).zfill(2) +
                              '_png/final_res.npy', allow_pickle=True).item()
                plt.imsave('fig_3_output/proj_em_' + str(t).zfill(2) + '.png', res['curr_shifted_image'], cmap='gray')
                res = np.load('dataset_all_scales/proj_mom_128_' + str(l) + '_00' + str(t).zfill(2) +
                              '_png/final_res.npy', allow_pickle=True).item()
                plt.imsave('fig_3_output/proj_mom_' + str(t).zfill(2) + '.png', res['curr_shifted_image'], cmap='gray')

    fig, axs = plt.subplots(2, 1)
    for l in range(len(l_list)):
        it = np.array(range(len(all_results_mom_proj[l][0])))
        axs[0].plot(it, np.mean([x + [x[-1]] * (it[-1] + 1 - len(x)) for x in all_results_mom_not_proj[l]], 0))
        axs[0].plot(it, np.mean(all_results_mom_proj[l], 0))
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')

        it = np.array(range(len(all_results_em_proj[l][0])))
        axs[1].plot(it, np.mean(all_results_em_not_proj[l], 0))
        axs[1].plot(it, np.mean(all_results_em_proj[l], 0))
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')

    axs[0].legend(['unprojected method of moments , ' + r'$\mathit{L}_{low}$=64',
                   'projected method of moments , ' + r'$\mathit{L}_{low}$=64',
                   'unprojected method of moments , ' + r'$\mathit{L}_{low}$=32',
                   'projected method of moments , ' + r'$\mathit{L}_{low}$=32',
                   'unprojected method of moments , ' + r'$\mathit{L}_{low}$=16',
                   'projected method of moments , ' + r'$\mathit{L}_{low}$=16',
                   'unprojected method of moments , ' + r'$\mathit{L}_{low}$=8',
                   'projected method of moments , ' + r'$\mathit{L}_{low}$=8'
                   ], fontsize=8)
    axs[0].set_xlabel('iterations', fontsize=16)
    axs[0].set_ylabel('mean error', fontsize=16)
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].yaxis.set_major_formatter(ticker.ScalarFormatter())

    axs[1].legend(['unprojected EM , ' + r'$\mathit{L}_{low}$=64', 'projected EM , ' + r'$\mathit{L}_{low}$=64',
                'unprojected EM , ' + r'$\mathit{L}_{low}$=32', 'projected EM , ' + r'$\mathit{L}_{low}$=32',
                'unprojected EM , ' + r'$\mathit{L}_{low}$=16', 'projected EM , ' + r'$\mathit{L}_{low}$=16',
                'unprojected EM , ' + r'$\mathit{L}_{low}$=8', 'projected EM , ' + r'$\mathit{L}_{low}$=8'
                ],fontsize=8)
    axs[1].set_xlabel('iterations',fontsize=16)
    axs[1].set_ylabel('mean error',fontsize=16)
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    yticks = np.array([0.05,0.07,0.1,0.3,0.5])
    axs[1].set_yticks(yticks)
    axs[1].yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.tight_layout()
    plt.savefig('figure_4.png')


    # create figure #5
    os.mkdir('fig_5_output')
    l_list = [64, 32, 16]
    for file_name in ['lena.png', 'cameraman.png', 'fruits.png', 'cryo_em_img_1.png']:
        str_name = str(file_name).replace('.', '_').replace('-', 'm')
        for l in l_list:
            res = np.load('visual_output/proj_em_128_' + str(l) + '_' + str_name +
                          '/final_res.npy', allow_pickle=True).item()
            plt.imsave('fig_5_output/proj_em_128_' + str(l) + '_' + str_name + '.png',
                       res['curr_shifted_image'], cmap='gray')


    # create figure #6
    all_log_sigma = np.linspace(-3, 3, 20)
    error_per_sigma = np.zeros(len(all_log_sigma))
    for l in range(len(all_log_sigma)):
        str_name = str(round(all_log_sigma[l], 2)).replace('.', '_').replace('-', 'm')
        all_results = []
        for t in range(100):
            res = np.load('noise_influence/test_num_' + str(t) + '_noise_' + str_name + '_proj_MOM/eval_vec.npy',
                          allow_pickle=True).item()
            all_results.append(res['error'][-1])
        error_per_sigma[l] = np.mean(all_results)

    img = Image.open('lena.png')
    L = 32
    img = img.resize((L, L), Image.ANTIALIAS)
    img = np.array(img.getdata()).reshape(img.size[0], img.size[1]) / 255
    norm_img = np.linalg.norm(img)

    fig, axs = plt.subplots(1, 1)
    axs.plot((norm_img ** 2) / ((L * np.exp(all_log_sigma)) ** 2), error_per_sigma, linewidth=3)
    axs.set_xlabel('SNR', fontsize=16)
    axs.set_ylabel('mean error', fontsize=16)
    axs.set_xscale('log')
    axs.set_yscale('log')
    plt.tight_layout()
    plt.savefig('figure_6.png')


    # create figure #7
    all_log_N = np.linspace(2, 6, 20)
    error_per_N = np.zeros(len(all_log_N))
    for n in range(len(all_log_N)):
        str_name = str(round(all_log_N[n], 2)).replace('.', '_')
        all_results = []
        for t in range(100):
            res = np.load('N_influence/test_num_' + str(t) + '_N' + str_name + '_proj_MOM/eval_vec.npy',
                          allow_pickle=True).item()
            all_results.append(res['error'][-1])
        error_per_N[n] = np.mean(all_results)

    fig, axs = plt.subplots(1, 1)
    axs.plot(10**all_log_N, error_per_N, linewidth=3)
    axs.set_xlabel('N', fontsize=16)
    axs.set_ylabel('mean error', fontsize=16)
    axs.set_xscale('log')
    axs.set_yscale('log')
    yticks = [0.2,0.25,0.3,0.35,0.4]
    axs.set_yticks(yticks)
    axs.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    plt.tight_layout()
    plt.savefig('figure_7.png')


    # create figure #8
    all_it = [1, 5, 10, 20, 50, 100]
    fig, axs = plt.subplots(1, 1)
    for it in all_it:
        all_results = []
        for t in range(10):
            res = np.load('F_influence/test_num_' + str(t) + '_it_' + str(it) + '_proj_MOM/eval_vec.npy',
                          allow_pickle=True).item()
            all_results.append(res['error'])
        convergence_curve = np.mean(all_results,0)
        x = np.array(range(1,1+len(convergence_curve)))
        axs.set_xscale('log')
        axs.set_yscale('log')
        axs.plot(x, convergence_curve, linewidth=2)
    axs.legend([r'$\mathit{F}=$' + str(x) for x in all_it])
    axs.set_xlabel('iterations',fontsize=16)
    axs.set_ylabel('mean error',fontsize=16)
    plt.tight_layout()
    plt.savefig('figure_8.png')


    # plot table 1
    all_sigma = [0.125, 0.25, 0.5]
    all_N = [100, 10000]
    scales = [(128, 32), (128, 16), (64, 32)]

    for sigma in all_sigma:
        for N in all_N:
            for scale in scales:
                final_res = np.zeros(4)
                for t in range(num_trails):
                    res_mom = np.load('proj_mom_em_comparison/' + str(scale[0]) + '_' + str(scale[1]) + '_sigma_' +
                                      str(sigma).replace('.', '_') + '_N_' + str(N) + '_t_' + str(t) +
                                      '_proj_MOM/eval_vec.npy', allow_pickle=True).item()
                    res_em = np.load('proj_mom_em_comparison/' + str(scale[0]) + '_' + str(scale[1]) + '_sigma_' +
                                      str(sigma).replace('.', '_') + '_N_' + str(N) + '_t_' + str(t) +
                                      '_proj_EM/eval_vec.npy', allow_pickle=True).item()
                    final_res[0] = final_res[0] + res_mom['error'][-1]
                    final_res[1] = final_res[1] + res_em['error'][-1]
                    final_res[2] = final_res[2] + res_mom['process_time'][-1]
                    final_res[3] = final_res[3] + res_em['process_time'][-1]
                print('sigma = ' + str(sigma) + ', N = ' + str(N) + ' scale = ' + str(scale[0]) + '_' + str(scale[1]))
                print(final_res / num_trails)



if __name__ == "__main__":
    main()