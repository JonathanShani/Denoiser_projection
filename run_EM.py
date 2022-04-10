
import torch
import numpy as np
from PIL import Image
import time
from bm3d import bm3d
import os
import general_utils



def SR_EM_iteration(device, x_est, rho_est, data_sum, fft_data, sigma, K):
    """
    Execute one EM iteration
    :param device: The torch process device
    :param x_est: Current estimation of the image
    :param rho_est: Current estimation of the joint distribution
    :param data_sum: observations norm
    :param fft_data: fft of the data
    :param sigma: noise level
    :param K: Original and observed resolution ration
    :return: Updated estimation of the image and its likelihood
    """
    M = x_est.shape[0]
    m = int(np.sqrt(M))

    aa = torch.zeros(M).to(device)
    bb = torch.zeros(M).to(device)

    split_val = int(np.ceil(data_sum.shape[0] / 1000))
    N = int(data_sum.shape[0] / split_val)
    bccb_ind_mat = general_utils.index_matrix_for_bccb(m, m)
    bccb_ind_mat = general_utils.update_SR_model(m, int(m / (K ** 1)), bccb_ind_mat).to(device)

    W_mean = torch.zeros(m, m, split_val)
    for nn in range(split_val):
        TT = torch.zeros(m, m, N).to(device)
        curr_fft = fft_data[:, :, (nn * N):((nn + 1) * N)]
        curr_sum = data_sum[(nn * N):((nn + 1) * N)]

        for i in range(K):
            for j in range(K):
                xk = torch.transpose(x_est[bccb_ind_mat[:, j * m + i]].reshape((int(m / K), int(m / K))), 0, 1)
                est_sum = torch.sum(xk ** 2)

                fft_est = torch.transpose(torch.transpose(torch.fft.fft2(xk.cpu()).repeat(N, 1, 1), 0, 1), 1, 2).to(
                    device)
                CC = torch.abs(
                    torch.fft.ifft2(torch.mul(fft_est, torch.conj(curr_fft)).cpu(), dim=(0, 1)).to(device))

                tmp_vec = torch.remainder(-torch.arange(0, m, K), m).to(device)
                TT[i + tmp_vec.repeat(int(m / K)), j + tmp_vec.repeat_interleave(int(m / K))] = \
                    torch.transpose((-(curr_sum + est_sum - 2 * CC) / (2 * (sigma ** 2))), 0, 1).flatten(0, 1)


        exp_T = torch.sum(torch.exp(TT), [0, 1])
        like_value = torch.sum(torch.log(exp_T))

        WW = torch.mul(torch.exp(TT - TT.max(0)[0].max(0)[0]), torch.unsqueeze(rho_est, 2).repeat(1, 1, N)) + 1.e-20
        WW = WW / torch.sum(WW, (0, 1))

        for i in range(K):
            for j in range(K):
                aa[bccb_ind_mat[:, j * m + i]] = torch.sum(WW[torch.arange(i, m, K)][:, torch.arange(j, m, K)])

                ifft_calc = (torch.fft.ifft2((torch.mul(
                    torch.fft.fft2((WW[torch.arange(i, m, K), :][:, torch.arange(j, m, K)]).cpu(), dim=(0, 1)).to(
                        device),
                    torch.conj(curr_fft))).cpu(), dim=(0, 1))).to(device)

                bb[bccb_ind_mat[:, j * m + i]] = torch.transpose(torch.flip(torch.roll(
                    torch.abs(torch.sum(ifft_calc, 2)), (int(m / K) - 1, int(m / K) - 1), (0, 1)), (0, 1)), 0,
                    1).flatten()

        W_mean[:, :, nn] = torch.sum(WW, 2) / torch.sum(WW)

    rho_est = torch.mean(W_mean, 2).to(device)
    A = torch.diag(aa)
    x_new = torch.matmul(torch.pinverse(A), bb.unsqueeze(1))[:, 0]

    return x_new, like_value


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def run_EM(flags, given_seen):
    """
    Execute EM optimization
    :param flags: execution parameters
    :param given_seed: desired seed given to numpy and torch
    :return: None
    """
    np.random.seed(given_seen)
    torch.manual_seed(given_seen)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    device = general_utils.update_device(flags.gpu_num)


    # load data
    Lh = flags.high_res

    os.mkdir(flags.name_exp)

    img = Image.open(flags.image_path).convert('L')
    img = img.resize((Lh, Lh), Image.ANTIALIAS)
    img = np.array(img.getdata()).reshape(Lh, Lh) / 255
    img = torch.FloatTensor(img).to(device)

    # generate random rho vectors
    real_rho_1 = np.random.rand(Lh, 1)
    real_rho_2 = np.random.rand(1, Lh)

    real_rho_1 = real_rho_1 / np.sum(real_rho_1)
    real_rho_2 = real_rho_2 / np.sum(real_rho_2)

    np.save(flags.name_exp + '/real_vars.npy', {'real_rho_1': real_rho_1, 'real_rho_2': real_rho_2, 'img': img})

    # collect observations
    P_C, tot_rho, _ = general_utils.generate_clean_shifted_images(device, img, real_rho_1, real_rho_2, Lh, flags.low_res)
    data = general_utils.generate_observations(P_C, flags.sigma, tot_rho, flags.N, device)
    data_im = torch.transpose(data.reshape(flags.low_res, flags.low_res, data.shape[1]), 0, 1)
    data_sum = torch.sum(data_im ** 2, (0, 1))
    fft_data = torch.fft.fft2(data_im.cpu(), dim=(0, 1)).to(device)


    max_likelihood = np.zeros(flags.iterations_number)
    error_val = np.zeros(flags.iterations_number)
    process_time = []
    t0 = time.time()

    initial_start = np.random.rand(Lh * Lh)
    initial_start_x = initial_start / np.max(initial_start)
    x_init = torch.FloatTensor(initial_start_x).to(device)

    init_rho_tot = np.random.rand(Lh, Lh)
    init_rho_tot = init_rho_tot / np.sum(init_rho_tot)
    init_rho_tot = torch.FloatTensor(init_rho_tot).to(device)

    for k in range(flags.iterations_number):
        x_curr_est, max_likelihood[k] = SR_EM_iteration(device, x_init, init_rho_tot, data_sum, fft_data,
                                                        flags.sigma, int(Lh / flags.low_res))

        rolled_img, curr_error = general_utils.evaluate_data(device, x_curr_est, Lh, Lh, img)
        error_val[k] = curr_error
        process_time.extend([time.time() - t0])

        curr_data = {'like_value': max_likelihood, 'error': error_val, 'curr_x': x_curr_est.cpu().detach().numpy(),
                     'curr_shifted_image': rolled_img.cpu().detach().numpy()}

        np.save(flags.name_exp + '/eval_vec.npy', {'like_value': max_likelihood, 'error': error_val,
                                                   'process_time': process_time})
        np.save(flags.name_exp + '/final_res.npy', curr_data)

        if flags.projected and np.mod(k + 1, 5) == 0:
            chosen_den = max(2 ** (-0.1 * (k + 1) / 5), 0.02)
            print('used den - ' + str(chosen_den))
            curr_img = torch.transpose(torch.reshape(x_curr_est, [Lh, Lh]), 0, 1)
            denoise_im = bm3d(np.atleast_3d(curr_img.cpu().detach().numpy()), chosen_den)
            x_init = torch.FloatTensor(np.transpose(denoise_im).flatten()).to(device)
        else:
            x_init = x_curr_est

