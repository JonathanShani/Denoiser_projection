
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
import time
from bm3d import bm3d
import os
import general_utils


def define_optimization_alg(alg_option, x, rho_1, rho_2, lr=0.01):
    """
    Define optimization algorithm according to alg_option given by the user
    :param alg_option: Desired optimization method
    :param x: Image estimation
    :param rho_1: Desired horizontal distribution
    :param rho_2: Desired vertical distribution
    :param lr: Learning rate (relevant to gradient-descent)
    :return: Torch optimization function
    """
    if alg_option == 'GD':
        opt = torch.optim.SGD([x, rho_1, rho_2], lr)
    else:
        opt = torch.optim.LBFGS([x, rho_1, rho_2], line_search_fn='strong_wolfe')
    return opt


def objective(x, rho_1, rho_2, mu1, mu2, model_lambda, bccb_ind_mat):
    """
    Determine the objective function of the optimization process
    :param x: Current estimation of the image
    :param rho_1: Current estimation of the horizontal distribution
    :param rho_2: Current estimation of the vertical distribution
    :param mu1: first observed moment
    :param mu2: second observed moment
    :param model_lambda: Scales the penalty of first and second moments fidelity
    :param bccb_ind_mat: The indexes of the BCCB matrix
    :return: Loss value
    """
    norm_rho = general_utils.normalize_rho_vector(rho_1, rho_2)
    C_X = x[bccb_ind_mat]
    elem_1 = torch.matmul(C_X, norm_rho)
    part_1 = torch.sum(torch.square(elem_1 - mu1))
    elem_2 = torch.matmul(C_X, torch.transpose(torch.mul(norm_rho.repeat(C_X.shape[0], 1), C_X), 0, 1))
    part_2 = torch.sum(torch.square(elem_2 - mu2))
    obj_val = part_1 + model_lambda * part_2
    return obj_val


def execute_matching(device, img, mu1, mu2, bccb_ind_mat, x, rho_1, rho_2, lr, model_lambda, projected,
                     iterations_number, name_exp, iterations_per_round):
    """
    Perform matching between the observed moments and the images and distributions
    :param device: The torch process device
    :param img: Original image
    :param mu1: first observed moment
    :param mu2: second observed moment
    :param bccb_ind_mat: The indexes of the BCCB matrix
    :param x: Current estimation of the image
    :param rho_1: Current estimation of the horizontal distribution
    :param rho_2: Current estimation of the vertical distribution
    :param lr: Learning rate
    :param model_lambda: Scales the penalty of first and second moments fidelity
    :param projected: Boolean determine if running the projected version
    :param iterations_number: Max number of iterations
    :param name_exp: Name of output directory
    :param iterations_per_round: number of iterations between projections
    :return: None
    """
    H = rho_1.size()[0]
    W = rho_2.size()[1]
    opt = define_optimization_alg('BFGS', x, rho_1, rho_2, lr)
    loss, error, process_time = [], [], []
    t0 = time.time()

    def closure():
        opt.zero_grad()
        obj = objective(x, rho_1, rho_2, mu1, mu2, model_lambda, bccb_ind_mat)
        obj.backward()
        return obj

    if not projected:
        for i in range(iterations_number):
            opt.step(closure)
            obj = objective(x, rho_1, rho_2, mu1, mu2, model_lambda, bccb_ind_mat)
            print("Objective: %.10f" % obj.item())
            loss.extend([obj.item()])
            rolled_img, curr_error = general_utils.evaluate_data(device, x, H, W, img)
            error.extend([curr_error])
            process_time.extend([time.time()-t0])
            curr_data = {'loss': loss, 'error': error, 'curr_x': x.cpu().detach().numpy(),
                         'curr_rho_1': rho_1.cpu().detach().numpy(), 'curr_rho_2': rho_2.cpu().detach().numpy(),
                         'curr_shifted_image': rolled_img.cpu().detach().numpy()}

            np.save(name_exp + '/eval_vec.npy', {'loss': loss, 'error': error, 'process_time': process_time})
            np.save(name_exp + '/final_res.npy', curr_data)
            if len(curr_data['loss']) > 100 and len(set(curr_data['loss'][-100:])) == 1:
                break
    else:
        for i in range(int(iterations_number / iterations_per_round)):
            for j in range(iterations_per_round):
                opt.step(closure)
                obj = objective(x, rho_1, rho_2, mu1, mu2, model_lambda, bccb_ind_mat)
                print("Objective: %.10f" % obj.item())
                loss.extend([obj.item()])
                rolled_img, curr_error = general_utils.evaluate_data(device, x, H, W, img)
                error.extend([curr_error])
                process_time.extend([time.time() - t0])
                curr_data = {'loss': loss, 'error': error, 'curr_x': x.cpu().detach().numpy(),
                             'curr_rho_1': rho_1.cpu().detach().numpy(), 'curr_rho_2': rho_2.cpu().detach().numpy(),
                             'curr_shifted_image': rolled_img.cpu().detach().numpy(), 'process_time': process_time}

                np.save(name_exp + '/eval_vec.npy', {'loss': loss, 'error': error, 'process_time': process_time})
                np.save(name_exp + '/final_res.npy', curr_data)

            y = torch.transpose(torch.reshape(x, [H, W]), 0, 1)
            chosen_den = max(2 ** (-0.1 * i), 0.02)
            print('used den - ' + str(chosen_den))
            y_clean = bm3d(np.atleast_3d(y.cpu().detach().numpy()), chosen_den)

            x = Variable(torch.FloatTensor(np.transpose(y_clean).flatten()).to(device), requires_grad=True)
            opt = define_optimization_alg('BFGS', x, rho_1, rho_2, lr)

def run_MoM(flags, given_seed):
    """
    Execute MoM optimization
    :param flags: Execution parameters
    :param given_seed: Desired seed given to numpy and torch
    :return: None
    """
    np.random.seed(given_seed)
    torch.manual_seed(given_seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    device = general_utils.update_device(flags.gpu_num)
    os.mkdir(flags.name_exp)
    t0 = time.time()

    # load data
    Lh = flags.high_res

    img = Image.open(flags.image_path).convert('L')
    img = img.resize((Lh, Lh), Image.ANTIALIAS)
    img = np.array(img.getdata()).reshape(Lh, Lh) / 255

    # generate random rho vectors
    real_rho_1 = np.random.rand(Lh, 1)
    real_rho_2 = np.random.rand(1, Lh)
    real_rho_1 = real_rho_1 / np.sum(real_rho_1)
    real_rho_2 = real_rho_2 / np.sum(real_rho_2)
    np.save(flags.name_exp + '/real_vars.npy', {'real_rho_1': real_rho_1, 'real_rho_2': real_rho_2, 'img': img})
    img = torch.FloatTensor(img).to(device)

    # calculate moments
    mu1, mu2, bccb_ind_mat = general_utils.calculate_moments(device, flags.sigma, img, real_rho_1, real_rho_2,
                                                                 flags.high_res, flags.low_res, flags.N)
    t1 = time.time()
    np.save(flags.name_exp + '/moments.npy', {'mu1': mu1.detach().cpu().numpy(), 'mu2': mu2.detach().cpu().numpy(),
                                              'time': t1-t0})

    # initial vars
    initial_start = np.random.rand(Lh * Lh)
    initial_start_x = initial_start * (2 * np.mean(mu1.cpu().numpy()) / np.mean(initial_start))
    initial_start_rho_1 = 6 * np.random.rand(Lh, 1) - 3
    initial_start_rho_2 = 6 * np.random.rand(1, Lh) - 3
    x = Variable(torch.FloatTensor(initial_start_x).to(device), requires_grad=True)
    rho_1 = Variable(torch.FloatTensor(initial_start_rho_1).to(device), requires_grad=True)
    rho_2 = Variable(torch.FloatTensor(initial_start_rho_2).to(device), requires_grad=True)

    model_lambda = 1 / (x.size(0) * (1 + flags.sigma ** 2))
    execute_matching(device, img, mu1, mu2, bccb_ind_mat, x, rho_1, rho_2, flags.lr, model_lambda, flags.projected,
                     flags.iterations_number, flags.name_exp, flags.iterations_per_round)
    return



