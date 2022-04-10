
import torch
import numpy as np
from scipy.linalg import circulant


def index_matrix_for_bccb(H, W):
    """
    Creates a matrix containing the indexes of the BCCB matrix (transforms an image to a matrix where its columns are
    all flatten shifted images)
    :param H: Original image height
    :param W: Original image width
    :return: The index matrix
    """
    ccc = np.reshape(1 + np.array(range(H * W)), [H, W])

    def my_func(a):
        return circulant(a)

    fff = np.apply_along_axis(my_func, 1, ccc)
    higher = circulant(1 + np.array(range(H)))
    all_small_ordered = fff[higher - 1, :, :]

    index_mat = np.concatenate(np.concatenate(all_small_ordered, axis=-2), axis=-1)
    bccb_ind_mat = index_mat - 1
    return torch.Tensor(bccb_ind_mat).type(torch.LongTensor)

def update_SR_model(high_res, low_res, bccb_ind_mat):
    """
    Updates the BCCB index matrix so its columns will represent sampled images
    :param high_res: Original image resolution
    :param low_res: Observations resolution
    :param bccb_ind_mat: The indexes of the BCCB matrix
    :return:
    """
    if low_res < high_res:
        base_mat = np.zeros((int(high_res / low_res), int(high_res / low_res)))
        base_mat[0, 0] = 1
        sample_mat = np.kron(np.ones((low_res, low_res)), base_mat)
        sample_C = np.repeat(np.expand_dims(sample_mat.flatten() == 1, axis=1), high_res ** 2, axis=1)
        bccb_ind_mat = torch.reshape(bccb_ind_mat.flatten()[sample_C.flatten()], (low_res ** 2, high_res ** 2))
    return bccb_ind_mat

def generate_observations(P_C, sigma, real_rho, N, device):
    """
    Generating observations according to all shifted observations, 2-D shift distribution and noise level
    :param P_C: A matrix containing all shifted (flatten) image options
    :param sigma: Noise level
    :param real_rho: Original 2-D shift distribution
    :param N: Number of observations
    :param device: The torch process device
    :return: A batch of observed data
    """
    ind_samples = list(torch.utils.data.WeightedRandomSampler(real_rho.squeeze(), N))
    trainLoader = torch.utils.data.DataLoader(dataset=torch.transpose(P_C, 0, 1), batch_size=N, sampler=ind_samples)
    curr_data = torch.transpose(list(trainLoader)[0], 0, 1)
    observed_data = curr_data + sigma * torch.randn(curr_data.shape).to(device)
    return observed_data

def calculate_M(batch, batch_id, sigma, prev_M1, prev_M2, device):
    """
    Adds batch into the moments calculation
    :param batch: Batch added to the current moments calculation
    :param batch_id: Number of batch added to the calculation
    :param sigma: Noise level
    :param prev_M1: Previous first moment
    :param prev_M2: Previous second moment
    :param device: The torch process device
    :return:
    Updated moments after consideration to the extra batch
    """
    eye_mat = (sigma ** 2) * torch.eye(batch.shape[0]).to(device)
    if batch_id == 0:
        M2 = torch.divide(torch.matmul(batch, torch.transpose(batch, 0, 1)), batch.shape[1]) - eye_mat
        M1 = torch.mean(batch, 1)
    else:
        all_prev_hist = (prev_M2 + eye_mat) * batch.shape[1] * batch_id
        curr_hist = all_prev_hist + torch.matmul(batch, torch.transpose(batch, 0, 1))
        M2 = torch.divide(curr_hist, (batch_id + 1) * batch.shape[1]) - eye_mat
        M1 = torch.divide(prev_M1 * batch_id + torch.mean(batch, 1), batch_id + 1)
    return M1, M2

def normalize_rho_vector(rho_1, rho_2, normalization='softmax'):
    """
    Normalizing distribution vectors and calculating their joist distribution
    :param rho_1: Horizontal shift distribution
    :param rho_2: Vertical shift distribution
    :param normalization: Method for normlization (softmax desired for projecting vectors which are not in [0,1])
    :return: The joint distribution
    """
    if normalization == 'softmax':
        norm_rho_1 = torch.exp(rho_1)
        norm_rho_1 = torch.divide(norm_rho_1, torch.sum(norm_rho_1))
        norm_rho_2 = torch.exp(rho_2)
        norm_rho_2 = torch.divide(norm_rho_2, torch.sum(norm_rho_2))
    elif normalization == 'uniform':
        norm_rho_1 = torch.divide(rho_1, torch.sum(rho_1))
        norm_rho_2 = torch.divide(rho_2, torch.sum(rho_2))
    else:
        assert False, "wrong rho normalization"
    distribution_curr_shifts = torch.matmul(norm_rho_1, norm_rho_2)
    tot_rho = torch.reshape(torch.transpose(distribution_curr_shifts, 0, 1),
                            [rho_1.shape[0] * rho_2.shape[1], 1]).squeeze()
    return tot_rho

def calculate_moments(device, sigma, img, real_rho_1, real_rho_2, high_res, low_res, N):
    """
    Calculates the first two moments of the observed data
    :param device: The torch process device
    :param sigma: Noise level
    :param img: Original image
    :param real_rho_1: Original horizontal shift distribution
    :param real_rho_2: Original vertical shift distribution
    :param high_res: Original image resolution
    :param low_res: Observations resolution
    :param N: Number of observations
    :return: The first two moments and
    """
    P_C, tot_rho, bccb_ind_mat = generate_clean_shifted_images(device, img, real_rho_1, real_rho_2, high_res, low_res)
    if N:
        prev_m1 = torch.zeros(low_res * low_res)
        prev_m2 = torch.zeros(low_res * low_res, low_res * low_res)
        num_it = 100
        NN = int(N / num_it)
        for it in range(num_it):
            print('working on batch number ' + str(it))
            batch_img = generate_observations(P_C, sigma, tot_rho, NN, device)
            mu1, mu2 = calculate_M(batch_img, it, sigma, prev_m1, prev_m2, device)
            prev_m1 = mu1
            prev_m2 = mu2
    else: # "perfect" moments
        mu1 = torch.matmul(P_C, tot_rho)
        mu2 = torch.matmul(P_C, torch.matmul(torch.diag(tot_rho), torch.transpose(P_C, 0, 1)))
    return mu1, mu2, bccb_ind_mat


def generate_clean_shifted_images(device, img, real_rho_1, real_rho_2, high_res, low_res):
    """
    Generates a matrix where its columns are all options of shifted (flatten) images
    :param device: The torch process device
    :param img: Original image
    :param real_rho_1: Original horizontal shift distribution
    :param real_rho_2: Original vertical shift distribution
    :param high_res: Original image resolution
    :param low_res: Observations resolution
    :return: The shifted images matrix, the joint distribution of both shifts, the index matrix creating BCCB matrix
    """
    bccb_ind_mat = index_matrix_for_bccb(real_rho_1.shape[0], real_rho_2.shape[1])
    bccb_ind_mat = update_SR_model(high_res, low_res, bccb_ind_mat)

    real_rho_1 = torch.FloatTensor(real_rho_1).to(device)
    real_rho_2 = torch.FloatTensor(real_rho_2).to(device)

    x = torch.reshape(torch.transpose(img, 0, 1), [real_rho_1.shape[0] * real_rho_2.shape[1], 1]).squeeze()
    P_C = x[bccb_ind_mat]
    tot_rho = normalize_rho_vector(real_rho_1, real_rho_2, 'uniform')
    return P_C, tot_rho, bccb_ind_mat

def evaluate_data(device, x_curr_est, H, W, img):
    """
    Calculate the error of the given estimation, considering the shift-invariance of the model
    :param device: The torch process device
    :param x_curr_est: Current estimation of x
    :param img: Original image
    :param H: Original image height
    :param W: Original image width
    :return: The aligned estimation and its error
    """
    curr_img = torch.transpose(torch.reshape(x_curr_est, [H, W]), 0, 1)
    fft_curr_img = torch.fft.fft2(curr_img.cpu()).to(device)
    fft_img = torch.fft.fft2(img.cpu()).to(device)
    CC = torch.abs(torch.fft.ifft2((torch.mul(fft_img, torch.conj(fft_curr_img))).cpu()).to(device))
    [row_shift, col_shift] = torch.where(CC == torch.max(CC))
    rolled_img = torch.roll(curr_img, shifts=(row_shift[0], col_shift[0]), dims=(0, 1))
    img_error = rolled_img - img
    curr_error = torch.norm(img_error) / torch.norm(img)
    print("Error: %.10f" % curr_error.item())
    return rolled_img, curr_error.item()

def update_device(gpu_num):
    """
    Define the torch process device
    :param gpu_num: Assignment to a specific GPU
    :return: The torch process device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_num))
        torch.cuda.empty_cache()
        print("Gpu #" + str(gpu_num))
    else:
        device = torch.device("cpu")
    return device