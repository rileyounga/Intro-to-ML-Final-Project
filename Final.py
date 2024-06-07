import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def Metropolis_Hastings(M, t, r, sigma, mu_l, sigma_l, N, M_2=None, r_2=None, cameras=1):
    """
    Perform Metropolis-Hastings sampling to sample from the posterior distribution 
    of the 3D line segment end-points.
    :param M: the first camera matrix
    :param t: relative offset ratios of points qs along the line segment between 
              the qi and qf endpoints
    :param r: observed points in the camera u,v image plane
    :param sigma: image projection noise covariance
    :param mu_l: priors over the 3D line segment endpoints
    :param sigma_l: priors over the 3D line segment endpoints
    :param N: number of iterations
    :param M_2: the second camera matrix
    :param r_2: observed points in the camera u,v image plane for the second camera
    :param cameras: number of cameras
    :return: pi_samples, pf_samples
    """
    # Initialize
    pi = np.random.multivariate_normal(mu_l, sigma_l)
    pf = np.random.multivariate_normal(mu_l, sigma_l)
    pi_samples = []
    pf_samples = []
    for _ in range(N):
        # Proposal
        pi_star = np.random.multivariate_normal(pi, sigma_l)
        pf_star = np.random.multivariate_normal(pf, sigma_l)
        # Acceptance ratio
        alpha = acceptance_ratio(M, pi_star, pf_star, pi, pf, t, r, sigma,
                                 mu_l, sigma_l, M_2, r_2, cameras)
        # Acceptance logic
        if alpha >= 1:
            pi = pi_star
            pf = pf_star
        else:
            if np.random.uniform(0, 1) < alpha:
                pi = pi_star
                pf = pf_star
        pi_samples.append(pi)
        pf_samples.append(pf)

    return pi_samples, pf_samples

def acceptance_ratio(M, pi_star, pf_star, pi, pf, t, r, Sigma, mu_l,
                     Sigma_l, M_2=None, r_2=None, cameras=1):
    """
    Compute the acceptance ratio for Metropolis-Hastings
    params inherited from Metropolis_Hastings +
    :param pi_star: proposed pi
    :param pf_star: proposed pf
    :return: acceptance ratio
    """
    proposed = log_likelihood_both(M, pi_star, pf_star, t, r, Sigma, M_2, r_2, cameras) \
        + log_prior(pi_star, pf_star, mu_l, Sigma_l)
    previous = log_likelihood_both(M, pi, pf, t, r, Sigma, M_2, r_2, cameras) \
        + log_prior(pi, pf, mu_l, Sigma_l)
    # convert back to probability
    return np.exp(proposed - previous)

def log_likelihood_both(M, pi, pf, t, r, Sigma, M_2=None, r_2=None, cameras=1):
    """
    Compute the log likelihood for Camera projections model
    params inherited from Metropolis_Hastings +
    :return: log likelihood
    """
    # Homogeneous coordinates
    pi = np.append(pi, 1)
    pf = np.append(pf, 1)
    # Projected points
    pi_cam1 = np.matmul(M, pi)
    pf_cam1 = np.matmul(M, pf)
    if cameras == 2:
        pi_cam2 = np.matmul(M_2, pi)
        pf_cam2 = np.matmul(M_2, pf)
    # 2D points
    qi_cam1 = 1 / pi_cam1[2] * np.array([pi_cam1[0], pi_cam1[1]])
    qf_cam1 = 1 / pf_cam1[2] * np.array([pf_cam1[0], pf_cam1[1]])
    if cameras == 2:
        qi_cam2 = 1 / pi_cam2[2] * np.array([pi_cam2[0], pi_cam2[1]])
        qf_cam2 = 1 / pf_cam2[2] * np.array([pf_cam2[0], pf_cam2[1]])

    log_likelihood = 0
    for i in range(len(t)):
        qs_cam1 = qi_cam1 + (qf_cam1 - qi_cam1) * t[i]
        log_likelihood += stats.multivariate_normal.logpdf(r[i], mean=qs_cam1, cov=Sigma)
        if cameras == 2:
            qs_cam2 = qi_cam2 + (qf_cam2 - qi_cam2) * t[i]
            log_likelihood += stats.multivariate_normal.logpdf(r_2[i],
                                                               mean=qs_cam2, cov=Sigma)
    return log_likelihood

def log_prior(pi, pf, mu_l, Sigma_l):
    """
    Compute the log prior for Camera projections model
    params inherited from Metropolis_Hastings +
    :return: log prior
    """
    log_prior = 0
    log_prior += stats.multivariate_normal.logpdf(pi, mean=mu_l, cov=Sigma_l)
    log_prior += stats.multivariate_normal.logpdf(pf, mean=mu_l, cov=Sigma_l)
    return log_prior

def task2(save=True):
    # Save and load samples since 50,000 iterations takes a while
    if 'pi_samples.npy' in os.listdir() and 'pf_samples.npy' in os.listdir():
        # load samples
        pi_samples = np.load('pi_samples.npy')
        pf_samples = np.load('pf_samples.npy')
    else:
        # Read data
        t = pd.read_csv('data/inputs.csv', header=None).values.flatten()
        r = pd.read_csv('data/points_2d_camera_1.csv', header=None).values
        # Parameters
        sigma = np.eye(2) * (0.05 ** 2)
        mu_l = np.array([0, 0, 6])
        sigma_l = np.eye(3) * 6
        M = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]])
        pi_samples, pf_samples, = Metropolis_Hastings(M, t, r, sigma, mu_l, sigma_l, 50000)
        # Save samples
        if save:
            # option to turn it off for github debugging
            np.save('pi_samples.npy', pi_samples)
            np.save('pf_samples.npy', pf_samples)

    # Plot accepted samples for pi
    plt.figure()
    plt.title('Accepted proposals for $p_i$')
    plt.xlabel('Samples')
    plt.ylabel('$p_i$')
    plt.plot(pi_samples)
    plt.legend(['$x$', '$y$', '$z$'])
    plt.savefig('figures/pi_samples.png')
    plt.show()
    # Plot accepted samples for pf
    plt.figure()
    plt.title('Accepted proposals for $p_f$')
    plt.xlabel('Samples')
    plt.ylabel('$p_f$')
    plt.plot(pf_samples)
    plt.legend(['$x$', '$y$', '$z$'])
    plt.savefig('figures/pf_samples.png')
    plt.show()

def task3(save=True):
    # Required data
    r = pd.read_csv('data/points_2d_camera_1.csv', header=None).values
    M = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]])

    if 'pi_samples.npy' in os.listdir() and 'pf_samples.npy' in os.listdir():
        # load samples
        pi_samples = np.load('pi_samples.npy')
        pf_samples = np.load('pf_samples.npy')
    else:
        # Data for Metropolis_Hastings, not directly required for MAP
        t = pd.read_csv('data/inputs.csv', header=None).values.flatten()
        # Parameters
        sigma = np.eye(2) * (0.05 ** 2)
        mu_l = np.array([0, 0, 6])
        sigma_l = np.eye(3) * 6
        pi_samples, pf_samples, = Metropolis_Hastings(M, t, r, sigma, mu_l, sigma_l, 50000)
        # Save samples
        if save:
            np.save('pi_samples.npy', pi_samples)
            np.save('pf_samples.npy', pf_samples)

    pi_MAP = np.mean(pi_samples, axis=0)
    pf_MAP = np.mean(pf_samples, axis=0)
    # Calculate projection
    pi_MAP_cam1 = np.matmul(M, np.append(pi_MAP, 1))
    pf_MAP_cam1 = np.matmul(M, np.append(pf_MAP, 1))
    qi_MAP = 1 / pi_MAP_cam1[2] * np.array([pi_MAP_cam1[0], pi_MAP_cam1[1]])
    qf_MAP = 1 / pf_MAP_cam1[2] * np.array([pf_MAP_cam1[0], pf_MAP_cam1[1]])
    # Plot MAP
    plt.figure()
    plt.title('MAP estimate of the line')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.plot([qi_MAP[0], qf_MAP[0]], [qi_MAP[1], qf_MAP[1]], 'r')
    plt.scatter(r[:, 0], r[:, 1])
    plt.legend(['MAP', 'Observations'])
    plt.savefig('figures/MAP_cam1.png')
    plt.show()
    # report MAP
    print("MAP estimate using camera 1:")
    print('pi_MAP = ', pi_MAP)
    print('pf_MAP = ', pf_MAP)

def task4(save=True):
    # Required data
    r_2 = pd.read_csv('data/points_2d_camera_2.csv', header=None).values
    M_2 = np.array([[0, 0, 1, -5],
                    [0, 1, 0, 0],
                    [-1, 0, 0, 5]])
    if 'pi_samples.npy' in os.listdir() and 'pf_samples.npy' in os.listdir():
        # load samples
        pi_samples = np.load('pi_samples.npy')
        pf_samples = np.load('pf_samples.npy')
    else:
        # Data for MH, not directly required for MAP
        t = pd.read_csv('data/inputs.csv', header=None).values.flatten()
        r = pd.read_csv('data/points_2d_camera_1.csv', header=None).values
        # Parameters
        sigma = np.eye(2) * (0.05 ** 2)
        mu_l = np.array([0, 0, 6])
        sigma_l = np.eye(3) * 6
        M = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]])
        pi_samples, pf_samples, = Metropolis_Hastings(M, t, r, sigma, mu_l, sigma_l, 50000)
        # Save samples
        if save:
            np.save('pi_samples.npy', pi_samples)
            np.save('pf_samples.npy', pf_samples)

    pi_MAP = np.mean(pi_samples, axis=0)
    pf_MAP = np.mean(pf_samples, axis=0)
    # Calculate projection
    pi_MAP_cam2 = np.matmul(M_2, np.append(pi_MAP, 1))
    pf_MAP_cam2 = np.matmul(M_2, np.append(pf_MAP, 1))
    qi_MAP = 1 / pi_MAP_cam2[2] * np.array([pi_MAP_cam2[0], pi_MAP_cam2[1]])
    qf_MAP = 1 / pf_MAP_cam2[2] * np.array([pf_MAP_cam2[0], pf_MAP_cam2[1]])
    # Plot MAP
    plt.figure()
    plt.title('MAP estimate of the line')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.plot([qi_MAP[0], qf_MAP[0]], [qi_MAP[1], qf_MAP[1]], 'r')
    plt.scatter(r_2[:, 0], r_2[:, 1])
    plt.legend(['MAP', 'Observations'])
    plt.savefig('figures/MAP_cam2.png')
    plt.show()

def task5(save=True):
    # Required data
    r = pd.read_csv('data/points_2d_camera_1.csv', header=None).values
    r_2 = pd.read_csv('data/points_2d_camera_2.csv', header=None).values
    M = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]])
    M_2 = np.array([[0, 0, 1, -5],
                    [0, 1, 0, 0],
                    [-1, 0, 0, 5]])
    # Metropolis-Hastings
    if 'pi_samples_both.npy' in os.listdir() \
        and 'pf_samples_both.npy' in os.listdir():
        # load samples
        pi_samples = np.load('pi_samples_both.npy')
        pf_samples = np.load('pf_samples_both.npy')
    else:
        t = pd.read_csv('data/inputs.csv', header=None).values.flatten()
        # Parameters
        sigma = np.eye(2) * (0.05 ** 2)
        mu_l = np.array([0, 0, 6])
        sigma_l = np.eye(3) * 6
        pi_samples, pf_samples, = Metropolis_Hastings(M, t, r, sigma, mu_l, sigma_l,
                                                      50000, M_2, r_2, 2)
        # Save samples
        if save:
            np.save('pi_samples_both.npy', pi_samples)
            np.save('pf_samples_both.npy', pf_samples)

    # Plot accepted samples for pi
    plt.figure()
    plt.title('Accepted proposals for $p_i$')
    plt.xlabel('Samples')
    plt.ylabel('$p_i$')
    plt.plot(pi_samples)
    plt.legend(['$x$', '$y$', '$z$'])
    plt.savefig('figures/pi_samples_both.png')
    plt.show()
    # Plot accepted samples for pf
    plt.figure()
    plt.title('Accepted proposals for $p_f$')
    plt.xlabel('Samples')
    plt.ylabel('$p_f$')
    plt.plot(pf_samples)
    plt.legend(['$x$', '$y$', '$z$'])
    plt.savefig('figures/pf_samples_both.png')
    plt.show()

    pi_MAP = np.mean(pi_samples, axis=0)
    pf_MAP = np.mean(pf_samples, axis=0)
    # Calculate projection
    pi_MAP_cam1 = np.matmul(M, np.append(pi_MAP, 1))
    pf_MAP_cam1 = np.matmul(M, np.append(pf_MAP, 1))
    qi_MAP = 1 / pi_MAP_cam1[2] * np.array([pi_MAP_cam1[0], pi_MAP_cam1[1]])
    qf_MAP = 1 / pf_MAP_cam1[2] * np.array([pf_MAP_cam1[0], pf_MAP_cam1[1]])
    # Plot MAP for camera 1
    plt.figure()
    plt.title('MAP estimate of the line')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.plot([qi_MAP[0], qf_MAP[0]], [qi_MAP[1], qf_MAP[1]], 'r')
    plt.scatter(r[:, 0], r[:, 1])
    plt.legend(['MAP', 'Observations'])
    plt.savefig('figures/MAP_both_cam1.png')
    plt.show()
    # Calculate projection
    pi_MAP_cam2 = np.matmul(M_2, np.append(pi_MAP, 1))
    pf_MAP_cam2 = np.matmul(M_2, np.append(pf_MAP, 1))
    qi_MAP = 1 / pi_MAP_cam2[2] * np.array([pi_MAP_cam2[0], pi_MAP_cam2[1]])
    qf_MAP = 1 / pf_MAP_cam2[2] * np.array([pf_MAP_cam2[0], pf_MAP_cam2[1]])
    # Plot MAP for camera 2
    plt.figure()
    plt.title('MAP estimate of the line')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.plot([qi_MAP[0], qf_MAP[0]], [qi_MAP[1], qf_MAP[1]], 'r')
    plt.scatter(r_2[:, 0], r_2[:, 1])
    plt.legend(['MAP', 'Observations'])
    plt.savefig('figures/MAP_both_cam2.png')
    plt.show()
    # report MAP
    print("MAP estimate using both cameras:")
    print('pi_MAP = ', pi_MAP)
    print('pf_MAP = ', pf_MAP)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, default=1)
    parser.add_argument('--save', type=bool, default=True)
    args = parser.parse_args()
    task = args.task
    save = args.save
    if task == 1:
        task2(save)
        task3(save)
        task4(save)
        task5(save)
    elif task == 2:
        task2(save)
    elif task == 3:
        task3(save)
    elif task == 4:
        task4(save)
    elif task == 5:
        task5(save)
    else:
        raise ValueError('Invalid task number')
