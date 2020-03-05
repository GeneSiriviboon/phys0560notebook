import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
def translation(size, dE):
    acceleration_term = np.zeros([size, size])
    acceleration_term[1:, :-1] = np.eye(size - 1)
    return acceleration_term
def decelerate(size, dE):
    acceleration_term = np.zeros([size, size])
    acceleration_term[:-1, 1:] =   np.eye(size - 1)
    return acceleration_term

def collision (E, E_c, n, type = 'elastic'):
    Energy, Energy_prime = np.meshgrid(E, E)
    if type == 'inelastic':
        condition = np.logical_and(Energy >= E_c, Energy_prime <=  Energy - E_c)
        condition = np.logical_and(condition, Energy_prime >= 0)
    elif type == 'elastic':
        condition = np.logical_and(Energy_prime >= 0, Energy_prime <=  Energy)
    g_E_prime_E = np.eye(E.shape[0])
    g_E_prime_E[Energy >= E_c] = 0
    g_E_prime_E[condition] = (2 / np.pi * np.sin(np.pi * (Energy_prime[condition]) / (Energy[condition] - E_c + 1e-3)))
    # if np.max(E) > E_c:

    
    return g_E_prime_E

def collision2 (E, E_c, type = 'elastic'):
    Energy, _ = np.meshgrid(E, E)
    g_E_prime_E = np.eye(E.shape[0])
    if type == 'inelastic':
        col = E > E_c
        idx = np.argmax(col)
        row = Energy >= E_c
        if np.any(col):
            g_E_prime_E[row] = 0
            g_E_prime_E[:np.sum(col), idx:] = np.eye(np.sum(col))
    return g_E_prime_E

def FranckHertz(V_acc, V_retarded, n = 100, E_c = 4.9, E_init = 1e-2, sigma = [1e-4, 5e-3, 1e-4, 1e-4]):

    E = np.linspace(-1e-3, max(V_acc, V_retarded) * 1.3 , n)
    p_init = np.zeros(len(E))
    E_init = 1e-2
    p_init[np.argmax(E > E_init)] = 1
    p = []
    p_i = p_init
    n1 = np.argmax(E > V_acc)
    n2 = np.argmax(E > V_retarded)
    
    translation_matrix = translation(len(E), E[1] - E[0])
    elastic = collision (E, 0, len(E), type = 'elastic')
    inelastic = collision (E, E_c, len(E), type = 'inelastic')

    sigma1 = sigma[0]
    sigma2 = sigma[1]
    transition = translation_matrix * (1 - sigma1 - sigma2) + elastic * sigma1 + inelastic * sigma2
    # plt.plot(p_i, 'r.')
    # plt.plot(transition @ p_i, 'b.' )
    # plt.show()
    for _ in range(n1):
        p.append(p_i[::-1])
        p_i = transition @ p_i  


    decelerate_matrix = decelerate(len(E), E[1] - E[0])
    sigma1_ = sigma[2]
    sigma2_ = sigma[3]
    transition2 = decelerate_matrix* (1 - sigma1_ - sigma2_) + elastic * sigma1_ + inelastic * sigma2_
    for _ in range(n2):
        p.append(p_i[::-1])
        p_i = transition2 @ p_i      
    
    p = np.array(p).T
    
    return E, p

def FranckHertz2(V_acc, V_retarded, n = 100, E_c = 4.9, E_init = 1e-2, sigma = [1e-4, 5e-3, 1e-4, 1e-4]):

    E = np.linspace(-1e-3, max(V_acc, V_retarded) * 1.3 , n)
    p_init = np.zeros(len(E))
    E_init = 1e-2
    p_init[np.argmax(E > E_init)] = 1
    p = []
    p_i = p_init
    n1 = np.argmax(E > V_acc)
    n2 = np.argmax(E > V_retarded)
    
    translation_matrix = translation(len(E), E[1] - E[0])
    elastic = collision2 (E, 0, type = 'elastic')
    inelastic = collision2 (E, E_c, type = 'inelastic')

    sigma1 = sigma[0]
    sigma2 = sigma[1]
    transition = translation_matrix * (1 - sigma1 - sigma2) + elastic * sigma1 + inelastic * sigma2
    # plt.plot(p_i, 'r.')
    # plt.plot(transition @ p_i, 'b.' )
    # plt.show()
    for _ in range(n1):
        p.append(p_i[::-1])
        p_i = transition @ p_i  


    decelerate_matrix = decelerate(len(E), E[1] - E[0])
    sigma1_ = sigma[2]
    sigma2_ = sigma[3]
    transition2 = decelerate_matrix* (1 - sigma1_ - sigma2_) + elastic * sigma1_ + inelastic * sigma2_
    for _ in range(n2):
        p.append(p_i[::-1])
        p_i = transition2 @ p_i      
    
    p = np.array(p).T
    
    return E, p



if __name__ == '__main__':

    # V_acc = np.linspace(0.1, 30, num = 200)
    # V_retarded = 3
    # I = []
    # for i, V in enumerate(V_acc):
    #     E, p = FranckHertz(V, V_retarded, n = 1000, sigma = [10e-5, 10e-3, 1e-5, 1e-5])
    #     if i%30 ==0:
    #         plt.imshow(p)
    #         plt.show()
    #     p_f = p[::-1, -1]
    #     p_f = p_f /np.sum(p_f)
    #     count  = np.sum(p_f[E > 1.0])
    #     I.append(count)
    # I = ndimage.gaussian_filter(I, sigma = 3)
    # plt.plot(V_acc, I)
    # plt.show()

    V_acc = np.linspace(0.1, 30, num = 200)
    V_retarded = 3
    I = []
    for i, V in enumerate(V_acc):
        E, p = FranckHertz2(V, V_retarded, n = 1000, sigma = [1e-3, 10e-3, 1e-5, 1e-5])
        # if i%30 ==0:
        #     plt.imshow(p)
        #     plt.show()
        p_f = p[::-1, -1]
        p_f = p_f /np.sum(p_f)
        count  = np.sum(p_f[E > 1.0])
        I.append(count)
    I = ndimage.gaussian_filter(I, sigma = 3)
    plt.plot(V_acc, I)
    plt.show()


