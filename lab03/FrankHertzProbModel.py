import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from tqdm import tqdm

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

    E = np.linspace(-1e-3, V_acc * 1.3 , n)
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
    inelastic2 = collision2 (E, E_c * 1.4, type = 'inelastic')

    sigma1 = sigma[0] / n
    sigma2 = sigma[1] / n
    transition = translation_matrix * (1 - sigma1 - sigma2) + elastic * sigma1 + inelastic * sigma2 + inelastic2 * sigma2
    # plt.imshow(translation_matrix)
    # plt.colorbar()
    # plt.title('translation matrix')
    # plt.show()
    # plt.imshow(elastic)
    # # plt.colorbar()
    # plt.title('elastic collision')
    # plt.show()
    # plt.imshow(inelastic)
    # # plt.colorbar()
    # plt.title('inelastic collision')
    # plt.show()
    # plt.plot(p_i, 'r.')
    # plt.plot(transition @ p_i, 'b.' )
    # plt.show()
    for _ in range(n1):
        p.append(p_i[::-1])
        p_i = transition @ p_i  
        # p_i /= np.sum(p_i)


    decelerate_matrix = decelerate(len(E), E[1] - E[0])
    sigma1_ = sigma[2] /n
    sigma2_ = sigma[3] /n
    transition2 = decelerate_matrix* (1 - sigma1_ - sigma2_) + elastic * sigma1_ + inelastic * sigma2_
    for _ in range(n2):
        p.append(p_i[::-1])
        p_i = transition2 @ p_i  
        # p_i /= np.sum(p_i)
        if (p_i == np.zeros(len(p_i))).all():
            break    
    
    p = np.array(p).T
    
    return E, p



if __name__ == '__main__':


    V_acc = np.linspace(4, 30, num = 200)
    V_retarded = 3
    I = []
    E = []
    E_c = 4.8
    for i,V in tqdm(enumerate(V_acc)):

        E_, p = FranckHertz(V, V_retarded, n = 1000, E_c = E_c, sigma = [1, 20, 0.00, 0.00])
        
        if V % E_c < 1e-1 or i ==0:
            p = (p/ np.max(p, axis = 0))
            plt.imshow(p, vmax  = 1)
            plt.colorbar()
            n_tick = 300
            x_positions = np.arange(0, p.shape[1], n_tick) # pixel count at label position
            x = np.round(np.linspace(0, 1, p.shape[1]), 2)
            x_labels = x[::n_tick] # labels you want to see
            plt.xticks(x_positions, x_labels)
            y_positions = np.arange(0, p.shape[0], n_tick)
            y_labels = np.round(E_[::-n_tick], 2)
            plt.yticks(y_positions, y_labels)
            plt.xlabel('Position in the tube (a.u.)')
            plt.ylabel('Energy (eV)')
            plt.show()
        p_f = p[::-1, -1]
        p_f = p_f /np.sum(p_f)
        count  = np.sum(p_f[E_ > 0.0])
        I.append(count)
        E_mean = np.sum(p_f[E_ > 0.0] * E_[E_ > 0.0])/np.sum(p_f[E_ > 0.0])
        E.append(E_mean)

    E = np.array(E)
    I = ndimage.gaussian_filter(I * E**0.5, sigma = 3)
    plt.plot(V_acc, I, label = 'Retarded Potential: 3V')

    I = []
    E = []
    V_retarded = 1.5
    for i,V in tqdm(enumerate(V_acc)):

        E_, p = FranckHertz(V, V_retarded, n = 1000, E_c = E_c, sigma = [1, 20, 0.00, 0.00])
        
        # if V % E_c < 1e-1 or i ==0:
            # p = (p/ np.max(p, axis = 0))
            # plt.imshow(p, vmax  = 1)
            # plt.colorbar()
            # n_tick = 300
            # x_positions = np.arange(0, p.shape[1], n_tick) # pixel count at label position
            # x = np.round(np.linspace(0, 1, p.shape[1]), 2)
            # x_labels = x[::n_tick] # labels you want to see
            # plt.xticks(x_positions, x_labels)
            # y_positions = np.arange(0, p.shape[0], n_tick)
            # y_labels = np.round(E_[::-n_tick], 2)
            # plt.yticks(y_positions, y_labels)
            # plt.xlabel('Position in the tube (a.u.)')
            # plt.ylabel('Energy (eV)')
            # plt.show()
        p_f = p[::-1, -1]
        p_f = p_f /np.sum(p_f)
        count  = np.sum(p_f[E_ > 0.0])
        I.append(count)
        E_mean = np.sum(p_f[E_ > 0.0] * E_[E_ > 0.0])/np.sum(p_f[E_ > 0.0])
        E.append(E_mean)

    E = np.array(E)
    I = ndimage.gaussian_filter(I * E**0.5, sigma = 3)
    plt.plot(V_acc, I, label = 'Retarded Potential: 1.5V')

    plt.xlabel('V_acc (V)')
    plt.ylabel('I (a.u.)')
    plt.legend()
    plt.show()


