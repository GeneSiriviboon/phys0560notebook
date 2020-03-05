import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
from scipy.ndimage.interpolation import shift

class Model(object):
    """
    Toy model describing the scattering process
    """
    def __init__(self, p_init, g1, g2, E, lamda ,  sigma1, sigma2 , V_acc ):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.lamda = lamda
        self.prob  = p_init
        self.g1 = g1
        self.g2 = g2
        self.V_acc = V_acc
        self.E = E
        self.dE = self.E[1] - self.E[0]
    
    def grad_matrix(self):
        # TODO: Fix acceleration term to be proper matrices
        acceleration_term = np.zeros([self.prob.shape[0], self.prob.shape[0]])
        acceleration_term[1:, :-1] += self.V_acc * np.eye(self.prob.shape[0] - 1)/2 /self.dE
        acceleration_term[:-1, 1:] +=  - self.V_acc * np.eye(self.prob.shape[0] - 1)/2 /self.dE
        acceleration_term[0,0] = self.V_acc/self.dE 
        acceleration_term[0,1] = - self.V_acc/self.dE
        acceleration_term[-1,-1] = - self.V_acc/self.dE
        acceleration_term[-1,-2] = self.V_acc/self.dE
        nocollision = - self.prob * (self.sigma1 + self.sigma2) /self.lamda * np.eye(self.prob.shape[0])
        elastic = self.sigma1/self.lamda * self.g1 
        inelastic = self.sigma2/self.lamda * self.g2 
        return acceleration_term + nocollision + elastic + inelastic

    def evolve(self):
 
        grads = self.grad_matrix()
        evolution_matrix = expm(grads)
        self.prob = evolution_matrix @ self.prob 
        self.prob[self.prob < 0] = 0
        return self.prob/ np.sum(self.prob)/self.dE


def collision (E, E_c, n, type = 'elastic'):
    Energy, Energy_prime = np.meshgrid(E, E)
    if type == 'inelastic':
        condition = np.logical_and(Energy >= E_c, Energy_prime <=  Energy - E_c)
        condition = np.logical_and(condition, Energy_prime >= 0)
    elif type == 'elastic':
        condition = np.logical_and(Energy_prime >= 0, Energy_prime <=  Energy)
    g_E_prime_E = np.eye(E.shape[0])
    g_E_prime_E[Energy >= E_c] = 0
    g_E_prime_E[condition] = (2 / np.pi * np.sin(np.pi * (Energy_prime[condition]) / (Energy[condition] - E_c)))
    # if np.max(E) > E_c:

    
    return g_E_prime_E

def counting(V_acc, V_retarded,  lamda, n, std, E_c, sigma1, sigma2, plot = False):
    E = np.linspace(- V_acc * 0.4, V_acc *1.3, num = n)
    p_init =  np.exp(- (E - 0.5)**2 / (2 * std **2))
    p_init = p_init/ np.sum(p_init)/(E[1] - E[0])
    g1 = collision(E, E_c = 0, n = n, type = 'elastic')

    g2 = collision(E, E_c = E_c,  n = n, type = 'inelastic')
    model = Model(p_init, g1 = g1, g2 = g2, E = E, lamda = lamda, V_acc = V_acc, sigma1 = sigma1, sigma2 = sigma2)
    prob  = model.evolve()
    counts = np.sum(prob[E > V_retarded]) * model.dE
    # counts = np.sum(prob * E)
    if plot:
        plt.imshow(g1)
        plt.show()
        plt.imshow(g2)
        plt.show()
        plt.plot(E, p_init)
        plt.plot(E, prob)
        plt.vlines(V_retarded, 0, np.max(prob), 'r', linestyle = '--')
        plt.title('V_acc = {:f}'.format(V_acc))
        plt.show()
    return E, p_init, prob[E > 0], counts

if __name__ == '__main__':   

    V_acc = np.linspace(1.0, 30, num = 200)
    # V_acc = np.linspace(10, 15, num = 200)
    V_retarded = 1.5
    counts = []
    prbs = []
    for i, V in enumerate(V_acc):
        prog = int((i+1)/len(V_acc) * 20)
        print('|', *['#'] * prog, *[' '] * (20 - prog), '|', ' {:d}%'.format(int(100 * (i+1)/len(V_acc))), sep = '', end='\r')

        # plot = (i+1)%40 == 0
        plot = False #if i != 1 else True
        E, p_init, prob, count = counting(V, V_retarded, std = 0.03, E_c = 1.3, n = 1000,lamda = 0.01,  sigma1 = 0.9, sigma2 = 1.2, plot = plot)
        counts.append(count)
        prbs.append(prob)
    
    plt.plot(V_acc, counts, '.')
    plt.xlabel('V_acc(V)')
    plt.ylabel('I (a.u.)')
    plt.show()
    plt.imshow(prbs)
    plt.show()

    # count = counting(V_acc = 2.0, V_retarded = 1.5, n = 1000, sigma1 = 2,  sigma2 = 2, plot = True)

