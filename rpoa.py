import numpy as np

def ff(x):
    return (x == 1)

def uu(x):
    return x

v = np.array([0,1,0])
utility = np.array([[[uu(1)*v[0]*ff(1),uu(1)*v[1]*ff(1)], [uu(1)*v[0]*ff(1),uu(1)*v[2]*ff(1)]],
                    [[uu(2)*v[1]*ff(2),uu(2)*v[1]*ff(2)], [uu(1)*v[1]*ff(1),uu(1)*v[2]*ff(1)]]])
welfare = np.array([[uu(1)*(v[0]+v[1]), uu(1)*(v[0]+v[2])],[uu(2)*v[1],uu(1)*(v[1]+v[2])]])

lam = np.arange(0,10,0.01)
mu = np.arange(-1,10,0.01)
lam_ = np.tile(lam, (np.size(mu),1))
mu_ = np.tile(mu, (np.size(lam),1))
lam_mu = np.transpose(np.stack((lam_,mu_.T)), (1,2,0))
filt = np.ones(lam_mu[:,:,0].shape)

for s1 in range(utility.shape[0]):
    for s2 in range(utility.shape[1]):
        for s1Star in range(utility.shape[0]):
            for s2Star in range(utility.shape[1]):
                lhs = utility[s1Star,s2,0]+utility[s1,s2Star,1]-np.sum(utility[s1,s2,:])+welfare[s1,s2]
                print(f'{lhs} >= {welfare[s1Star,s2Star]}L - {welfare[s1,s2]}M')
                filt = np.logical_and(filt, (lhs >= welfare[s1Star,s2Star]*lam_mu[:,:,0] - welfare[s1,s2]*lam_mu[:,:,1]))
                
rpoa = lam_mu[:,:,0]*filt / (1 + lam_mu[:,:,1]*filt)
print(f'rpoa is {np.max(rpoa)}')
print(f'lambda and mu are {lam_mu[int(np.argmax(rpoa)/rpoa.shape[1]),np.argmax(rpoa)%rpoa.shape[1]]}')
