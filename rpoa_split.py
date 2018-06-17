###################################################################################################
#
#	Author: Rahul Chandan
#	Email: rchandan@ucsb.edu
#	File: rpoa_split.py
#
#	Description:
#	This program tests the tightness of the split generalized-smoothness inequality.
#
###################################################################################################

import numpy as np


# Distribution function
ff = lambda x: (x == 1)
# Welfare function
uu = lambda x: x

# Generate a two-dimensional array containing all permutations of \lambda and \mu within
# a predefined range, call this two-dimensional array 'lam_mu'

lam = np.arange(0,10,0.01)
mu = np.arange(-1,10,0.01)
lam_ = np.tile(lam, (np.size(mu),1))
mu_ = np.tile(mu, (np.size(lam),1))
lam_mu = np.transpose(np.stack((lam_,mu_.T)), (1,2,0))

filt = np.ones(lam_mu[:,:,0].shape)
k = 2

for j in np.arange(3):
    for l in np.arange(3):
        if (j + l > 2):
            continue
        lhs = ff(np.min([j+1,k]))*uu(np.min([j+1,k]))*l - uu(j)*ff(j)*j + uu(j)
        filt = np.logical_and(filt, (lhs >= uu(l)*lam_mu[:,:,0] - uu(j)*lam_mu[:,:,1]))

rpoa = lam_mu[:,:,0]*filt / (1 + lam_mu[:,:,1]*filt)
print(f'rpoa is {np.max(rpoa)}')
print(f'lambda and mu are {lam_mu[int(np.argmax(rpoa)/rpoa.shape[1]), np.argmax(rpoa)%rpoa.shape[1]]}')

