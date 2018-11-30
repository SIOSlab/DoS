import numpy as np
import scipy.interpolate as interpolate
import astropy.units as u
from EXOSIMS.StarCatalog.EXOCAT1 import EXOCAT1
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.optimize as optimize
try:
    import cPickle as pickle
except:
    import pickle
import os

"""
This script does not use the DoSFuncs object to calculate depth-of-search. Instead, it
calculates depth-of-search for a list of targets in the EXOCAT1 StarCatalog from EXOSIMS
using WFIRST parameters and contrast value calculated at the separation value. The target 
list comes from the 'targets.txt' file. Depth-of-search is calculated using a numerical version 
of depth-of-search from Garrett et al. 2017. This is done with a Monte Carlo method where 
orbital eccentricity is assumed to be Rayleigh distributed.

p*Phi comes from grids produced by STScI. These grids are evaluated at solar metallicity (0.0)
and averaged over cloud level using the frequencies from Mark. The resulting p*Phi is an 
interpolant for each of the listed dists (semi-major axis) with phase angle (beta) the 
independent variable (Riemann sum over wavelength has been performed to build the interpolant).

The pickled dictionary 'DoS.res' contains the results. Top level keys include:
    aedges: 1-D ndarray of semi-major axis bin edges
    Rpedges: 1-D ndarray of planetary radius bin edges
    DoS: dictionary of depth-of-search results where the keys are the target names from 'targets.txt'

Plots of depth-of-search for each target are saved in the Plots folder. 
"""
# number of MC samples
samps = int(2e5)  # gives standard deviation of ~1e-3 in completeness

# semi-major axis and planetary radius limits and number of bins
amin = 0.1  # AU
amax = 100.0  # AU
abins = 100
Rpmin = 1.0  # R_earth
Rpmax = 22.6  # R_earth
Rpbins = 30
# earth radius in units of AU
REinAU = (1.0*u.earthRad).to('AU').value

# wavelength and bandpass information
lam = 575  # nm
bp = 10  # bandpass in %
band = np.array([-1,1])*float(lam)/1000.*bp/200.0 + lam/1000.
[ws, wstep] = np.linspace(band[0], band[1], 100, retstep=True)
bw = float(np.diff(band))

# weights for averaging cloud data (cloud values of [0, 0.01, 0.03, 0.10, 0.30, 1.00, 3.00, 6.00])
cloud_weights = [0.099, 0.001, 0.005, 0.010, 0.025, 0.280, 0.300, 0.280]

# ====================================================================================
# find optical system parameters IWA, OWA, and contrast curve
# (source: B. Nemati, Mar 2018, 20180308 Nemati Updated CGI Planet Sensitivity and Yield.pptx)
WA = np.array([3.1, 3.6, 4.2, 5.0, 6.0, 7.0, 7.9])*0.0476663  # working angles in arcsec
C = np.array([1.5e-9, 1.2e-9, 9.3e-10, 6.4e-10, 6.1e-10, 5.7e-10, 5.7e-10])  # contrast
contrast = interpolate.InterpolatedUnivariateSpline(WA, C)
res = optimize.minimize_scalar(contrast, bounds=[WA[0],WA[-1]], method='bounded')
Cmin_abs = res.fun
as_to_rad = u.arcsec.to('rad')  # to convert arcseconds to radians


def eccanom(M, e):
    """Finds eccentric anomaly from mean anomaly and eccentricity

    This method uses algorithm 2 from Vallado to find the eccentric anomaly
    from mean anomaly and eccentricity.

    Args:
        M (float or ndarray):
            mean anomaly
        e (float or ndarray):
            eccentricity (eccentricity may be a scalar if M is given as
            an array, but otherwise must match the size of M.

    Returns:
        E (float or ndarray):
            eccentric anomaly

    """

    # make sure M and e are of the correct format.
    # if 1 value provided for e, array must match size of M
    M = np.array(M).astype(float)
    if not M.shape:
        M = np.array([M])
    e = np.array(e).astype(float)
    if not e.shape:
        e = np.array([e] * len(M))

    assert e.shape == M.shape, "Incompatible inputs."
    assert np.all((e >= 0) & (e < 1)), "e defined outside [0,1)"

    # initial values for E
    E = M / (1 - e)
    mask = e * E ** 2 > 6 * (1 - e)
    E[mask] = (6 * M[mask] / e[mask]) ** (1. / 3)

    # Newton-Raphson setup
    tolerance = np.finfo(float).eps * 4.01
    numIter = 0
    maxIter = 200
    err = 1.
    while err > tolerance and numIter < maxIter:
        E = E - (M - E + e * np.sin(E)) / (e * np.cos(E) - 1)
        err = np.max(abs(M - (E - e * np.sin(E))))
        numIter += 1

    if numIter == maxIter:
        raise Exception("eccanom failed to converge. Final error of %e" % err)

    return E


# ================================================================================================
# load photometric data
tmp = np.load('allphotdata_2015.npz')
allphotdata = tmp['allphotdata']
clouds = tmp['clouds']
wavelns = tmp['wavelns']
betas = tmp['betas']*np.pi/180.0
dists = tmp['dists']

# np.savez_compressed('allphotdata_2015v2', allphotdata=tmp['allphotdata'], clouds=tmp['clouds'], wavelns=tmp['wavelns'],
#                     betas=tmp['betas'], dists=tmp['dists'])
# interpolant to get nearest semi-major axis for later interpolants
distinterp = interpolate.interp1d(dists, dists, kind='nearest', bounds_error=False,
                                  fill_value=(dists.min(), dists.max()))

# =======================================================================================
# 2-D photometric interpolant for phase angle and wavelength
# averaged over clouds using cloud_weights
photinterps2 = {}
for j, d in enumerate(dists):
    tmpdata = allphotdata[0, j, 0, :, :]*cloud_weights[0]
    for k, cloud in enumerate(clouds):
        if k != 0:
            tmpdata += allphotdata[0, j, k, :, :]*cloud_weights[k]
    photinterps2[d] = interpolate.RectBivariateSpline(betas, wavelns, tmpdata)

# ===============================================================================
# 1-D p*Phi(beta) and inverse interpolants for each distance given
pphi = {}
pphinv = {}
beta = np.linspace(0.0, np.pi, 200)
for d in dists:
    tmpphi = (photinterps2[float(distinterp(d))](beta, ws).sum(1)*wstep/bw)
    pphi[d] = interpolate.InterpolatedUnivariateSpline(beta, tmpphi, k=1, ext=1)
    inds = np.argsort(tmpphi)
    pphinv[d] = interpolate.InterpolatedUnivariateSpline(tmpphi[inds], beta[inds], k=1, ext=1)


# ================================================================================
# presample phase angle, eccentricity, and eccentric anomaly
sig = 0.175/np.sqrt(np.pi/2.0)
b = np.arccos(1.0 - 2.0*np.random.uniform(0.0, 1.0, samps))
sinb = np.sin(b)
e = sig*np.sqrt(-2.0*np.log(1.0 - np.random.uniform(0.0, 1.0, samps)))
M = np.random.uniform(0.0, 2.0*np.pi, samps)
E = eccanom(M, e)
ecosE = 1.0 - e*np.cos(E)


def F(a, Rp, smin, smax, d):
    """
    Completeness given semi-major axis and planetary radius

    Args:
        a (float): semi-major axis (in AU)
        Rp (float): planetary radius (in AU)
        smin (float): minimum separation (in AU)
        smax (float): maximum separation (in AU)
        d (float): distance to star (in pc)

    Returns:
        comp (float): completeness value
    """
    if 2.0*a < smin or (Rp/0.01/a)**2*pphi[float(distinterp(a))](0.0) < Cmin_abs:
        comp = 0.0
    else:
        r = a*ecosE
        s = r*sinb
        FR = Rp**2/r**2*pphi[float(distinterp(a))](b)
        Cmins = contrast(s/d)

        # where smin < s < smax
        sgood = np.where((s > smin) & (s < smax))[0]
        FRgood = np.where(FR > Cmins)[0]
        allgood = np.intersect1d(sgood, FRgood)

        comp = float(len(allgood))/float(samps)

    return comp


# vectorize the F function since scipy.integrate functions are used
F_v = np.vectorize(F)


# calculate depth-of-search for each bin
def DoS_bins(a, Rp, smin, smax, d):
    """
    Calculates depth-of-search for each bin

    Args:
        a (ndarray): 2-D array of semi-major axis bin edges
        Rp (ndarray): 2-D array of planetary radius bin edges
        smin (float): minimum projected separation (IWA*d in AU)
        smax (float): maximum projected separation (OWA*d in AU)
        d (float): distance to star in pc

    Returns:
        f (ndarray): 2-D array of depth-of-search values in each bin
    """

    tmp = F_v(a, Rp, smin, smax, d)
    f = 0.25*(tmp[:-1, :-1] + tmp[1:, :-1] + tmp[:-1, 1:] + tmp[1:, 1:])

    return f


def plot_dos(aedges, Rpedges, DoS, name, path=None):
    """Plots depth of search as a filled contour plot with contour lines

    Args:
        aedges (ndarray):
            1-D array of semi-major axis bin edges
        Rpedges (ndarray):
            1-D array of planetary radius bin edges
        DoS (ndarray):
            2-D array of depth-of-search values
        name (str):
            string indicating what to put in title of figure
        path (str):
            desired path to save figure (png, optional)

    """

    acents = 0.5 * (aedges[1:] + aedges[:-1])
    a = np.hstack((aedges[0], acents, aedges[-1]))
    a = np.around(a, 4)
    Rcents = 0.5 * (Rpedges[1:] + Rpedges[:-1])
    R = np.hstack((Rpedges[0], Rcents, Rpedges[-1]))
    R = np.around(R, 4)
    # extrapolate to left-most boundary
    tmp = DoS[:, 0] + (a[0] - a[1]) * ((DoS[:, 1] - DoS[:, 0]) / (a[2] - a[1]))
    DoS = np.insert(DoS, 0, tmp, axis=1)
    # extrapolate to right-most boundary
    tmp = DoS[:, -1] + (a[-1] - a[-2]) * ((DoS[:, -1] - DoS[:, -2]) / (a[-2] - a[-3]))
    DoS = np.insert(DoS, -1, tmp, axis=1)
    # extrapolate to bottom-most boundary
    tmp = DoS[0, :] + (R[0] - R[1]) * ((DoS[1, :] - DoS[0, :]) / (R[2] - R[1]))
    DoS = np.insert(DoS, 0, tmp, axis=0)
    # extrapolate to upper-most boundary
    tmp = DoS[-1, :] + (R[-1] - R[-2]) * ((DoS[-1, :] - DoS[-2, :]) / (R[-2] - R[-3]))
    DoS = np.insert(DoS, -1, tmp, axis=0)
    DoS = np.ma.masked_where(DoS <= 0.0, DoS)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contourf(a, R, DoS, locator=ticker.LogLocator(), levels=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    cs2 = ax.contour(a, R, DoS, colors='k', levels=cs.levels)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('a (AU)')
    ax.set_ylabel('$R_p$ ($R_\oplus$)')
    ax.set_title('Depth of Search - ' + name)
    cbar = fig.colorbar(cs)
    ax.clabel(cs2, fmt=ticker.LogFormatterMathtext(), colors='k')
    if path is not None:
        fig.savefig(path, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


# =============================================================================
# set up depth-of-search calculations
aedges = np.logspace(np.log10(amin), np.log10(amax), abins+1)  # AU
Rpedges = np.logspace(np.log10(Rpmin), np.log10(Rpmax), Rpbins+1)*REinAU  # AU
aa, RR = np.meshgrid(aedges, Rpedges)  # all in AU

# =============================================================================
# get targets
cat = EXOCAT1()
with open('targets.txt', 'r') as f:
    targs = f.read().split('\n')

# =============================================================================
# set up an output dictionary to save results
out_dict = {}
out_dict['aedges'] = aedges
out_dict['Rpedges'] = Rpedges/REinAU
out_dict['DoS'] = {}
catName = cat.Name.tolist()

# =============================================================================
# do depth-of-search calculations for each star in target list
if not os.path.isdir('Plots'):
    os.mkdir('Plots')
for i in xrange(len(targs)):
    print('Depth-of-search for {} - {}/{} targets'.format(targs[i], i+1, len(targs)))
    # get target index in catalog
    sInd = np.where(targs[i] == cat.Name)[0]
    # minimum and maximum projected separation
    smin = (np.tan(WA[0]*as_to_rad)*cat.dist[sInd]).to('AU').value
    smax = (np.tan(WA[-1]*as_to_rad)*cat.dist[sInd]).to('AU').value
    d = cat.dist[sInd].to('pc').value
    # depth-of-search calculation
    dos = DoS_bins(aa, RR, smin, smax, d)
    # save a plot
    plot_dos(aedges, Rpedges/REinAU, dos, targs[i], 'Plots/'+targs[i]+'.png')
    # store result in output dictionary
    out_dict['DoS'][catName[int(sInd)]] = dos

# save depth-of-search results to disk
with open('DoS.res', 'wb') as f:
    pickle.dump(out_dict, f)
