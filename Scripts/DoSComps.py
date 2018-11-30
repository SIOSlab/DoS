import numpy as np
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import astropy.units as u
from EXOSIMS.StarCatalog.EXOCAT1 import EXOCAT1
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
try:
    import cPickle as pickle
except:
    import pickle
import os

"""
This script does not use the DoSFuncs object to calculate depth-of-search. Instead, it
calculates depth-of-search for a list of targets in the EXOCAT1 StarCatalog from EXOSIMS
using WFIRST parameters and contrast value calculated via weighted average of the PDF of
separation given semi-major axis. The target list comes from the 'targets.txt' file. 
Depth-of-search is calculated using a numerical version of depth-of-search from 
Garrett et al. 2017 where orbital eccentricity is assumed to be zero.

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

# semi-major axis and planetary radius limits and number of bins
amin = 0.1  # AU
amax = 100.0  # AU
abins = 100
Rpmin = 1.0  # R_earth
Rpmax = 22.6  # R_earth
Rpbins = 30

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
as_to_rad = u.arcsec.to('rad')  # to convert arcseconds to radians


def get_Cmin(a, d, smin, smax):
    """
    Finds contrast by weighted average over PDF of separation given semi-major axis
    does this by performing a transformation: t = sqrt(1-(s/a)**2)

    Args:
         a (float): semi-major axis in AU
         d (float): distance to star in pc
         smin (float): minimum separation in AU
         smax (float): maximum separation in AU
    Returns:
        Cmin (float): weighted average separation
    """
    if a < smin:
        Cmin = 1.0
    else:
        if a > smax:
            su = smax
        else:
            su = a
        # find expected value of contrast from contrast curve
        tup = np.sqrt(1.0 - (smin / a) ** 2)
        tlow = np.sqrt(1.0 - (su / a) ** 2)
        ft = lambda t, a=a, d=d: contrast(a * np.sqrt(1.0 - t ** 2) / d)
        val = integrate.quadrature(ft, tlow, tup, tol=1e-6, rtol=1e-6)[0]
        Cmin = val / (tup - tlow)

    return Cmin


# ================================================================================================
# load photometric data
tmp = np.load('allphotdata_2015.npz')
allphotdata = tmp['allphotdata']
clouds = tmp['clouds']
wavelns = tmp['wavelns']
betas = tmp['betas']*np.pi/180.0
dists = tmp['dists']
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
# earth radius in units of AU
REinAU = (1.0*u.earthRad).to('AU').value


# ================================================================================
# conditional PDF of flux ratio given semi-major axis and planetary radius
def f_FR_given_a_Rp(FR, a, Rp):
    """
    Conditional PDF of flux ratio given semi-major axis and planetary radius

    Args:
         FR (ndarray): flux ratio
         a (ndarray): semi-major axis (in AU)
         Rp (ndarray): planetary radius (in AU)

    Returns:
        f (ndarray): PDF value
    """

    first = 0.5*np.sin(pphinv[float(distinterp(a))](FR*a**2/Rp**2))
    second = np.abs(pphinv[float(distinterp(a))].derivative(1)(FR*a**2/Rp**2)*a**2/Rp**2)
    f = first*second

    return f


# probability density of flux ratio given semi-major axis and planetary radius
# marginalized over instrument constraints
def F(a, Rp, smin, smax, d):
    """
    Conditional probability of FR given a, Rp (integral of f_FR_given_a_Rp)

    Args:
        a (float): semi-major axis (in AU)
        Rp (float): planetary radius (in AU)
        smin (float): minimum projected separation (IWA*d in AU)
        smax (float): maximum projected separation (OWA*d in AU)
        Cmin (float): distance to star in pc

    Returns:
        f (float): conditional probability
    """
    # get contrast value
    Cmin = get_Cmin(a, d, smin, smax)

    if smin > a:
        f = 0.0
    elif smax > a:
        C1 = Rp**2/a**2*pphi[float(distinterp(a))](np.pi - np.arcsin(smin/a))
        C2 = Rp**2/a**2*pphi[float(distinterp(a))](np.arcsin(smin/a))
        if Cmin > C2:
            f = 0.0
        elif Cmin > C1:
            Cm = Cmin + 0.5*(C2-Cmin)
            f = integrate.quadrature(f_FR_given_a_Rp, Cmin, Cm, args=(a, Rp), tol=1e-6, rtol=1e-6, maxiter=1400)[0]
            f+= integrate.quadrature(f_FR_given_a_Rp, Cm, C2, args=(a, Rp), tol=1e-6, rtol=1e-6, maxiter=1400)[0]
        else:
            Cm = C1 + 0.5*(C2-C1)
            f = integrate.quadrature(f_FR_given_a_Rp, C1, Cm, args=(a, Rp), tol=5e-6, rtol=5e-6, maxiter=1500)[0]
            f += integrate.quadrature(f_FR_given_a_Rp, Cm, C2, args=(a, Rp), tol=5e-6, rtol=5e-6, maxiter=1500)[0]
    else:
        C1 = Rp ** 2 / a ** 2 * pphi[float(distinterp(a))](np.pi - np.arcsin(smin / a))
        C2 = Rp ** 2 / a ** 2 * pphi[float(distinterp(a))](np.arcsin(smin / a))
        C3 = Rp**2/a**2*pphi[float(distinterp(a))](np.pi - np.arcsin(smax/a))
        C4 = Rp**2/a**2*pphi[float(distinterp(a))](np.arcsin(smax/a))
        if Cmin > C2:
            f = 0.0
        elif Cmin > C4:
            f = integrate.quadrature(f_FR_given_a_Rp, Cmin, C2, args=(a, Rp), tol=1e-6, rtol=1e-6, maxiter=400)[0]
        elif Cmin > C3:
            f = integrate.quadrature(f_FR_given_a_Rp, C4, C2, args=(a, Rp), tol=1e-6, rtol=1e-6, maxiter=500)[0]
        elif Cmin > C1:
            f = integrate.quadrature(f_FR_given_a_Rp, C4, C2, args=(a, Rp), tol=1e-6, rtol=1e-6, maxiter=500)[0]
            f += integrate.quadrature(f_FR_given_a_Rp, Cmin, C3, args=(a, Rp), tol=1e-6, rtol=1e-6, maxiter=1100)[0]
        else:
            f = integrate.quadrature(f_FR_given_a_Rp, C4, C2, args=(a, Rp), tol=1e-6, rtol=1e-6, maxiter=500)[0]
            f += integrate.quadrature(f_FR_given_a_Rp, C1, C3, args=(a, Rp), tol=1e-6, rtol=1e-6, maxiter=1000)[0]
    # completeness must be <= 1
    if f > 1:
        f = 1.0

    return f


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
