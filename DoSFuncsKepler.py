# -*- coding: utf-8 -*-
"""
Created on Tues June 20, 2017

@author: dg622@cornell.edu
"""
# IWA and OWA should use detection mode values
import numpy as np
import EXOSIMS.MissionSim as MissionSim
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.optimize as optimize
#import astropy.constants as const
import astropy.units as u
#import sympy
#from sympy.solvers import solve

from DoSFuncs import DoSFuncs

class DoSFuncsKepler(DoSFuncs):
    '''Calculates depth of search values for a given input EXOSIMS json script 
    or instantiated EXOSIMS.MissionSim object. 
    
    path or sim must be specified but not both
    
    NOTE: This is intended for use with KeplerLike1 or KeplerLike2 
    PlanetPopulation modules from EXOSIMS. The occurrence rates used to 
    convolve with the depth of search grid come from the Kepler distribution.
    
    Args:
        path (str):
            path to json script for EXOSIMS
        abins (int):
            number of semi-major axis bins for depth of search grid (optional)
        Rbins (int):
            number of planetary radius bins for depth of search grid (optional)
        maxTime (float):
            maximum total integration time in days (optional)
        intCutoff (float):
            integration cutoff time per target in days (optional)
        WA_targ (astropy Quantity):
            working angle for target instrument contrast (optional)
            
    Attributes:
        result (dict):
            dictionary containing results of the depth of search calculations
            Keys include:
                NumObs (dict):
                    dictionary containing number of observations for each 
                    stellar type, keys are: 'Mstars', 'Kstars', 'Gstars', 
                    'Fstars', and 'Entire'
                aedges (ndarray):
                    1D array of semi-major axis bin edges in AU
                Redges (ndarray):
                    1D array of planetary radius bin edges in R_earth
                DoS (dict):
                    dictionary containing 2D arrays of depth of search for
                    each stellar type, keys are: 'Mstars', 'Kstars', 'Gstars',
                    'Fstars', and 'Entire'
                occ_rates (dict):
                    dictionary containing 2D arrays of occurrence rates
                    extrapolated from Mulders 2015, keys are: 'Mstars', 'Kstars',
                    'Gstars', and 'Fstars'
                DoS_occ (dict):
                    dictionary containing 2D arrays of depth of search convolved
                    with the extrapolated occurrence rates, keys are: 'Mstars',
                    'Kstars', 'Gstars', 'Fstars', and 'Entire'
        sim (object):
            EXOSIMS.MissionSim object used to generate target list and 
            integration times
        outspec (dict):
            EXOSIMS.MissionSim output specification
    
    '''
    
    def __init__(self, path=None, abins=100, Rbins=30, maxTime=365.0, intCutoff=30.0, dMag=None, WA_targ=None):
        if path is None:
            raise ValueError('path must be specified')
        if path is not None:
            # generate EXOSIMS.MissionSim object to calculate integration times
            self.sim = MissionSim.MissionSim(scriptfile=path)
            print 'Acquired EXOSIMS data from %r' % (path)
        if dMag is not None:
            try:
                float(dMag)
            except TypeError:
                print 'dMag can have only one value'
        if WA_targ is not None:
            try:
                float(WA_targ.value)
            except AttributeError:
                print 'WA_targ must be astropy Quantity'
            except TypeError:
                print 'WA_targ can have only one value'
        self.result = {}
        # minimum and maximum values of semi-major axis and planetary radius
        # NO astropy Quantities
        amin = self.sim.PlanetPopulation.arange[0].to('AU').value
        amax = self.sim.PlanetPopulation.arange[1].to('AU').value
        Rmin = self.sim.PlanetPopulation.Rprange[0].to('earthRad').value
        Rmax = self.sim.PlanetPopulation.Rprange[1].to('earthRad').value
        assert Rmax > Rmin, 'Maximum planetary radius is less than minimum planetary radius'
        # need to get Cmin from contrast curve
        mode = filter(lambda mode: mode['detectionMode'] == True, self.sim.OpticalSystem.observingModes)[0]
        WA = np.linspace(mode['IWA'], mode['OWA'], 50)
        syst = mode['syst']
        lam = mode['lam']
        if dMag is None:
            # use dMagComp or dMagLim when dMag not specified
            dMag = self.sim.Completeness.dMagComp if self.sim.Completeness.dMagComp is not None else self.sim.OpticalSystem.dMagLim
        fZ = self.sim.ZodiacalLight.fZ0
        fEZ = self.sim.ZodiacalLight.fEZ0
        if WA_targ is None:
            core_contrast = syst['core_contrast'](lam,WA)
            contrast = interpolate.interp1d(WA.to('arcsec').value, core_contrast, \
                                    kind='cubic', fill_value=1.0)
            # find minimum value of contrast
            opt = optimize.minimize_scalar(contrast, \
                                       bounds=[mode['IWA'].to('arcsec').value, \
                                               mode['OWA'].to('arcsec').value],\
                                               method='bounded')
            Cmin = opt.fun
            WA_targ = opt.x*u.arcsec
        
        t_int1 = self.sim.OpticalSystem.calc_intTime(self.sim.TargetList,0,fZ,fEZ,dMag,WA_targ,mode)
        core_contrast = 10.0**(-0.4*self.sim.OpticalSystem.calc_dMag_per_intTime(t_int1,self.sim.TargetList,np.array([0]),fZ,fEZ,WA,mode))
        contrast = interpolate.interp1d(WA.to('arcsec').value,core_contrast,kind='cubic',fill_value=1.0)
        opt = optimize.minimize_scalar(contrast,bounds=[mode['IWA'].to('arcsec').value,mode['OWA'].to('arcsec').value],method='bounded')
        Cmin = opt.fun
        
        # find expected values of p and R
        if self.sim.PlanetPopulation.prange[0] != self.sim.PlanetPopulation.prange[1]:
            f = lambda p: p*self.sim.PlanetPopulation.dist_albedo(p)
            pexp, err = integrate.quad(f,self.sim.PlanetPopulation.prange[0],\
                                       self.sim.PlanetPopulation.prange[1],\
                                        epsabs=0,epsrel=1e-6,limit=100)
        else:
            pexp = self.sim.PlanetPopulation.prange[0]
            
        print 'Expected value of geometric albedo: %r' % (pexp)
        if self.sim.PlanetPopulation.Rprange[0] != self.sim.PlanetPopulation.Rprange[1]:
            f = lambda R: R*self.sim.PlanetPopulation.dist_radius(R)
            Rexp, err = integrate.quad(f,self.sim.PlanetPopulation.Rprange[0].to('earthRad').value,\
                                       self.sim.PlanetPopulation.Rprange[1].to('earthRad').value,\
                                        epsabs=0,epsrel=1e-4,limit=100)
            Rexp *= u.earthRad.to('AU')
        else:
            Rexp = self.sim.PlanetPopulation.Rprange[0].to('AU').value
        
        # minimum and maximum separations
        smin = (np.tan(mode['IWA'])*self.sim.TargetList.dist).to('AU').value
        smax = (np.tan(mode['OWA'])*self.sim.TargetList.dist).to('AU').value
        smax[smax>amax] = amax
    
        # include only stars where smin > amin
        bigger = np.where(smin>amin)[0]
        self.sim.TargetList.revise_lists(bigger)
        smin = smin[bigger]
        smax = smax[bigger]
    
        # include only stars where smin < amax
        smaller = np.where(smin<amax)[0]
        self.sim.TargetList.revise_lists(smaller)
        smin = smin[smaller]
        smax = smax[smaller]
        
        sInds = np.arange(self.sim.TargetList.nStars)        
        # calculate maximum integration time
        t_int = self.sim.OpticalSystem.calc_intTime(self.sim.TargetList, sInds, fZ, fEZ, dMag, WA_targ, mode)

        # remove integration times above cutoff
        cutoff = np.where(t_int.to('day').value<intCutoff)[0]
        self.sim.TargetList.revise_lists(cutoff)
        smin = smin[cutoff]
        smax = smax[cutoff]
        t_int = t_int[cutoff]

        print 'Beginning ck calculations'
        # calculate ck
        ck = self.find_ck(amin,amax,smin,smax,Cmin,pexp,Rexp)
        # offset to account for zero ck values with nonzero completeness
        ck += ck[ck>0.0].min()*1e-2
        print 'Finished ck calculations'

        print 'Beginning ortools calculations to determine list of observed stars'
        # use ortools to select observed stars
        sInds = self.select_obs(t_int.to('day').value,maxTime,ck)
        print 'Finished ortools calculations'

        # include only stars chosen for observation
        self.sim.TargetList.revise_lists(sInds)
        smin = smin[sInds]
        smax = smax[sInds]
        t_int = t_int[sInds]
        ck = ck[sInds]
        
        # get contrast array for given integration times
        sInds2 = np.arange(self.sim.TargetList.nStars)
        C_inst = 10.0**(-0.4*self.sim.OpticalSystem.calc_dMag_per_intTime(t_int,self.sim.TargetList,sInds2,fZ,fEZ,WA,mode))
        # store number of observed stars in result
        self.result['NumObs'] = {"all": self.sim.TargetList.nStars}
        print 'Number of observed targets: %r' % self.sim.TargetList.nStars
        # find bin edges for semi-major axis and planetary radius in AU
        aedges = np.logspace(np.log10(amin), np.log10(amax), abins+1)
        Redges = np.logspace(np.log10(Rmin*u.earthRad.to('AU')), \
                         np.log10(Rmax*u.earthRad.to('AU')), Rbins+1)
        # store aedges and Redges in result
        self.result['aedges'] = aedges
        self.result['Redges'] = Redges/u.earthRad.to('AU')
    
        aa, RR = np.meshgrid(aedges,Redges) # in AU
    
        # get depth of search 
        print 'Beginning depth of search calculations for observed stars'
        if self.sim.TargetList.nStars > 0:
            DoS = self.DoS_sum(aedges, aa, Redges, RR, pexp, smin, smax, \
                           self.sim.TargetList.dist.to('pc').value, C_inst, WA.to('arcsecond').value)
        else:
            DoS = np.zeros((aa.shape[0]-1,aa.shape[1]-1))
        print 'Finished depth of search calculations'
        # store DoS in result
        self.result['DoS'] = {"all": DoS}
        Redges /= u.earthRad.to('AU')
        # get occurrence data from KeplerLike
        occAll = self.sim.PlanetPopulation.Rvals
        R = self.sim.PlanetPopulation.Rs
        adist = self.sim.PlanetPopulation.dist_sma
        etas = np.zeros((len(Redges)-1,len(aedges)-1))
        # occurrence rate as function of R
        Rvals = np.zeros((len(Redges)-1,))
        for i in xrange(len(Redges)-1):
            for j in xrange(len(R)):
                if Redges[i] < R[j]:
                    break
            for k in xrange(len(R)):
                if Redges[i+1] < R[k]:
                    break
            if k-j == 0:
                Rvals[i] = (Redges[i+1]-Redges[i])/(R[j]-R[j-1])*occAll[j-1]
            elif k-j == 1:
                Rvals[i] = (R[j]-Redges[i])/(R[j]-R[j-1])*occAll[j-1]
                Rvals[i] += (Redges[i+1]-R[j])/(R[j+1]-R[j])*occAll[j]
            else:
                Rvals[i] = (R[j]-Redges[i])/(R[j]-R[j-1])*occAll[j-1]
                Rvals[i] += np.sum(occAll[j:k-1])
                Rvals[i] += (Redges[i+1]-R[k-1])/(R[k]-R[k-1])*occAll[k-1]
        
        # extrapolate to new grid
        for i in xrange(len(aedges)-1):
            fac2 = integrate.quad(adist, aedges[i], aedges[i+1])[0]
            etas[:,i] = Rvals*fac2

        self.result['occ_rates'] = {"all": etas}
        # perform convolution of depth of search with occurrence rates
        r_norm = Redges[1:] - Redges[:-1]
        a_norm = aedges[1:] - aedges[:-1]
        norma, normR = np.meshgrid(a_norm,r_norm)
        print 'Multiplying depth of search grid with occurrence rate grid'
        DoS_occ = DoS*etas*norma*normR
        self.result['DoS_occ'] = {"all": DoS_occ}
        
        # store MissionSim output specification dictionary
        self.outspec = self.sim.genOutSpec()
        print 'Calculations finished'