# -*- coding: utf-8 -*-
"""
Created on Wed Feb 1, 2017
Updated Thurs May 4, 2017

@author: dg622@cornell.edu
"""

import numpy as np
import os, copy
import EXOSIMS.MissionSim as MissionSim
import sympy
from sympy.solvers import solve
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import astropy.constants as const
import astropy.units as u
try:
    import cPickle as pickle
except:
    import pickle
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class DoSFuncs(object):
    '''Calculates depth of search values for a given input EXOSIMS json script 
    or instantiated EXOSIMS.MissionSim object. Only stellar types M, K, G, and 
    F are used. All other stellar types are filtered out. Occurrence rates are 
    extrapolated from data in Mulders 2015.
    
    'core_contrast' must be specified in the input json script as either a 
    path to a fits file or a constant value, otherwise the default contrast 
    value from EXOSIMS will be used
    
    path or sim must be specified but not both
    
    Args:
        path (str):
            path to json script for EXOSIMS
        sim (object):
            existing EXOSIMS.MissionSim object 
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
    
    def __init__(self, path=None, sim=None, abins=100, Rbins=30, maxTime=365.0, intCutoff=30.0, WA_targ=None):
        if path is None and sim is None:
            raise ValueError('path or sim must be specified')
        if path is not None and sim is not None:
            raise ValueError('specify path or sim, not both')
        if path is not None and sim is None:
            # generate EXOSIMS.MissionSim object to calculate integration times
            self.sim = MissionSim.MissionSim(scriptfile=path)
            print 'Acquired EXOSIMS data from %r' % (path)
        if path is None and sim is not None:
            # EXOSIMS.MissionSim object has been pre-initialized
            # make deepcopy so that original sim is not overwritten
            self.sim = copy.deepcopy(sim)
            print 'Acquired existing EXOSIMS.MissionSim object'
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
        Rmin = (self.sim.PlanetPopulation.Rprange[0]/const.R_earth).decompose().value
        assert Rmin < 45.0, 'Minimum planetary radius is above extrapolation range'
        if Rmin < 0.35:
            print 'Rmin reset to 0.35*R_earth'
            Rmin = 0.35
        Rmax = (self.sim.PlanetPopulation.Rprange[1]/const.R_earth).decompose().value
        assert Rmax > 0.35, 'Maximum planetary radius is below extrapolation range'
        if Rmax > 45.0:
            print 'Rmax reset to 45.0*R_earth'
        assert Rmax > Rmin, 'Maximum planetary radius is less than minimum planetary radius'
        # need to get Cmin from contrast curve
        WA = np.linspace(self.sim.OpticalSystem.IWA, self.sim.OpticalSystem.OWA, 50)
        mode = self.sim.OpticalSystem.observingModes[0]
        syst = mode['syst']
        lam = mode['lam']
        dMag = self.sim.OpticalSystem.dMagLim
        fZ = self.sim.ZodiacalLight.fZ0
        fEZ = self.sim.ZodiacalLight.fEZ0
        if WA_targ == None:
            core_contrast = syst['core_contrast'](lam,WA)
            contrast = interpolate.interp1d(WA.to('arcsec').value, core_contrast, \
                                    kind='cubic', fill_value=1.0)
            # find minimum value of contrast
            opt = optimize.minimize_scalar(contrast, \
                                       bounds=[self.sim.OpticalSystem.IWA.to('arcsec').value, \
                                               self.sim.OpticalSystem.OWA.to('arcsec').value],\
                                               method='bounded')
            Cmin = opt.fun
            WA_targ = opt.x*u.arcsec
        
        t_int1 = self.sim.OpticalSystem.calc_intTime(self.sim.TargetList,np.array([0]),fZ,fEZ,dMag,WA_targ,mode)
        core_contrast = self.sim.OpticalSystem.calc_contrast_per_intTime(t_int1,self.sim.TargetList,np.array([0]),fZ,fEZ,WA,mode,dMag=dMag)
        contrast = interpolate.interp1d(WA.to('arcsec').value,core_contrast,kind='cubic',fill_value=1.0)
        opt = optimize.minimize_scalar(contrast,bounds=[self.sim.OpticalSystem.IWA.to('arcsec').value,self.sim.OpticalSystem.OWA.to('arcsec').value],method='bounded')
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
            Rexp, err = integrate.quad(f,self.sim.PlanetPopulation.Rprange[0].to('km').value,\
                                       self.sim.PlanetPopulation.Rprange[1].to('km').value,\
                                        epsabs=0,epsrel=1e-4,limit=100)
            Rexp *= self.sim.PlanetPopulation.Rprange.unit.to('AU')
        else:
            Rexp = self.sim.PlanetPopulation.Rprange[0].to('AU').value
        
        # include only F G K M stars
        spec = np.array(map(str, self.sim.TargetList.Spec))
        iF = np.where(np.core.defchararray.startswith(spec, 'F'))[0]
        iG = np.where(np.core.defchararray.startswith(spec, 'G'))[0]
        iK = np.where(np.core.defchararray.startswith(spec, 'K'))[0]
        iM = np.where(np.core.defchararray.startswith(spec, 'M'))[0]
        i = np.append(np.append(iF, iG), iK)
        i = np.append(i,iM)
        i = np.unique(i)
        self.sim.TargetList.revise_lists(i)
        print 'Filtered target stars to only include M, K, G, and F type'
        # minimum and maximum separations
        smin = (np.tan(self.sim.OpticalSystem.IWA)*self.sim.TargetList.dist).to('AU').value
        smax = (np.tan(self.sim.OpticalSystem.OWA)*self.sim.TargetList.dist).to('AU').value
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
        
        # calculate integration times
        sInds = np.arange(self.sim.TargetList.nStars)
        # select detection mode
        mode = filter(lambda mode: mode['detectionMode'] == True, self.sim.OpticalSystem.observingModes)[0]
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
        WA = np.linspace(self.sim.OpticalSystem.IWA, self.sim.OpticalSystem.OWA, 50)
        sInds2 = np.arange(self.sim.TargetList.nStars)
        C_inst = self.sim.OpticalSystem.calc_contrast_per_intTime(t_int,self.sim.TargetList,sInds2,fZ,fEZ,WA,mode)
    
        # find which are M K G F stars
        spec = np.array(map(str, self.sim.TargetList.Spec))
        Mlist = np.where(np.core.defchararray.startswith(spec, 'M'))[0]
        Klist = np.where(np.core.defchararray.startswith(spec, 'K'))[0]
        Glist = np.where(np.core.defchararray.startswith(spec, 'G'))[0]
        Flist = np.where(np.core.defchararray.startswith(spec, 'F'))[0]
        print '%r M stars observed' % (len(Mlist))
        print '%r K stars observed' % (len(Klist))
        print '%r G stars observed' % (len(Glist))
        print '%r F stars observed' % (len(Flist))
        print '%r total stars observed' % (len(Mlist)+len(Klist)+len(Glist)+len(Flist))
        NumObs = {'Mstars':len(Mlist), 'Kstars':len(Klist), 'Gstars':len(Glist),\
              'Fstars':len(Flist), 'Entire':(len(Mlist)+len(Klist)+len(Glist)\
                           +len(Flist))}
        # store number of observed stars in result
        self.result['NumObs'] = NumObs
        # find bin edges for semi-major axis and planetary radius in AU
        aedges = np.logspace(np.log10(amin), np.log10(amax), abins+1)
        Redges = np.logspace(np.log10(Rmin*const.R_earth.to('AU').value), \
                         np.log10(Rmax*const.R_earth.to('AU').value), Rbins+1)
        # store aedges and Redges in result
        self.result['aedges'] = aedges
        self.result['Redges'] = Redges/const.R_earth.to('AU').value
    
        aa, RR = np.meshgrid(aedges,Redges) # in AU
    
        # get depth of search for each stellar type
        DoS = {}
        print 'Beginning depth of search calculations for observed M stars'
        if len(Mlist) > 0:
            DoS['Mstars'] = self.DoS_sum(aedges, aa, Redges, RR, pexp, smin[Mlist], \
               smax[Mlist], self.sim.TargetList.dist[Mlist].to('pc').value, C_inst[Mlist,:], WA)
        else:
            DoS['Mstars'] = np.zeros((aa.shape[0]-1,aa.shape[1]-1))
        print 'Finished depth of search calculations for observed M stars'
        print 'Beginning depth of search calculations for observed K stars'
        if len(Klist) > 0:
            DoS['Kstars'] = self.DoS_sum(aedges, aa, Redges, RR, pexp, smin[Klist], \
               smax[Klist], self.sim.TargetList.dist[Klist].to('pc').value, C_inst[Klist,:], WA)
        else:
            DoS['Kstars'] = np.zeros((aa.shape[0]-1,aa.shape[1]-1))
        print 'Finished depth of search calculations for observed K stars'
        print 'Beginning depth of search calculations for observed G stars'
        if len(Glist) > 0:
            DoS['Gstars'] = self.DoS_sum(aedges, aa, Redges, RR, pexp, smin[Glist], \
               smax[Glist], self.sim.TargetList.dist[Glist].to('pc').value, C_inst[Glist,:], WA)
        else:
            DoS['Gstars'] = np.zeros((aa.shape[0]-1,aa.shape[1]-1))
        print 'Finished depth of search calculations for observed G stars'
        print 'Beginning depth of search calculations for observed F stars'
        if len(Flist) > 0:
            DoS['Fstars'] = self.DoS_sum(aedges, aa, Redges, RR, pexp, smin[Flist], \
               smax[Flist], self.sim.TargetList.dist[Flist].to('pc').value, C_inst[Flist,:], WA)
        else:
            DoS['Fstars'] = np.zeros((aa.shape[0]-1,aa.shape[1]-1))
        print 'Finished depth of search calculations for observed F stars'
        DoS['Entire'] = DoS['Mstars'] + DoS['Kstars'] + DoS['Gstars'] + DoS['Fstars']
        # store DoS in result
        self.result['DoS'] = DoS
    
        # load occurrence data from file
        print 'Loading occurrence data'
        directory = os.path.dirname(os.path.abspath(__file__))
        rates = pickle.load(open(directory+'/Mulders.ocr','rb'))
    
        # values from Mulders
        Redges /= const.R_earth.to('AU').value     
        Periods = rates['PeriodEdges']*u.day
        Radii = rates['RpEdges']
        dP = np.log10(Periods[1:]/Periods[:-1]).decompose().value
        dR = np.log10(Radii[1:]/Radii[:-1])
        ddP, ddR = np.meshgrid(dP, dR)
    
        # extrapolate occurrence values to new grid
        occ_rates = {}
        print 'Extrapolating occurrence rates for M stars'
        occ_rates['Mstars'] = self.find_occurrence(0.35*const.M_sun,ddP,ddR,Radii,\
                 Periods,rates['MstarsMean'],aedges,Redges,\
                              self.sim.PlanetPopulation.dist_sma,amin)
        print 'Extrapolating occurrence rates for K stars'
        occ_rates['Kstars'] = self.find_occurrence(0.70*const.M_sun,ddP,ddR,Radii,\
                 Periods,rates['KstarsMean'],aedges,Redges,\
                              self.sim.PlanetPopulation.dist_sma,amin)
        print 'Extrapolating occurrence rates for G stars'
        occ_rates['Gstars'] = self.find_occurrence(0.91*const.M_sun,ddP,ddR,Radii,\
                 Periods,rates['GstarsMean'],aedges,Redges,\
                              self.sim.PlanetPopulation.dist_sma,amin)
        print 'Extrapolating occurrence rates for F stars'
        occ_rates['Fstars'] = self.find_occurrence(1.08*const.M_sun,ddP,ddR,Radii,\
                 Periods,rates['FstarsMean'],aedges,Redges,\
                              self.sim.PlanetPopulation.dist_sma,amin)
        self.result['occ_rates'] = occ_rates
          
        # convolve depth of search with occurrence rates
        DoS_occ = {}
        print 'Convolving depth of search with occurrence rates'
        DoS_occ['Mstars'] = DoS['Mstars']*occ_rates['Mstars']
        DoS_occ['Kstars'] = DoS['Kstars']*occ_rates['Kstars']
        DoS_occ['Gstars'] = DoS['Gstars']*occ_rates['Gstars']
        DoS_occ['Fstars'] = DoS['Fstars']*occ_rates['Fstars']
        DoS_occ['Entire'] = DoS_occ['Mstars']+DoS_occ['Kstars']+DoS_occ['Gstars']+DoS_occ['Fstars']
        self.result['DoS_occ'] = DoS_occ
        
        # store MissionSim output specification dictionary
        self.outspec = self.sim.genOutSpec()
        self.ck = ck
    
    def one_DoS_grid(self,a,R,p,smin,smax,Cmin):
        '''Calculates completeness for one star on constant semi-major axis--
        planetary radius grid
    
        Args:
            a (ndarray):
                2D array of semi-major axis values in AU
            R (ndarray):
                2D array of planetary radius values in AU
            p (float):
                average geometric albedo value
            smin (float):
                minimum separation in AU
            smax (float):
                maximum separation in AU
            Cmin (ndarray):
                2D array of minimum contrast
    
        Returns:
            f (ndarray):
                2D array of depth of search values for one star on 2D grid
    
        '''
        
        a = np.array(a, ndmin=1, copy=False)
        R = np.array(R, ndmin=1, copy=False)
        Cmin = np.array(Cmin, ndmin=1, copy=False)

        f = np.zeros(a.shape)
        # work on smax < a first
        fg = f[smax<a]
        ag = a[smax<a]
        Rg = R[smax<a]
        Cgmin = Cmin[smax<a]

        b1g = np.arcsin(smin/ag)
        b2g = np.pi-np.arcsin(smin/ag)
        b3g = np.arcsin(smax/ag)
        b4g = np.pi-np.arcsin(smax/ag)
        
        C1g = (p*(Rg/ag)**2*np.cos(b1g/2.0)**4)
        C2g = (p*(Rg/ag)**2*np.cos(b2g/2.0)**4)
        C3g = (p*(Rg/ag)**2*np.cos(b3g/2.0)**4)
        C4g = (p*(Rg/ag)**2*np.cos(b4g/2.0)**4)
        
        C2g[C2g<Cgmin] = Cgmin[C2g<Cgmin]
        C3g[C3g<Cgmin] = Cgmin[C3g<Cgmin]
        
        vals = C3g > C1g
        C3g[vals] = 0.0
        C1g[vals] = 0.0
        vals = C2g > C4g
        C2g[vals] = 0.0
        C4g[vals] = 0.0
        
        fg = (ag/np.sqrt(p*Rg**2)*(np.sqrt(C4g)-np.sqrt(C2g)+np.sqrt(C1g)-np.sqrt(C3g)))
        
        fl = f[smax>=a]
        al = a[smax>=a]
        Rl = R[smax>=a]
        Clmin = Cmin[smax>=a]
    
        b1l = np.arcsin(smin/al)
        b2l = np.pi-np.arcsin(smin/al)

        C1l = np.nan_to_num((p*(Rl/al)**2*np.cos(b1l/2.0)**4))
        C2l = np.nan_to_num((p*(Rl/al)**2*np.cos(b2l/2.0)**4))

        C2l[C2l<Clmin] = Clmin[C2l<Clmin]
        vals = C2l > C1l

        C1l[vals] = 0.0
        C2l[vals] = 0.0

        fl = (al/np.sqrt(p*Rl**2)*(np.sqrt(C1l)-np.sqrt(C2l)))

        f[smax<a] = fg
        f[smax>=a] = fl
        f[smin>a] = 0.0

        return f
    
    def one_DoS_bins(self,a,R,p,smin,smax,Cmin):
        '''Calculates depth of search for each bin by integrating the
        completeness for given semi-major axis and planetary radius
        
        Args:
            a (ndarray):
                2D grid of semi-major axis bin edges in AU
            R (ndarray):
                2D grid of planetary radius bin edges in R_Earth
            p (float):
                expected value of geometric albedo
            smin (float):
                minimum separation in AU
            smax (float):
                maximum separation in AU
            Cmin (ndarray):
                2D grid of minimum contrast
        
        Returns:
            f (ndarray):
                2D array of depth of search values in each bin
        
        '''
        
        tmp = self.one_DoS_grid(a,R,p,smin,smax,Cmin)
        f = 0.25*(a[1:,1:]-a[1:,:-1])*(R[1:,1:]-R[:-1,1:])/const.R_earth.to('AU').value*(tmp[:-1,:-1]+tmp[1:,:-1]+tmp[:-1,1:]+tmp[1:,1:])
        
        return f

    def DoS_sum(self,a,aa,R,RR,pexp,smin,smax,dist,C_inst,WA):
        '''Sums the depth of search
        
        Args:
            a (ndarray):
                1D array of semi-major axis bin edge values in AU
            aa (ndarray):
                2D grid of semi-major axis bin edge values in AU
            R (ndarray):
                1D array of planetary radius bin edge values in AU
            RR (ndarray):
                2D grid of planetary radius bin edge values in AU
            pexp (float):
                expected value of geometric albedo
            smin (ndarray):
                1D array of minimum separation values in AU
            smax (ndarray):
                1D array of maximum separation values in AU
            dist (ndarray):
                1D array of stellar distance values in pc
            C_inst (ndarray):
                instrument contrast at working angle
            WA (ndarray):
                working angles in arcseconds
            
        Returns:
            DoS (ndarray):
                2D array of depth of search values summed for input stellar list
        
        '''
        
        DoS = np.zeros((aa.shape[0]-1,aa.shape[1]-1))
        for i in xrange(len(smin)):
            Cs = interpolate.interp1d(WA, C_inst[i], kind='cubic', fill_value=1.0)
            Cmin = np.zeros(a.shape)
            # expected value of Cmin calculations for each separation
            for j in xrange(len(a)):
                if a[j] < smin[i]:
                    Cmin[j] = 1.0
                else:
                    f = lambda s: Cs(s/dist[i])*s/(a[j]**2*np.sqrt(1.0-(s/a[j])**2))
                    if a[j] > smax[i]:
                        su = smax[i]
                    else:
                        su = a[j]
                    # find expected value of minimum contrast from contrast curve
                    val,err = integrate.quad(f,smin[i],su,epsabs=0,epsrel=1e-3,limit=100)
                    den = np.sqrt(1.0-(smin[i]/a[j])**2) - np.sqrt(1.0-(su/a[j])**2)
                    Cmin[j] = val/den

            Cmin,RR = np.meshgrid(Cmin,R)
            DoS += self.one_DoS_bins(aa,RR,pexp,smin[i],smax[i],Cmin)
        
        return DoS

    def find_ck(self,amin,amax,smin,smax,Cmin,pexp,Rexp):
        '''Finds ck metric
        
        Args:
            amin (float):
                minimum semi-major axis value in AU
            amax (float):
                maximum semi-major axis value in AU
            smin (ndarray):
                1D array of minimum separation values in AU
            smax (ndarray):
                1D array of maximum separation values in AU
            Cmin (float):
                minimum contrast value
            pexp (float):
                expected value of geometric albedo
            Rexp (float):
                expected value of planetary radius in AU
            
        Returns:
            ck (ndarray):
                1D array of ck metric
        
        '''
        
        an = 1.0/np.log(amax/amin)
        cg = an*(np.sqrt(1.0-(smax/amax)**2) - np.sqrt(1.0-(smin/amax)**2) + np.log(smax/(np.sqrt(1.0-(smax/amax)**2)+1.0))-np.log(smin/(np.sqrt(1.0-(smin/amax)**2)+1.0)))
        
        # calculate ck
        anp = an/cg 
        # intermediate values
        k1 = np.cos(0.5*(np.pi-np.arcsin(smin/amax)))**4/amax**2
        k2 = np.cos(0.5*(np.pi-np.arcsin(smax/amax)))**4/amax**2
        k3 = np.cos(0.5*np.arcsin(smax/amax))**4/amax**2
        k4 = 27.0/64.0*smax**(-2)
        k5 = np.cos(0.5*np.arcsin(smin/amax))**4/amax**2
        k6 = 27.0/64.0*smin**(-2)
        
        # set up
        z = sympy.Symbol('z', positive=True)
        k = sympy.Symbol('k', positive=True)
        b = sympy.Symbol('b', positive=True)
        # solve
        sol = solve(z**4 - z**3/sympy.sqrt(k) + b**2/(4*k), z)
        # third and fourth roots give valid roots
        # lambdify these roots
        sol3 = sympy.lambdify((k,b), sol[2], "numpy")
        sol4 = sympy.lambdify((k,b), sol[3], "numpy")
        
        # find ck   
        ck = np.zeros(smin.shape)
        kmin = Cmin/(pexp*Rexp**2)
        for i in xrange(len(ck)):
            if smin[i] == smax[i]:
                ck[i] = 0.0
            else:
                # equations to integrate
                al1 = lambda k: sol3(k,smin[i])
                au1 = lambda k: sol4(k,smin[i])
                au2 = lambda k: sol3(k,smax[i])
                al2 = lambda k: sol4(k,smax[i])
                
                al1 = np.vectorize(al1)
                au1 = np.vectorize(au1)
                au2 = np.vectorize(au2)
                al2 = np.vectorize(al2)
                
                f12 = lambda k: anp[i]/(2.0*np.sqrt(k))*(amax - al1(k))
                f23 = lambda k: anp[i]/(2.0*np.sqrt(k))*(au2(k) - al1(k))
                f34 = lambda k: anp[i]/(2.0*np.sqrt(k))*(amax - al2(k) + au2(k) - al1(k))
                f45 = lambda k: anp[i]/(2.0*np.sqrt(k))*(amax - al1(k))
                f56 = lambda k: anp[i]/(2.0*np.sqrt(k))*(au1(k) - al1(k))
                f35 = lambda k: anp[i]/(2.0*np.sqrt(k))*(amax - al2(k) + au2(k) - al1(k))
                f54 = lambda k: anp[i]/(2.0*np.sqrt(k))*(au1(k) - al2(k) + au2(k) - al1(k))
                f46 = lambda k: anp[i]/(2.0*np.sqrt(k))*(au1(k) - al1(k))
                
                if k4[i] < k5[i]:
                    if kmin < k1[i]:
                        ck[i] = integrate.quad(f12,k1[i],k2[i],limit=100)[0]
                        if k2[i] != k3[i]:
                            ck[i] += integrate.quad(f23,k2[i],k3[i],limit=100)[0]
                        ck[i] += integrate.quad(f34,k3[i],k4[i],limit=100)[0]
                        ck[i] += integrate.quad(f45,k4[i],k5[i],limit=100)[0]
                        ck[i] += integrate.quad(f56,k5[i],k6[i],limit=100)[0]
                    elif (kmin > k1[i]) and (kmin < k2[i]):
                        ck[i] = integrate.quad(f12,kmin,k2[i],limit=100)[0]
                        if k2[i] != k3[i]:
                            ck[i] += integrate.quad(f23,k2[i],k3[i],limit=100)[0]
                        ck[i] += integrate.quad(f34,k3[i],k4[i],limit=100)[0]
                        ck[i] += integrate.quad(f45,k4[i],k5[i],limit=100)[0]
                        ck[i] += integrate.quad(f56,k5[i],k6[i],limit=100)[0]
                    elif (kmin > k2[i]) and (kmin < k3[i]):
                        ck[i] = integrate.quad(f23,kmin,k3[i],limit=100)[0]
                        ck[i] += integrate.quad(f34,k3[i],k4[i],limit=100)[0]
                        ck[i] += integrate.quad(f45,k4[i],k5[i],limit=100)[0]
                        ck[i] += integrate.quad(f56,k5[i],k6[i],limit=100)[0]
                    elif (kmin > k3[i]) and (kmin < k4[i]):
                        ck[i] = integrate.quad(f34,kmin,k4[i],limit=100)[0]
                        ck[i] += integrate.quad(f45,k4[i],k5[i],limit=100)[0]
                        ck[i] += integrate.quad(f56,k5[i],k6[i],limit=100)[0]
                    elif (kmin > k4[i]) and (kmin < k5[i]):
                        ck[i] = integrate.quad(f45,kmin,k5[i],limit=100)[0]
                        ck[i] += integrate.quad(f56,k5[i],k6[i],limit=100)[0]
                    elif (kmin < k6[i]):
                        ck[i] = integrate.quad(f56,kmin,k6[i],limit=100)[0]
                    else:
                        ck[i] = 0.0
#                        print 'kmin: %r / k6: %r' % (kmin,k6[i])
                else:
                    if kmin < k1[i]:
                        ck[i] = integrate.quad(f12,k1[i],k2[i],limit=100)[0]
                        if k2[i] != k3[i]:
                            ck[i] += integrate.quad(f23,k2[i],k3[i],limit=100)[0]
                        ck[i] += integrate.quad(f35,k3[i],k5[i],limit=100)[0]
                        ck[i] += integrate.quad(f54,k5[i],k4[i],limit=100)[0]
                        ck[i] += integrate.quad(f46,k4[i],k6[i],limit=100)[0]
                    elif (kmin > k1[i]) and (kmin < k2[i]):
                        ck[i] = integrate.quad(f12,kmin,k2[i],limit=100)[0]
                        if k2[i] != k3[i]:
                            ck[i] += integrate.quad(f23,k2[i],k3[i],limit=100)[0]
                        ck[i] += integrate.quad(f35,k3[i],k5[i],limit=100)[0]
                        ck[i] += integrate.quad(f54,k5[i],k4[i],limit=100)[0]
                        ck[i] += integrate.quad(f46,k4[i],k6[i],limit=100)[0]
                    elif (kmin > k2[i]) and (kmin < k3[i]):
                        ck[i] = integrate.quad(f23,kmin,k3[i],limit=100)[0]
                        ck[i] += integrate.quad(f35,k3[i],k5[i],limit=100)[0]
                        ck[i] += integrate.quad(f54,k5[i],k4[i],limit=100)[0]
                        ck[i] += integrate.quad(f46,k4[i],k6[i],limit=100)[0]
                    elif (kmin > k3[i]) and (kmin < k5[i]):
                        ck[i] = integrate.quad(f35,kmin,k5[i],limit=100)[0]
                        ck[i] += integrate.quad(f54,k5[i],k4[i],limit=100)[0]
                        ck[i] += integrate.quad(f46,k4[i],k6[i],limit=100)[0]
                    elif (kmin > k5[i]) and (kmin < k4[i]):
                        ck[i] = integrate.quad(f54,kmin,k4[i],limit=100)[0]
                        ck[i] += integrate.quad(f46,k4[i],k6[i],limit=100)[0]
                    elif (kmin < k6[i]):
                        ck[i] = integrate.quad(f46,kmin,k6[i],limit=100)[0]
                    else:
                        ck[i] = 0.0
#                        print 'kmin: %r / k6: %r' % (kmin,k6[i])
                
        return ck

    def select_obs(self,t0,maxTime,ck):
        '''Selects stars for observation using ortools
        
        Args:
            t0 (ndarray):
                1D array of integration times in days
            maxTime (float):
                total observation time allotted in days
            ck (ndarray):
                1D array of ck metric
        
        Returns:
            sInds (ndarray):
                1D array of star indices selected for observation
        
        '''
        
        #set up solver
        solver = pywraplp.Solver('SolveIntegerProblem',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        #need one var per state
        xs = [ solver.IntVar(0.0,1.0, 'x'+str(j)) for j in range(len(ck)) ]
        #constraint is x_i*t_i < maxtime
        constraint1 = solver.Constraint(-solver.infinity(),maxTime)
        for j,x in enumerate(xs):
            constraint1.SetCoefficient(x, t0[j])
        #objective is max x_i*comp_i
        objective = solver.Objective()
        for j,x in enumerate(xs):
            objective.SetCoefficient(x, ck[j])
        objective.SetMaximization()
        res = solver.Solve()
        print 'Objective function value: %r' % (solver.Objective().Value())
        #collect result
        xs2 = np.array([x.solution_value() for x in xs])
        
        # observed star indices for depth of search calculations
        sInds = np.where(xs2>0)[0]
        
        return sInds

    def find_occurrence(self,Mass,ddP,ddR,R,P,Matrix,aedges,Redges,fa,amin):
        '''Extrapolates occurrence rates from Mulders 2015
        
        Args:
            Mass (Quantity):
                Stellar type mass astropy Quantity in kg
            ddP (ndarray):
                2D array of log differences in period (days) from Mulders
            ddR (ndarray):
                2D array of log differences in planetary radius (R_earth) from Mulders
            R (ndarray):
                1D array of planetary radius values from Mulders
            P (Quantity):
                1D array of period values astropy Quantity in days from Mulders
            Matrix (ndarray):
                2D array of occurrence rates from Mulders
            aedges (ndarray):
                1D array of desired semi-major axis grid in AU
            Redges (ndarray):
                1D array of desired planetary radius grid in R_earth
            fa (callable):
                probability density function of semi-major axis
            amin (float):
                minimum semi-major axis in AU
        
        Returns:
            etas (ndarray):
                2D array of extrapolated occurrence rates
        
        '''
        
        sma = ((const.G*Mass*P**2/(4.0*np.pi**2))**(1.0/3.0)).decompose().to('AU').value
        
        occ = Matrix*ddP*ddR
        occAll = np.sum(occ, axis=1)
        
        etas = np.zeros((len(Redges)-1,len(aedges)-1))
        fac1 = integrate.quad(fa, amin, sma[-1])[0]
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
            fac2 = integrate.quad(fa, aedges[i], aedges[i+1])[0]
            etas[:,i] = Rvals*fac2/fac1
        
        return etas

    def plot_dos(self,targ,name,path=None):
        '''Plots depth of search as a filled contour plot with contour lines
        
        Args:
            targ (str):
                string indicating which key to access from depth of search 
                result dictionary
            name (str):
                string indicating what to put in title of figure
            path (str):
                desired path to save figure (pdf, optional)
        
        '''
        
        acents = 0.5*(self.result['aedges'][1:]+self.result['aedges'][:-1])
        a = np.hstack((self.result['aedges'][0],acents,self.result['aedges'][-1]))
        a = np.around(a,4)
        Rcents = 0.5*(self.result['Redges'][1:]+self.result['Redges'][:-1])
        R = np.hstack((self.result['Redges'][0],Rcents,self.result['Redges'][-1]))
        R = np.around(R,4)
        DoS = self.result['DoS'][targ]
        # extrapolate to left-most boundary
        tmp = DoS[:,0] + (a[0]-a[1])*((DoS[:,1]-DoS[:,0])/(a[2]-a[1]))
        DoS = np.insert(DoS, 0, tmp, axis=1)
        # extrapolate to right-most boundary
        tmp = DoS[:,-1] + (a[-1]-a[-2])*((DoS[:,-1]-DoS[:,-2])/(a[-2]-a[-3]))
        DoS = np.insert(DoS, -1, tmp, axis=1)
        # extrapolate to bottom-most boundary
        tmp = DoS[0,:] + (R[0]-R[1])*((DoS[1,:]-DoS[0,:])/(R[2]-R[1]))
        DoS = np.insert(DoS, 0, tmp, axis=0)
        # extrapolate to upper-most boundary
        tmp = DoS[-1,:] + (R[-1]-R[-2])*((DoS[-1,:]-DoS[-2,:])/(R[-2]-R[-3]))
        DoS = np.insert(DoS, -1, tmp, axis=0)
        DoS = np.ma.masked_where(DoS<=0.0, DoS)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cs = ax.contourf(a,R,DoS,locator=ticker.LogLocator())
        cs2 = ax.contour(a,R,DoS,levels=cs.levels[1:],colors='k')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('a (AU)')
        ax.set_ylabel('R ($R_\oplus$)')
        ax.set_title('Depth of Search - '+name+' ('+str(self.result['NumObs'][targ])+')')
        cbar = fig.colorbar(cs)
        ax.clabel(cs2, fmt=ticker.LogFormatterMathtext(), colors='k')
        if path != None:
            fig.savefig(path, format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
        fig.show()

    def plot_nplan(self,targ,name,path=None):
        '''Plots depth of search convolved with occurrence rates as a filled 
        contour plot with contour lines
        
        Args:
            targ (str):
                string indicating which key to access from depth of search 
                result dictionary
            name (str):
                string indicating what to put in title of figure
            path (str):
                desired path to save figure (pdf, optional)
        
        '''
        
        acents = 0.5*(self.result['aedges'][1:]+self.result['aedges'][:-1])
        a = np.hstack((self.result['aedges'][0],acents,self.result['aedges'][-1]))
        a = np.around(a,4)
        Rcents = 0.5*(self.result['Redges'][1:]+self.result['Redges'][:-1])
        R = np.hstack((self.result['Redges'][0],Rcents,self.result['Redges'][-1]))
        R = np.around(R,4)
        DoS_occ = self.result['DoS_occ'][targ]
        # extrapolate to left-most boundary
        tmp = DoS_occ[:,0] + (a[0]-a[1])*((DoS_occ[:,1]-DoS_occ[:,0])/(a[2]-a[1]))
        DoS_occ = np.insert(DoS_occ, 0, tmp, axis=1)
        # extrapolate to right-most boundary
        tmp = DoS_occ[:,-1] + (a[-1]-a[-2])*((DoS_occ[:,-1]-DoS_occ[:,-2])/(a[-2]-a[-3]))
        DoS_occ = np.insert(DoS_occ, -1, tmp, axis=1)
        # extrapolate to bottom-most boundary
        tmp = DoS_occ[0,:] + (R[0]-R[1])*((DoS_occ[1,:]-DoS_occ[0,:])/(R[2]-R[1]))
        DoS_occ = np.insert(DoS_occ, 0, tmp, axis=0)
        # extrapolate to upper-most boundary
        tmp = DoS_occ[-1,:] + (R[-1]-R[-2])*((DoS_occ[-1,:]-DoS_occ[-2,:])/(R[-2]-R[-3]))
        DoS_occ = np.insert(DoS_occ, -1, tmp, axis=0)
        DoS_occ = np.ma.masked_where(DoS_occ <= 0.0, DoS_occ)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cs = ax.contourf(a,R,DoS_occ,locator=ticker.LogLocator())
        cs2 = ax.contour(a,R,DoS_occ,levels=cs.levels[1:],colors='k')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('a (AU)')
        ax.set_ylabel('R ($R_\oplus$)')
        ax.set_title('Number of Planets - '+name+' ('+str(self.result['NumObs'][targ])+')')
        cbar = fig.colorbar(cs)
        ax.clabel(cs2, fmt=ticker.LogFormatterMathtext(), colors='k')
        if path != None:
            fig.savefig(path, format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
        fig.show()
    
    def save_results(self, path):
        '''Saves results and outspec dictionaries to disk
        
        Args:
            path (str):
                string containing path for saved results
        
        '''
        
        x = {'Results': self.result, 'outspec': self.outspec}
        pickle.dump(x, open(path,'wb'))
        print 'Results saved as '+path
        
    def save_json(self, path):
        '''Saves json file used to generate results to disk
        
        Args:
            path (str):
                string containing directory path for file
        
        '''
        
        self.sim.genOutSpec(tofile=path)
        print 'json script saved as '+path
        
    def save_csvs(self, directory):
        '''Saves results as individual csv files to disk
        
        Args:
            directory (str):
                string containing directory path for files
                
        '''
        
        # save aedges and Redges first
        np.savetxt(directory+'/aedges.csv', self.result['aedges'], delimiter=', ')
        np.savetxt(directory+'/Redges.csv', self.result['Redges'], delimiter=', ')
        
        # save NumObs
        keys = self.result['NumObs'].keys()
        x = []
        h = ', '
        for i in xrange(len(keys)):
            x.append(self.result['NumObs'][keys[i]])
            h += keys[i]+', '
        h += '\n'
        np.savetxt(directory+'/NumObs.csv', x, delimiter=', ', newline=', ', header=h)
        
        # save DoS
        for key in self.result['DoS'].keys():
            np.savetxt(directory+'/DoS_'+key+'.csv', self.result['DoS'][key], delimiter=', ')
        
        # save occ_rates
        for key in self.result['occ_rates'].keys():
            np.savetxt(directory+'/occ_rates_'+key+'.csv', self.result['occ_rates'][key], delimiter=', ')
        
        # save DoS_occ
        for key in self.result['DoS_occ'].keys():
            np.savetxt(directory+'/DoS_occ_'+key+'.csv', self.result['DoS_occ'][key], delimiter=', ')
            
        