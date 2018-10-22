# -*- coding: utf-8 -*-
"""
Created on Tues March 7, 2018
Updated Mon Oct 22, 2018

@author: dg622@cornell.edu
"""

import numpy as np
import os
import EXOSIMS.MissionSim as MissionSim
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import astropy.constants as const
import astropy.units as u
try:
    import cPickle as pickle
except:
    import pickle
from DoSFuncs import DoSFuncs

class DoSFuncsMulders(DoSFuncs):
    '''Calculates depth of search values for a given input EXOSIMS json script. 
    Only stellar types M, K, G, and F are used. All other stellar types are 
    filtered out. Occurrence rates are extrapolated from data in Mulders 2015.
    
    'core_contrast' must be specified in the input json script as either a 
    path to a fits file or a constant value, otherwise the default contrast 
    value from EXOSIMS will be used
    
    path must be specified
    
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
        dMag (float):
            limiting dMag value for integration time calculation (optional)
        WA_targ (astropy Quantity):
            working angle for target astrophysical contrast (optional)
            
    Attributes:
        result (dict):
            dictionary containing results of the depth of search calculations
            Keys include:
                NumObs (dict):
                    dictionary containing number of observations for each 
                    stellar type, keys are: 'Mstars', 'Kstars', 'Gstars', 
                    'Fstars', and 'all'
                aedges (ndarray):
                    1D array of semi-major axis bin edges in AU
                Redges (ndarray):
                    1D array of planetary radius bin edges in R_earth
                DoS (dict):
                    dictionary containing 2D arrays of depth of search for
                    each stellar type, keys are: 'Mstars', 'Kstars', 'Gstars',
                    'Fstars', and 'all'
                occ_rates (dict):
                    dictionary containing 2D arrays of occurrence rates
                    extrapolated from Mulders 2015, keys are: 'Mstars', 'Kstars',
                    'Gstars', and 'Fstars'
                DoS_occ (dict):
                    dictionary containing 2D arrays of depth of search convolved
                    with the extrapolated occurrence rates, keys are: 'Mstars',
                    'Kstars', 'Gstars', 'Fstars', and 'all'
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
        assert Rmin < 45.0, 'Minimum planetary radius is above extrapolation range'
        if Rmin < 0.35:
            print 'Rmin reset to 0.35*R_earth'
            Rmin = 0.35
        Rmax = self.sim.PlanetPopulation.Rprange[1].to('earthRad').value
        assert Rmax > 0.35, 'Maximum planetary radius is below extrapolation range'
        if Rmax > 45.0:
            print 'Rmax reset to 45.0*R_earth'
        assert Rmax > Rmin, 'Maximum planetary radius is less than minimum planetary radius'
        # need to get Cmin from contrast curve
        mode = filter(lambda mode: mode['detectionMode'] == True, self.sim.OpticalSystem.observingModes)[0]
        WA = np.linspace(mode['IWA'], mode['OWA'], 50)
        syst = mode['syst']
        lam = mode['lam']
        if dMag is None:
            # use dMagLim when dMag not specified
            dMag = self.sim.Completeness.dMagLim
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
        
        t_int1 = self.sim.OpticalSystem.calc_intTime(self.sim.TargetList,np.array([0]),fZ,fEZ,dMag,WA_targ,mode)
        t_int1 = np.repeat(t_int1.value,len(WA))*t_int1.unit
        sInds = np.repeat(0,len(WA))
        fZ1 = np.repeat(fZ.value,len(WA))*fZ.unit
        fEZ1 = np.repeat(fEZ.value,len(WA))*fEZ.unit
        core_contrast = 10.0**(-0.4*self.sim.OpticalSystem.calc_dMag_per_intTime(t_int1,self.sim.TargetList,sInds,fZ1,fEZ1,WA,mode))
        contrast = interpolate.interp1d(WA.to('arcsec').value,core_contrast,kind='cubic',fill_value=1.0)
        opt = optimize.minimize_scalar(contrast,bounds=[mode['IWA'].to('arcsec').value,mode['OWA'].to('arcsec').value],method='bounded')
        Cmin = opt.fun
        
        # find expected values of p and R
        if self.sim.PlanetPopulation.prange[0] != self.sim.PlanetPopulation.prange[1]:
            if hasattr(self.sim.PlanetPopulation,'ps'):
                f = lambda R: self.sim.PlanetPopulation.get_p_from_Rp(R*u.earthRad)*self.sim.PlanetPopulation.dist_radius(R)
                pexp, err = integrate.quad(f,self.sim.PlanetPopulation.Rprange[0].value,\
                                           self.sim.PlanetPopulation.Rprange[1].value,\
                                           epsabs=0,epsrel=1e-6,limit=100)
            else:
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
        
        # calculate integration times
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
        ck = self.find_ck(amin,amax,smin,smax,Cmin,pexp,Rexp)
        # offset to account for zero ck values with nonzero completeness
        ck += ck[ck>0.0].min()*1e-2
        print 'Finished ck calculations'
        
        print 'Beginning ortools calculations to determine list of observed stars'
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
        fZ2 = np.repeat(fZ.value,len(WA))*fZ.unit
        fEZ2 = np.repeat(fEZ.value,len(WA))*fEZ.unit
        C_inst = np.zeros((len(sInds2),len(WA)))
        for i in xrange(len(sInds2)):
            t_int2 = np.repeat(t_int[i].value,len(WA))*t_int.unit
            sInds2a = np.repeat(sInds2[i],len(WA))
            C_inst[i,:] = 10.0**(-0.4*self.sim.OpticalSystem.calc_dMag_per_intTime(t_int2,self.sim.TargetList,sInds2a,fZ2,fEZ2,WA,mode))
        
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
              'Fstars':len(Flist), 'all':(len(Mlist)+len(Klist)+len(Glist)\
                           +len(Flist))}
        # store number of observed stars in result
        self.result['NumObs'] = NumObs
        # find bin edges for semi-major axis and planetary radius in AU
        aedges = np.logspace(np.log10(amin), np.log10(amax), abins+1)
        Redges = np.logspace(np.log10(Rmin*u.earthRad.to('AU')), \
                         np.log10(Rmax*u.earthRad.to('AU')), Rbins+1)
        # store aedges and Redges in result
        self.result['aedges'] = aedges
        self.result['Redges'] = Redges/u.earthRad.to('AU')
    
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
        DoS['all'] = DoS['Mstars'] + DoS['Kstars'] + DoS['Gstars'] + DoS['Fstars']
        # store DoS in result
        self.result['DoS'] = DoS
    
        # load occurrence data from file
        print 'Loading occurrence data'
        directory = os.path.dirname(os.path.abspath(__file__))
        rates = pickle.load(open(directory+'/Mulders.ocr','rb'))
    
        # values from Mulders
        Redges /= u.earthRad.to('AU')     
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
          
        # Multiply depth of search with occurrence rates
        r_norm = Redges[1:] - Redges[:-1]
        a_norm = aedges[1:] - aedges[:-1]
        norma, normR = np.meshgrid(a_norm,r_norm)
        DoS_occ = {}
        print 'Multiplying depth of search grid with occurrence rate grid'
        DoS_occ['Mstars'] = DoS['Mstars']*occ_rates['Mstars']*norma*normR
        DoS_occ['Kstars'] = DoS['Kstars']*occ_rates['Kstars']*norma*normR
        DoS_occ['Gstars'] = DoS['Gstars']*occ_rates['Gstars']*norma*normR
        DoS_occ['Fstars'] = DoS['Fstars']*occ_rates['Fstars']*norma*normR
        DoS_occ['all'] = DoS_occ['Mstars']+DoS_occ['Kstars']+DoS_occ['Gstars']+DoS_occ['Fstars']
        self.result['DoS_occ'] = DoS_occ
        
        # store MissionSim output specification dictionary
        self.outspec = self.sim.genOutSpec()
        print 'Calculations finished'
        
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