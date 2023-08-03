from __future__ import division
import numpy as np
from scipy import integrate
from scipy import interpolate
from const import *
from const_ion import *

def assignL(INPUT):
    #Given inputs [logQ,lognH,logZ,T4,VOII2VHII,VOIII2VHII], compute HII region line luminosities in unit of L_sun.
    #Q:          Incident spectrum hydrogen ionization photon generation rate [s^-1]
    #nH:         HII region hydrogen number density [cm^-3]
    #Z:          HII region gas phase metallicity in unit of (O/H)/10**-3.31
    #            Here O/H is the oxygen to hydrogen number density ratio. O/H=10**-3.31 is the solar value. 
    #T4:         HII region gas temperature T [K]/10000 [K]
    #VOII2VHII:  VOII/VHII
    #VOIII2VHII: VOIII/VHII
    logQ, lognH, logZ, T4, VOII2VHII, VOIII2VHII = INPUT[0], INPUT[1], INPUT[2], INPUT[3], INPUT[4], INPUT[5]
    nO     = 10**-3.31*(10**logZ)*10**lognH          #number density of O
    ne     = 10**lognH*(1+0.0737+0.0293*(10**logZ))  #number density of free electron
    logne  = np.log10(ne)
    #Solve OIII level population abundances
    A      = np.zeros((4,4))
    B      = np.array([R01_OIII(ne,T4),R02_OIII(ne,T4),R03_OIII(ne,T4),R04_OIII(ne,T4)])

    A[0,0] =  R10_OIII(ne,T4)+R12_OIII(ne,T4)+R13_OIII(ne,T4)+R14_OIII(ne,T4)
    A[0,1] = -R21_OIII(ne,T4)
    A[0,2] = -R31_OIII(ne,T4)
    A[0,3] = -R41_OIII(ne,T4)
    A[1,0] = -R12_OIII(ne,T4)
    A[1,1] =  R20_OIII(ne,T4)+R21_OIII(ne,T4)+R23_OIII(ne,T4)+R24_OIII(ne,T4)
    A[1,2] = -R32_OIII(ne,T4)
    A[1,3] = -R42_OIII(ne,T4)
    A[2,0] = -R13_OIII(ne,T4)
    A[2,1] = -R23_OIII(ne,T4)
    A[2,2] =  R30_OIII(ne,T4)+R31_OIII(ne,T4)+R32_OIII(ne,T4)+R34_OIII(ne,T4)
    A[2,3] = -R43_OIII(ne,T4)
    A[3,0] = -R14_OIII(ne,T4)
    A[3,1] = -R24_OIII(ne,T4)
    A[3,2] = -R34_OIII(ne,T4)
    A[3,3] =  R40_OIII(ne,T4)+R41_OIII(ne,T4)+R42_OIII(ne,T4)+R43_OIII(ne,T4)

    levelPop = np.linalg.solve(A,B)
    
    n0       = nO/(1+np.sum(levelPop))
    n1       = n0*levelPop[0]
    n2       = n0*levelPop[1]
    n3       = n0*levelPop[2]
    n4       = n0*levelPop[3]
    
    logL10_OIII = np.log10(h)+np.log10(nu10_OIII*n1*A10_OIII/alphaB_HI(T4)*Jps2Lsun)+logQ-lognH-logne+np.log10(VOIII2VHII)
    logL21_OIII = np.log10(h)+np.log10(nu21_OIII*n2*A21_OIII/alphaB_HI(T4)*Jps2Lsun)+logQ-lognH-logne+np.log10(VOIII2VHII)
    logL31_OIII = np.log10(h)+np.log10(nu31_OIII*n3*A31_OIII/alphaB_HI(T4)*Jps2Lsun)+logQ-lognH-logne+np.log10(VOIII2VHII)
    logL32_OIII = np.log10(h)+np.log10(nu32_OIII*n3*A32_OIII/alphaB_HI(T4)*Jps2Lsun)+logQ-lognH-logne+np.log10(VOIII2VHII)
    logL43_OIII = np.log10(h)+np.log10(nu43_OIII*n4*A43_OIII/alphaB_HI(T4)*Jps2Lsun)+logQ-lognH-logne+np.log10(VOIII2VHII)
    
    #Solve OII level population abundances
    A      = np.zeros((4,4))
    B      = np.array([R01_OII(ne,T4),R02_OII(ne,T4),R03_OII(ne,T4),R04_OII(ne,T4)])

    A[0,0] =  R10_OII(ne,T4)+R12_OII(ne,T4)+R13_OII(ne,T4)+R14_OII(ne,T4)
    A[0,1] = -R21_OII(ne,T4)
    A[0,2] = -R31_OII(ne,T4)
    A[0,3] = -R41_OII(ne,T4)
    A[1,0] = -R12_OII(ne,T4)
    A[1,1] =  R20_OII(ne,T4)+R21_OII(ne,T4)+R23_OII(ne,T4)+R24_OII(ne,T4)
    A[1,2] = -R32_OII(ne,T4)
    A[1,3] = -R42_OII(ne,T4)
    A[2,0] = -R13_OII(ne,T4)
    A[2,1] = -R23_OII(ne,T4)
    A[2,2] =  R30_OII(ne,T4)+R31_OII(ne,T4)+R32_OII(ne,T4)+R34_OII(ne,T4)
    A[2,3] = -R43_OII(ne,T4)
    A[3,0] = -R14_OII(ne,T4)
    A[3,1] = -R24_OII(ne,T4)
    A[3,2] = -R34_OII(ne,T4)
    A[3,3] =  R40_OII(ne,T4)+R41_OII(ne,T4)+R42_OII(ne,T4)+R43_OII(ne,T4)

    levelPop = np.linalg.solve(A,B)
    
    n0       = nO/(1+np.sum(levelPop))
    n1       = n0*levelPop[0]
    n2       = n0*levelPop[1]
    n3       = n0*levelPop[2]
    n4       = n0*levelPop[3]
    
    logL10_OII  = np.log10(h)+np.log10(nu10_OII*n1*A10_OII/alphaB_HI(T4)*Jps2Lsun)+logQ-lognH-logne+np.log10(VOII2VHII)
    logL20_OII  = np.log10(h)+np.log10(nu20_OII*n2*A20_OII/alphaB_HI(T4)*Jps2Lsun)+logQ-lognH-logne+np.log10(VOII2VHII)
    logL30_OII  = np.log10(h)+np.log10(nu30_OII*n3*A30_OII/alphaB_HI(T4)*Jps2Lsun)+logQ-lognH-logne+np.log10(VOII2VHII)
    logL31_OII  = np.log10(h)+np.log10(nu31_OII*n3*A31_OII/alphaB_HI(T4)*Jps2Lsun)+logQ-lognH-logne+np.log10(VOII2VHII)
    logL32_OII  = np.log10(h)+np.log10(nu32_OII*n3*A32_OII/alphaB_HI(T4)*Jps2Lsun)+logQ-lognH-logne+np.log10(VOII2VHII)
    logL40_OII  = np.log10(h)+np.log10(nu40_OII*n4*A40_OII/alphaB_HI(T4)*Jps2Lsun)+logQ-lognH-logne+np.log10(VOII2VHII)
    
    logLHalpha  = np.log10(h)+np.log10(nu_Halpha*alphaB_Halpha(T4)/alphaB_HI(T4)*Jps2Lsun)+logQ
    logLHbeta   = np.log10(h)+np.log10(nu_Hbeta*alphaB_Hbeta(T4)/alphaB_HI(T4)*Jps2Lsun)+logQ
    return logL10_OIII, logL21_OIII, logL31_OIII, logL32_OIII, logL43_OIII, logL10_OII, logL20_OII, logL30_OII, logL31_OII, logL32_OII, logL40_OII, logLHalpha, logLHbeta

def y(nHI_para,nHeI_para,nuy):
    #Given inputs [nHI,nHeI], return fraction of photons with energy above 24.60 eV that ionize hydrogen.
    #nHI : number density of neutral hydrogen.
    #nHeI: number density of neutral helium.
    #nuy:  (24.59eV+kb*T_gas)/h
    if nHI_para==0 and nHeI_para==0:
        return 0.5
    else:
        return nHI_para*sigma_HI(nuy)/(nHI_para*sigma_HI(nuy)+nHeI_para*sigma_HeI(nuy))

def solV(nu,L,logQ,lognH,logZ,T4,eps=0.1,returnProfiles=False):
    #Given  input [nu,L,logQ,lognH,logZ,T4], return VOII/VHII and VOIII/VHII
    #nu:    Incident spectrum frequency array in unit of Hz
    #L:     Incident spectrum array in unit of J/s/Hz
    #       L and nu determine the incident spectrum shape
    #logQ:  log10(Q), Q is the incident spectrum hydrogen ionizing photon generation rate in unit of s^-1
    #lognH: log10(nH), nH is the HII region gas hydrogen density in unit of cm^-3
    #logZ:  log10(Z), Z is the HII region gas phase metallicity (O/H)/10**-3.31
    #T4:    HII retion gas temperature T/10000K
    #eps:   Maximum fractional variation of neutral hydrogen (nHI) and helium number density (nHeI) in adjacent radial bins.
    #returnProfiles: If True, return fractional abundance radial profiles for HI,HeI,OI,OII,OIII, and VOIII/VHII, VOII/VHII.
    #                If False, return VOIII/VHII, VOII/VHII.

    idxValid   = np.where(L>0)[0]
    nu         = nu[idxValid]
    L          = L[idxValid]
    fspectrum  = interpolate.interp1d(nu,L)
    #intbar     = lambda nu: fspectrum(nu)/h/nu
    #QHI        = integrate.quad(intbar,13.6*eV2J/h ,nu[-1])[0] #Compute QHI for the incident spectrum
    nu_idx     = np.where(nu>13.6*eV2J/h)[0]
    nu_sample  = np.append(13.6*eV2J/h,nu[nu_idx])
    L_sample   = fspectrum(nu_sample)/h/nu_sample
    QHI        = integrate.simpson(L_sample,nu_sample)
    L          = 10**(np.log10(L)+logQ-np.log10(QHI))          #Renormalize the incident spectrum such that its amplitude matches the input Q

    E          = nu*h/eV2J                           #eV
    dnu        = np.append(nu[0],nu[1:]-nu[:-1])
    idx_H      = np.where(E>13.6)[0]
    idx_HI     = np.where((E>13.6)&(E<=24.59))[0]
    idx_HeI    = np.where(E>24.59)[0]
    idx_OII    = np.where(E>35.12)[0]
    idx_OI     = np.where(E>13.62)[0]
 
    nH         = 10**lognH
    Z          = 10**logZ
    nHe        = nH*(0.0737+0.0293*Z)
    nO         = nH*10**-3.31*Z
    nuy        = np.array([(24.59*eV2J+kb*T4*10000)/h])
    #Stromgren Radius
    logR_HII   = 1/3*(np.log10(3)+logQ-np.log10(4*np.pi*alphaB_HI(T4))-2*lognH)
    R_HII      = 10**logR_HII
    V_HII      = 4*np.pi/3*R_HII**3

    #Initial condition
    r_grid     = np.array([R_HII/100])
    nHI        = np.array([0])
    nHeI       = np.array([0])
    nOI        = np.array([0])
    nOII       = np.array([0])
    nOIII      = np.array([0])
    tau        = np.zeros(len(nu))
    while nHI[-1]<0.5*nH:
        dr_temp    = R_HII/100
        r_temp     = r_grid[-1]+dr_temp
        tau_temp          = np.zeros(len(tau))
        tau_temp[idx_HI]  = tau[idx_HI]+nHI[-1]*sigma_HI(nu[idx_HI])*dr_temp
        tau_temp[idx_HeI] = tau[idx_HeI]+nHI[-1]*sigma_HI(nu[idx_HeI])*dr_temp+nHeI[-1]*sigma_HeI(nu[idx_HeI])*dr_temp
        A          = L/nu/h*sigma_HI(nu)*np.exp(-tau_temp)*dnu
        A_coeff    = np.sum(A[idx_H])/4/np.pi/r_temp**2
        nHI_temp   = (2*nH*alphaB_HI(T4)+A_coeff-np.sqrt((2*nH*alphaB_HI(T4)+A_coeff)**2-4*alphaB_HI(T4)**2*nH**2))/2/alphaB_HI(T4)
        nHII_temp  = nH-nHI_temp
        A          = L/nu/h*sigma_HeI(nu)*np.exp(-tau_temp)*dnu
        B_coeff    = np.sum(A[idx_HeI])/4/np.pi/r_temp**2
        C_coeff    = (1-y(nHI[-1],nHeI[-1],nuy))*alpha1_HeI(T4)
        D_coeff    = -B_coeff+C_coeff*nHII_temp-alphaA_HeI(T4)*nHII_temp
        nHeII_temp = (-D_coeff-np.sqrt(D_coeff**2-4*(C_coeff-alphaA_HeI(T4))*B_coeff*nHe))/2/(C_coeff-alphaA_HeI(T4))
        nHeI_temp  = nHe-nHeII_temp
        if nHI[-1]>0 and nHeI[-1]>0:
            while nHI_temp/nHI[-1]>(1+eps) and nHeI_temp/nHeI[-1]>(1+eps) and r_grid[-1]/R_HII>0.5:
                dr_temp    = dr_temp/2
                r_temp     = r_grid[-1]+dr_temp
                tau_temp          = np.zeros(len(tau))
                tau_temp[idx_HI]  = tau[idx_HI]+nHI[-1]*sigma_HI(nu[idx_HI])*dr_temp
                tau_temp[idx_HeI] = tau[idx_HeI]+nHI[-1]*sigma_HI(nu[idx_HeI])*dr_temp+nHeI[-1]*sigma_HeI(nu[idx_HeI])*dr_temp
                A          = L/nu/h*sigma_HI(nu)*np.exp(-tau_temp)*dnu
                A_coeff    = np.sum(A[idx_H])/4/np.pi/r_temp**2
                nHI_temp   = (2*nH*alphaB_HI(T4)+A_coeff-np.sqrt((2*nH*alphaB_HI(T4)+A_coeff)**2-4*alphaB_HI(T4)**2*nH**2))/2/alphaB_HI(T4)
                A          = L/nu/h*sigma_HeI(nu)*np.exp(-tau_temp)*dnu
                B_coeff    = np.sum(A[idx_HeI])/4/np.pi/r_temp**2
                C_coeff    = (1-y(nHI[-1],nHeI[-1],nuy))*alpha1_HeI(T4)
                D_coeff    = -B_coeff+C_coeff*nHII_temp-alphaA_HeI(T4)*nHII_temp
                nHeII_temp = (-D_coeff-np.sqrt(D_coeff**2-4*(C_coeff-alphaA_HeI(T4))*B_coeff*nHe))/2/(C_coeff-alphaA_HeI(T4))
                nHeI_temp  = nHe-nHeII_temp
        tau      = tau_temp
        r_grid   = np.append(r_grid,r_temp)
        nHI      = np.append(nHI,nHI_temp)
        nHeI     = np.append(nHeI,nHeI_temp)

        ne        = nH-nHI_temp+nHe-nHeI_temp
        A_coeff   = L/nu/h*sigma_OII(nu)*np.exp(-tau)*dnu
        B_coeff   = np.sum(A_coeff[idx_OII])
        C_coeff   = np.sum(A_coeff[idx_OI])
        D_coeff   = B_coeff/4/np.pi/r_temp**2/(ne*alphaB_OIII+nHI_temp*delta_OII)
        E_coeff   = (C_coeff/4/np.pi/r_temp**2+(nH-nHI_temp)*k0r_OI_ct(T4))/(ne*alphaB_OII+nHI_temp*(k0_OI_ct(T4)+k1_OI_ct(T4)+k2_OI_ct(T4)))

        nOI       = np.append(nOI,nO/(1+E_coeff+E_coeff*D_coeff))
        nOII      = np.append(nOII,nOI[-1]*E_coeff)
        nOIII     = np.append(nOIII,nOII[-1]*D_coeff)

    VOII2HII   = np.sum(4*np.pi*r_grid[1:]**2*(r_grid[1:]-r_grid[:-1])*nOII[1:]/nO)/V_HII
    VOIII2HII  = np.sum(4*np.pi*r_grid[1:]**2*(r_grid[1:]-r_grid[:-1])*nOIII[1:]/nO)/V_HII
    if returnProfiles == True:
        return r_grid,nHI/nH,nHeI/nHe,nOI/nO,nOII/nO,nOIII/nO,VOII2HII,VOIII2HII #Output HI,HeI,OI,OII,OIII fractional abundance radial profiles
    if returnProfiles == False:
        return VOII2HII, VOIII2HII



