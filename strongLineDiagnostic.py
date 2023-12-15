from __future__ import division
import numpy as np
from const_ion import *

def MeasurelogZ(R_Obs,O3O2_Obs,Rmode,tag):
    logZQ        = np.linspace(-2,0,1000)
    ZQ           = 10**logZQ
    if tag == 'local':
        T4OIII           = -0.32*logZQ**2-1.5*logZQ+0.41
        T4OII            = -0.22*T4OIII**2+1.2*T4OIII+0.066
        T4OII[T4OII<0.5] = 0.5
    if tag == 'EoR':
        T4OIII           = 0.81*logZQ**2+ 0.14*logZQ+ 1.1
        T4OII            = -0.744+T4OIII*(2.338-0.610*T4OIII)
        T4OII[T4OII<0.5] = 0.5
    if tag == 'cosmicNoon':
        T4OIII           = 0.88*logZQ**2+ 0.44*logZQ+ 1.2
        T4OII            = -0.744+T4OIII*(2.338-0.610*T4OIII)
        T4OII[T4OII<0.5] = 0.5 
    idx_sol = []
    while len(idx_sol)==0:
        fO3O2  = 1/(0.75*k03_OIII(T4OIII)*nu32_OIII/(k01_OII(T4OII)*nu10_OII+k02_OII(T4OII)*nu20_OII))
        FOIII  = fO3O2*O3O2_Obs/(1+fO3O2*O3O2_Obs)
        FOII   = 1-FOIII
        T4HII  = T4OIII*FOIII+T4OII*FOII
        fR2    = 10**-3.31/nu_Hbeta/alphaB_Hbeta(T4HII)*(k01_OII(T4OII)*nu10_OII+k02_OII(T4OII)*nu20_OII)
        R2     = fR2*FOII*ZQ
        fR3    = 0.75*10**-3.31*k03_OIII(T4OIII)*nu32_OIII/nu_Hbeta/alphaB_Hbeta(T4HII)
        R3     = fR3*FOIII*ZQ
        fR3p   = 0.25*10**-3.31*k03_OIII(T4OIII)*nu31_OIII/nu_Hbeta/alphaB_Hbeta(T4HII)
        R3p    = fR3p*FOIII*ZQ
        R23    = R2+R3+R3p
        if Rmode == 'R2':
            R = R2
        if Rmode == 'R3':
            R = R3
        if Rmode == 'R23':
            R = R23
        if np.max(R)>=R_Obs:
            diff      = R-R_Obs
            diff_prod = diff[1:]*diff[:-1]
            idx_sol   = np.where(diff_prod<=0)[0]
        if np.max(R)<R_Obs:
            T4OIII       = T4OIII+0.1
            T4OII        = T4OII+0.1
    w1 = (R[idx_sol+1]-R_Obs)/(R[idx_sol+1]-R[idx_sol])
    w2 = 1-w1
    return w1*logZQ[idx_sol]+w2*logZQ[idx_sol+1]

def MeasurelogZ_withN2O2(R_Obs,O3O2_Obs,N2O2,Rmode,tag):
    logZQ        = np.linspace(-2,0,1000)
    ZQ           = 10**logZQ
    if tag == 'local':
        T4OIII           = -0.32*logZQ**2-1.5*logZQ+0.41
        T4OII            = -0.22*T4OIII**2+1.2*T4OIII+0.066
        T4OII[T4OII<0.5] = 0.5
    if tag == 'EoR':
        T4OIII           = 0.81*logZQ**2+ 0.14*logZQ+ 1.1
        T4OII            = -0.744+T4OIII*(2.338-0.610*T4OIII)
        T4OII[T4OII<0.5] = 0.5
    if tag == 'cosmicNoon':
        T4OIII           = 0.88*logZQ**2+ 0.44*logZQ+ 1.2
        T4OII            = -0.744+T4OIII*(2.338-0.610*T4OIII)
        T4OII[T4OII<0.5] = 0.5 
    idx_sol = []
    while len(idx_sol)==0:
        fO3O2  = 1/(0.75*k03_OIII(T4OIII)*nu32_OIII/(k01_OII(T4OII)*nu10_OII+k02_OII(T4OII)*nu20_OII))
        FOIII  = fO3O2*O3O2_Obs/(1+fO3O2*O3O2_Obs)
        FOII   = 1-FOIII
        T4HII  = T4OIII*FOIII+T4OII*FOII
        fR2    = 10**-3.31/nu_Hbeta/alphaB_Hbeta(T4HII)*(k01_OII(T4OII)*nu10_OII+k02_OII(T4OII)*nu20_OII)
        R2     = fR2*FOII*ZQ
        fR3    = 0.75*10**-3.31*k03_OIII(T4OIII)*nu32_OIII/nu_Hbeta/alphaB_Hbeta(T4HII)
        R3     = fR3*FOIII*ZQ
        fR3p   = 0.25*10**-3.31*k03_OIII(T4OIII)*nu31_OIII/nu_Hbeta/alphaB_Hbeta(T4HII)
        R3p    = fR3p*FOIII*ZQ
        R23    = R2+R3+R3p
        if Rmode == 'R2':
            R = R2
        if Rmode == 'R3':
            R = R3
        if Rmode == 'R23':
            R = R23
        if np.max(R)>=R_Obs:
            diff      = R-R_Obs
            diff_prod = diff[1:]*diff[:-1]
            idx_sol   = np.where(diff_prod<=0)[0]
        if np.max(R)<R_Obs:
            T4OIII       = T4OIII+0.1
            T4OII        = T4OII+0.1
    w1 = (R[idx_sol+1]-R_Obs)/(R[idx_sol+1]-R[idx_sol])
    w2 = 1-w1
    
    T4OII_sol  = w1*T4OII[idx_sol]+w2*T4OII[idx_sol+1]
    T4OIII_sol = w1*T4OIII[idx_sol]+w2*T4OIII[idx_sol+1]
    N2O2_model = 0.0738*k03_NII(T4OII_sol)/(k01_OII(T4OII_sol)+k02_OII(T4OII_sol))
    idx_true   = np.where(np.abs(N2O2_model-N2O2)==np.min(np.abs(N2O2_model-N2O2)))[0]
    return w1[idx_true]*logZQ[idx_sol][idx_true]+w2[idx_true]*logZQ[idx_sol+1][idx_true]
