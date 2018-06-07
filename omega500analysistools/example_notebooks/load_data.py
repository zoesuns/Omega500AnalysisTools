import omega500analysistools.IO.load_db as load_db
from omega500fitstools.IO.read_Omega500 import *
from scipy.stats.stats import pearsonr   
from scipy.stats import spearmanr
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import pylab
from scipy import stats
import pandas as pd
import numpy as np
import sqlite3

print "smt contain info about all z=0 CLs' mass at different epoch"
print " merger (calc by r200m) contain T/F info about whether CL experienced merger after a=?"
print "groupbyz0id.get_group(CLno) gives all info for CLno mass(z)"
print "allelldata, 85*15, 85 CLs, [0] is CL id, following 3D gas ell at rlist r500c"
print "e.g. allelldata[:,0] gives all CLids, in increasing number order"
print " allelldata[?,4] gives CL? ell at rlist[4]=0.3 r500c"


########################################
####### load acc/merger data ###########
# smt contain info about all z=0 CLs' mass at different epoch
# merger contain T/F info about whether CL experienced merger after a=? 
smt=pd.read_csv('../data/SMT_NR.csv') 
merger=pd.read_csv('../data/mergers.csv')
# groupbyz0id.get_group(CLno) gives all info for CLno mass(z)
groupbyz0id=smt.groupby('z0_parent_id')
#----------------------------------------


########################################
####### load 3D gas ell data ###########

Lv8elldata=np.loadtxt('../data/Lv8_r500c_ell_NR_gas.txt')
Lv7elldata=np.loadtxt('../data/Lv7_r500c_ell_NR_gas.txt')
Lv6elldata=np.loadtxt('../data/Lv6_r500c_ell_NR_gas.txt')

rlist=[0,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.1,1.2,1.3]
# allelldata, 85*15, 85 CLs, [0] is CL id, following 3D gas ell at rlist r500c
# e.g. allelldata[:,0] gives all CLids, in increasing number order
#      allelldata[?,4] gives CL? ell at 0.3 r500c
allelldata=np.concatenate((np.concatenate((Lv8elldata,Lv7elldata[:,1:]),axis=1),Lv6elldata[:,1:]),axis=1)
allz0id=allelldata[:,0]
#-----------------------------------------


#########################################
def calc_Gamma(acutGamma=0.7,Mdef='M_total_200m'):
    print 'default definition is M_total_200m'
    Gamma=[]
    ia=int(40-40*acutGamma)
    for CLid in allz0id:
        group=groupbyz0id.get_group(CLid).iloc[0:25]
        nom=(np.log10(group[Mdef].iloc[0])-np.log10(group[Mdef].iloc[ia]))
        denom=(np.log10(group['aexp'].iloc[0])-np.log10(group['aexp'].iloc[ia]))
        Gamma.append(nom/denom)
    return np.array(Gamma)



##############################
#### load big database #######
database = '/Users/hqchen/filacf/Omega500/databases/L500_NR_0.db'
newdatabase = '/Users/hqchen/filacf/Omega500/databases/L500_NR_0.db.new'

df_new=load_db.return_table(newdatabase)


df=load_db.return_table(database)
df['halos'].columns

Mtot500c=df['halos']['M_total_500c'][df['halos']['aexp']>1].as_matrix()
Mtot200m=df['halos']['M_total_200m'][df['halos']['aexp']>1].as_matrix()

Mgas500c=df['halos']['M_gas_500c'][df['halos']['aexp']>1].as_matrix()

######################################
#### load observable file ############
obsfile= '/Users/hqchen/filacf/Omega500/databases/NR_mass_observables_a1.0005.txt'
obsfile_90= '/Users/hqchen/filacf/Omega500/databases/NR_mass_observables_a1.0005_2.txt'
ids,r500c,M500c,Lx,Tx,Mgas,Yx=np.loadtxt(obsfile,unpack=True)

ids_90, r500c_90, M500c_90, Mhse500c_90, Mgas500c_90, Tx500cEz_tot_90, Tx500cEz_tot_nocore_90, Tx500cEz_bulk_90, Tx500cEz_bulk_nocore_90, Lx500cEz_tot_90,  Lx500cEz_tot_nocore_90,  Lx500cEz_bulk_90,  Lx500cEz_bulk_nocore_90, fnt_90 = np.loadtxt(obsfile_90,unpack=True)

    
