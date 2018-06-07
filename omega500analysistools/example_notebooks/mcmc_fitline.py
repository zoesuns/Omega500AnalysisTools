import numpy as np
import matplotlib.pyplot as plt

def mcmc_sample(x, nparams=2, nwalkers=100, nRval=100, modelpdf = None, 
               ipar_active = None, params=[], nsteps=1000000000, Rlim = 1.001):
    """
    MCMC sampler implementing the Goodman & Weare (2010) affine-invariant algorithm
    inner loop is vectorized
    
    run for nsteps or until R_GR=Rlim is reached, whichever comes first
    
    """
    
    try:
        import acor
    except:
        raise Exception("acor package is not installed! do: pip install acor")
        
    # parameters used to draw random number with the GW10 proposal distribution
    ap = 2.0; api = 1.0/ap; asqri=1.0/np.sqrt(ap); afact=(ap-1.0)

    # calculate effective number of parameters if some are specified to be fixed
    ia = (ipar_active==1)
    npareff = np.size(ipar_active[ia])
    print(("starting sampling with %d active parameters of the total %d parameters"%(npareff, nparams)))
    
    # initialize some auxiliary arrays and variables 
    chain = []; Rval = []; 

    naccept = 0; ntry = 0; nchain = 0
    mw = np.zeros((nwalkers,npareff)); sw = np.zeros((nwalkers,npareff))
    m = np.zeros(npareff); Wgr = np.zeros(npareff); Bgr = np.zeros(npareff); Rgr = np.zeros(npareff)
    
    mutx = []; taux = []
    for i in range(npareff): 
        mutx.append([]); taux.append([])
        Rval.append([])

    gxo = np.zeros((2,nwalkers/2))
    gxo[0,:] = modelpdf(x[0,:,:], params); gxo[1,:] = modelpdf(x[1,:,:], params)
    converged = False;
    while not converged:
        # for parallelization (not implemented here but the MPI version is available)
        # the walkers are split into two complementary sub-groups (see GW10)
        for kd in range(2):
            k = abs(kd-1)
            # vectorized inner loop of walkers stretch move in the Goodman & Weare sampling algorithm
            xchunk = x[k,:,:]
            jcompl = np.random.randint(0,nwalkers/2,nwalkers/2)
            xcompl = x[kd,jcompl,:]
            gxold  = gxo[k,:]
            zf= np.random.rand(nwalkers/2)   # the next few steps implement Goodman & Weare sampling algorithm
            zf = zf * afact; zr = (1.0+zf)*(1.0+zf)*api
            zrtile = np.transpose(np.tile(zr,(nparams,1))) # duplicate zr for nparams
            xtry  = xcompl + zrtile*(xchunk-xcompl)
            gxtry = modelpdf(xtry, params); gx    = gxold 
            gr   = gxtry - gx
            iacc = np.where(gr>0.)
            xchunk[iacc] = xtry[iacc]
            gxold[iacc] = gxtry[iacc]
            aprob = (npareff-1)*np.log(zr) + (gxtry - gx)
            u = np.random.uniform(0.0,1.0,np.shape(xchunk)[0])        
            iprob = np.where(aprob>np.log(u))
            xchunk[iprob] = xtry[iprob]
            gxold[iprob] = gxtry[iprob]
            naccept += len(iprob[0])

            x[k,:,ia] = np.transpose(xchunk[:,ia])
            gxo[k,:] = gxold        
            xdum = x[:,:,ia]

            for i in range(nwalkers/2):
                chain.append(np.array(xdum[k,i,:]))

            for i in range(nwalkers/2):
                mw[k*nwalkers/2+i,:] += xdum[k,i,:]
                sw[k*nwalkers/2+i,:] += xdum[k,i,:]**2
                ntry += 1

        nchain += 1
        
        # compute means for the auto-correlation time estimate
        for i in range(npareff):
            mutx[i].append(np.sum(xdum[:,:,i])/(nwalkers))

        # compute Gelman-Rubin indicator for all parameters
        if ( nchain%nRval == 0):
            # calculate Gelman & Rubin convergence indicator
            mwc = mw/(nchain-1.0)
            swc = sw/(nchain-1.0)-np.power(mwc,2)

            for i in range(npareff):
                # within chain variance
                Wgr[i] = np.sum(swc[:,i])/nwalkers
                # mean of the means over Nwalkers
                m[i] = np.sum(mwc[:,i])/nwalkers
                # between chain variance
                Bgr[i] = nchain*np.sum(np.power(mwc[:,i]-m[i],2))/(nwalkers-1.0)
                # Gelman-Rubin R factor
                Rgr[i] = (1.0 - 1.0/nchain + Bgr[i]/Wgr[i]/nchain)*(nwalkers+1.0)/nwalkers - (nchain-1.0)/(nchain*nwalkers)
                tacorx = acor.acor(np.abs(mutx[i]))[0]; taux[i].append(np.max(tacorx))
                Rval[i].append(Rgr[i]-1.0)

            print(("nchain = %d; tcorr = %.2e"%(nchain, np.max(tacorx))))
            print(("R_GR = ", Rgr))
            if (np.max(np.abs(Rgr-1.0)) < np.abs(Rlim-1.0)) or (nchain >= nsteps): converged = True
        
    print(("MCMC sampler generated %d samples using %d walkers"%(ntry, nwalkers)))
    print(("with step acceptance ratio of %.3f"%(1.0*naccept/ntry)))
        
    # record integer auto-correlation time at the final iteration
    nthin = int(tacorx)
    return chain, Rval, nthin

def mcmc_sample_init(nparams=2, nwalkers=100, x0=None, step=None, ipar_active=None):
    """
    distribute initial positions of walkers in an isotropic Gaussian around the initial point
    """
    np.random.seed()
    
    # in this implementation the walkers are split into 2 subgroups and thus nwalkers must be divisible by 2
    if nwalkers%2:
        raise ValueError("MCMCsample_init: nwalkers must be divisible by 2!")
         
    x = np.zeros([2,nwalkers/2,nparams])

    for i in range(nparams):
        x[:,:,i] = np.reshape(np.random.normal(x0[i],step[i],nwalkers),(2,nwalkers/2))
    ina = (ipar_active==0)
    if np.size(ina) > 0:
        x[:,:,ina] = x0[ina]
    return x

#from mcmc import mcmc_sample, mcmc_sample_init
cmin = -100; cmax = 100 # flat prior range on normalization
smin = 0.; smax = 2. # flat prior range on scatter
# normalization factor; 0.5 is from the normalization by the integral int^{infty}_{-infty} dm/(1+m^2)^{3/2}

normfact = 0.5/(smax - smin)/(cmax - cmin)

def prior(xd):
    """
    defines parameter priors
    
    Parameters
    ----------
    xd : numpy 1d array
        xd[0] = slope; xd[1] = normalization; xd[2] = scatter
        
    Returns
    -------
    numpy float
        ln(prior)
        
    """
    if cmin <= xd[1] and xd[1] < cmax and smin <= xd[2] and xd[2] < smax:
        return np.log(normfact/(1. + xd[0]**2)**1.5)
    else:
        return -100.

def line_fit_vert_like (x, params=None): 
    """
    likelihood for a linear model for data with error bars in both directions 
    and intrinsic scatter in y direction
    the merit function is also in y-direction (see d'Agostino 2005)
    Thus, this likelihood will resolve in different results when fit as y(x) or x(y)
    
    Parameters
    -----------
    x : vector of parameters: 
    x[0] : slope m; x[1] = intercept c; x[2]=intrinsic scatter
    params : [x, y, c00, c01, c11]            
    
    Returns
    -------
    numpy array
        likelihood values for each walker
    """
    p = params; nw = np.shape(x)[0]; res = np.zeros(nw)
    for i in range(nw):
        dummy = x[i,2] + p[4] + x[i,0]**2*p[2]
        res[i] = -0.5*(np.sum(np.log(dummy))+np.sum((p[1]-x[i,0]*p[0]-x[i,1])**2/dummy)) \
                + prior(x[i,:])
    return res

def line_fit_like(x, params=None):
    """
    likelihood describing model with Gaussian distribution perpendicular to the mean
    linear relation and data points with correlated Gaussian uncertainties in both x and y
    This likelihood should return identical results for fits of y(x) and 
    
    Parameters
    ----------
    x: numpy vector
     x[0] = slope; x[1] = normalization; x[2] = variance in the perp direction
    params: numpy vector
     params = [x, y, c00, c01, c11] 
     
    Returns
    -------
    numpy array
        likelihood values for each walker
        
    """
    p = params; nw = np.shape(x)[0]; res = np.zeros(nw)
    
    for i in range(nw):
        m2 = x[i,0] * x[i,0]
        r = 1. + m2
        sigtot2  = x[i,2]*r + p[2]*m2 - 2.*p[3]*x[i,0] + p[4]
        d2 = (p[1] - x[i,0]*p[0] - x[i,1])**2 # y-mx-s
        siginv1 = r / (2.0*np.pi*sigtot2)
        res[i] = np.sum(0.5*(np.log(siginv1) - d2/sigtot2)) + \
                        prior(x[i,:])

    return res

def mcmc_fit(x, y, ex, ey, pini=None, ipar_active=None, nwalkers=None, modelpdf=None):
    # covariance matrix of errors
    c00 = ex*ex; c01 = 0.; c11 = ey*ey
    p = pini; params = [x, y, c00, c01, c11]
    nparams = 3
    x0 = np.array(p[0:nparams]); step = 0.1*np.array(np.abs(p[0:nparams]));
    iz = (step == 0); step[iz] = 0.01 
    # define which parameters are active (=1), and which should stay fixed (=0)
    
    # initialize MCMC walkers
    xwalk = mcmc_sample_init(nparams=nparams, nwalkers=nwalkers, x0=x0, step=step, ipar_active=ipar_active)
    
    # run the sampler
    nRval = 500 # record Gelman-Rubin R indicator each nRval'th step
    # now get the chain and how many values to thin based on the final auto-correlation time
    chain, Rval, nthin = mcmc_sample(xwalk, nparams=nparams, nwalkers=nwalkers, nRval=nRval, 
                                    modelpdf=modelpdf, Rlim = 1.01, params=params, ipar_active=ipar_active)
    return chain, Rval, nthin

def pstats(x):
    """
    print chain mean, median, and confidence intervals
    """
    xmed = np.median(x); xm = np.mean(x); xsd = np.std(x)
    xcfl11 = np.percentile(x,16); xcfl12 = np.percentile(x,84)
    xcfl21 = np.percentile(x,2.5); xcfl22 = np.percentile(x,97.5)
    dashedline = '----------------------------------------------'
    print('mean, median = %.3f, %.3f, st.dev=%.4f'%(xm, xmed, xsd))
    print('68perc interval = %.3f, %.3f'%(xcfl11,xcfl12))
    print('95perc interval = %.3f, %.3f'%(xcfl21,xcfl22))
    print('%s'%dashedline)
    
def run_fit(x, ex, y, ey, nwalkers=100, m0=None, c0=None, s0= None, 
            ipar_active=None, modelpdf=line_fit_like):
    """
    run MCMC sampling of the posterior and extract fit parameters from the chain
    """

    chain, Rval, nthin = mcmc_fit(x, y, ex, ey, pini=[m0, c0, s0], 
                                  ipar_active=ipar_active, nwalkers=nwalkers, 
                                  modelpdf=modelpdf)

    nburn = int(20*nwalkers*nthin)
    if ipar_active[0]==1:
        m = zip(*chain)[0]; c = zip(*chain)[1]; s = np.sqrt(zip(*chain)[2])
        mc = m[nburn:]; cc = c[nburn:]; sc = s[nburn:]
        
    elif ipar_active[0]==0:
        m= m0; c = zip(*chain)[0]; s = np.sqrt(zip(*chain)[1])
        mc= m; cc = c[nburn:]; sc = s[nburn:]
    return mc, cc, sc

def plot_2d_dist(x, y, xlim, ylim, nxbins, nybins, weights=None, xlabel='x', ylabel='y', 
                 clevs=None, smooth=None, fig_setup=None, savefig=None):
    """
    routine to grid and plot a 2d histogram representing distribution of points with input coordinates x and y
    along with contour levels enclosing a given percentage of points specified as input. 
    
    x, y = float numpy arrays with input x and y positions of the points
    nxbins, nybins = int number of histogram bins in x and y directions
    weights = float weights to use for different points in the histogram
    xlabel, ylabel = string labels for x and y axis
    clevs = float contour levels to plot
    smooth = Boolean optional smoothing with Wiener filter is applied if True
    fig_setup = optional, this variable can pass matplotlib axes to this routine, if it is used within another plotting environment
    savefig = string, if specified the figure is saved in a file given by the path in the string
    
    Authors: Andrey Kravtsov and Vadim Semenov
    """
    if fig_setup == None:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        #ax = plt.add_subplot(1,1,1)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
    else:
        ax = fig_setup
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
    
    if xlim[1] < 0.: ax.invert_xaxis()

    if weights == None: weights = np.ones_like(x)
    H, xbins, ybins = np.histogram2d(x, y, weights=weights, bins=(np.linspace(xlim[0], xlim[1], nxbins),np.linspace(ylim[0], ylim[1], nybins)))
    
    H = np.rot90(H); H = np.flipud(H); 
             
    X,Y = np.meshgrid(xbins,ybins) 
    if smooth != None:
        from scipy.signal import wiener
        H = wiener(H, mysize=2)

    H = H/np.sum(H)        
    Hmask = np.ma.masked_where(H==0,H)
    
    pcol = ax.pcolormesh(X,Y,(Hmask), vmin=1.e-4*np.max(Hmask), cmap=plt.cm.BuPu, norm = LogNorm(), linewidth=0., rasterized=True)
    pcol.set_edgecolor('face')

    if clevs != None:
        lvls = []
        for cld in clevs:  
            sig = opt.brentq( conf_interval, 0., 1., args=(H,cld) )   
            lvls.append(sig)
                   
        ax.contour(H, linewidths=(1.0,0.75, 0.5, 0.25), colors='black', levels = lvls, 
                    norm = LogNorm(), extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]])
    if savefig:
        plt.savefig(savefig,bbox_inches='tight')
    if fig_setup == None:
        plt.show()
    return

clevs = (0.683, 0.955, 0.997) # standard contour levels

def plot_fit(m, c, s, mlim=None, clim=None, slim=None, p_stats = False):
    """
    plot posterior and some of its statistics
    
    """
    fig, ax = plt.subplots(1,2, figsize=(4.,2.))
    
    plt.tight_layout(); plt.rc('font',size=9)
    fig.subplots_adjust(hspace=1.7)

    plot_2d_dist(m, c, xlim=mlim, ylim=clim, nxbins=41, nybins=41, clevs=clevs[::-1], 
                 smooth=True, xlabel=r'$\mathrm{slope}$', ylabel=r'$\mathrm{normalization}$', fig_setup=ax[0])

    ax[1].yaxis.set_label_position('right')
    plot_2d_dist(m, s, xlim=mlim, ylim=slim, nxbins=41, nybins=41, clevs=clevs[::-1], smooth=True,
                 xlabel=r'$\mathrm{slope}$', ylabel=r'$\mathrm{scatter}$', fig_setup=ax[1])
    plt.show()
    if p_stats:
        print("%s"%"====== Bradford et al. BTFR sample perpendicular likelihood ======")
        print("%s"%"best fit slope:")
        pstats(m)
        print("%s"%"best fit normalization")
        pstats(c)
        print("%s"%"best fit scatter:")
        pstats(s)
        
    return
