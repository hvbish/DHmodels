import numpy as np
import pandas as pd
from astropy.io import fits
import emcee, corner,time
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.stats import norm, chisquare
from sklearn.metrics import r2_score, mean_squared_error
import os, shutil
from datetime import datetime
from DHmodels import *


def lnprior(pars):

    ##### FlatSandwich priors ######
    if densmod == FlatSandwich_Density:
        vflat, lag, vz, h0 = pars
        if vflat<0 or lag<0 or h0<0:
            return -np.inf
        # Gaussian priors (pdfs)
        h_prior   = norm.pdf(h0,loc=5,scale=2)
        lag_prior = norm.pdf(lag,loc=10,scale=2)
        vf_prior  = norm.pdf(vflat,loc=240,scale=20)
        vz_prior  = norm.pdf(vz,loc=0,scale=10)
        prior = np.log(h_prior*lag_prior*vz_prior*vf_prior)
        # Flat priors
        #pr = (0.0001<h0<10) and (0<lag<200) and (200<vflat<280) and (-20<vz<20)
        #prior = 0 if pr else -np.inf
    
    ##### GaussianSandwich priors ######
    elif densmod == GaussianSandwich_Density:
        vflat, lag, vz, h0, sigma = pars
        if vflat<0 or lag<0 or h0<0 or sigma<0:
            return -np.inf
        # Gaussian priors (pdfs)
        vf_prior  = norm.pdf(vflat,loc=240,scale=20)
        lag_prior = norm.pdf(lag,loc=10,scale=2)
        vz_prior  = norm.pdf(vz,loc=0,scale=10)
        h_prior   = norm.pdf(h0,loc=5,scale=2)
        sig_prior = norm.pdf(sigma,loc=1,scale=0.5)
        prior = np.log(h_prior*lag_prior*vz_prior*vf_prior*sig_prior)

    ##### ThickSandwich priors ######
    elif densmod == ThickSandwich_Density:
        vflat, lag, vz, hmin, hmax = pars
        if vflat<0 or lag<0 or hmin<0 or hmax<0 or hmin>hmax:
            return -np.inf
        # Gaussian priors (pdfs)
        vf_prior  = norm.pdf(vflat,loc=240,scale=20)
        lag_prior = norm.pdf(lag,loc=10,scale=2)
        vz_prior  = norm.pdf(vz,loc=0,scale=10)
        hmin_prior   = norm.pdf(hmin,loc=3,scale=1)
        hmax_prior = norm.pdf(hmax,loc=5,scale=1)
        prior = np.log(hmin_prior*hmax_prior*lag_prior*vz_prior*vf_prior)

    ##### RadialVerticalExponential priors ######
    elif densmod == RadialVerticalExponential_Density:
        vflat, lag, vz, R0, h0 = pars
        if vflat<0 or lag<0 or R0<0 or h0<0:
            return -np.inf
        # Gaussian priors (pdfs)
        vf_prior  = norm.pdf(vflat,loc=240,scale=20)
        lag_prior = norm.pdf(lag,loc=10,scale=2)
        vz_prior  = norm.pdf(vz,loc=0,scale=10)
        R_prior = norm.pdf(R0,loc=5,scale=1) # HB 10/8/21: Is this reasonable?
        h_prior   = norm.pdf(h0,loc=3,scale=2)
        prior = np.log(h_prior*lag_prior*vz_prior*vf_prior*R_prior)

    ##### VerticalExponential priors ######
    elif densmod == VerticalExponential_Density:
        vflat, lag, vz, h0 = pars
        if vflat<0 or lag<0 or h0<0:
            return -np.inf
        # Gaussian priors (pdfs)
        vf_prior  = norm.pdf(vflat,loc=240,scale=20)
        lag_prior = norm.pdf(lag,loc=10,scale=2)
        vz_prior  = norm.pdf(vz,loc=0,scale=10)
        h_prior   = norm.pdf(h0,loc=3,scale=2)
        prior = np.log(h_prior*lag_prior*vz_prior*vf_prior)

    ##### Constant priors ######
    elif densmod == Constant_Density:
        vflat, lag, vz = pars
        if vflat<0 or lag<0:
            return -np.inf
        # Constant priors (pdfs)
        vf_prior  = norm.pdf(vflat,loc=240,scale=20)
        lag_prior = norm.pdf(lag,loc=10,scale=2)
        vz_prior  = norm.pdf(vz,loc=0,scale=10)
        prior = np.log(lag_prior*vz_prior*vf_prior)
    
    else:
        raise NameError('WARNING: invalid density model in lnprior!')

    return prior


def lnlike(pars,data):
    
    ##### FlatSandwich parameters ######
    if densmod == FlatSandwich_Density:
        vf, lag, vz, h0 = pars
        velopars = (vf,lag,0.,vz)
        denspars = (1E-05,h0)
    
    ##### GaussianSandwich parameters ######
    elif densmod == GaussianSandwich_Density:
        vf, lag, vz, h0, sigma = pars
        velopars = (vf,lag,0.,vz)
        denspars = (1E-05,h0,sigma)

    ##### ThickSandwich parameters ######
    elif densmod == ThickSandwich_Density:
        vf, lag, vz, hmin, hmax = pars
        velopars = (vf,lag,0.,vz)
        denspars = (1E-05,hmin,hmax)

    ##### RadialVerticalExponential parameters ######
    elif densmod == RadialVerticalExponential_Density:
        vf, lag, vz, R0, h0 = pars
        velopars = (vf,lag,0.,vz)
        denspars = (1E-05, R0, h0)

    ##### VerticalExponential parameters ######
    elif densmod == VerticalExponential_Density:
        vf, lag, vz, h0 = pars
        velopars = (vf,lag,0.,vz)
        denspars = (1E-05, h0)

    ##### Constant parameters ######
    elif densmod == Constant_Density:
        vf, lag, vz = pars # HB 10/8/21: Is this correct?
        velopars = (vf,lag,0.,vz)
        denspars = (1E-05)

    else:
        raise NameError('WARNING: invalid density model in lnlike!')


    # Calculating the model
    mod = kinematic_model(data.lon,data.lat,velopars=velopars,densmodel=densmod,\
                          denspars=denspars,useC=False,nthreads=4)
    diff = np.nansum((mod.vlsr-data.vlsr)**2)
    return -diff


def lnprob(pars,data):
    lp = lnprior(pars)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(pars,data)



if __name__ == '__main__':

    # Choose density model
    densmod = FlatSandwich_Density 
    # densmod = GaussianSandwich_Density
    # densmod = ThickSandwich_Density 
    # densmod = RadialVerticalExponential_Density 
    # densmod = VerticalExponential_Density 
    # densmod = Constant_Density 

    run_in_parallel = True # If False, code will use a single thread (takes much longer)
    lat_lim = 60 # Only fit sightlines below this Galactic latitude

    HVC_flag = '3' # Choose which HVC flag we want

    for ion in ['CIV', 'SiIV', 'CII*', 'SiII', 'SII', 'FeII', 'NiII', 'NV']:
        
        ###########################################################################
        # FlatSandwich
        if densmod == FlatSandwich_Density:
            model_name = "Flat Sandwich"
            p0     = [230, 15, -5, 1]               # Initial guesses
            labels = ["vflat", "lag","vz", "h0"]    # Names of parameters to fit

        # GaussianSandwich 
        elif densmod == GaussianSandwich_Density:
            model_name = "Gaussian Sandwich"
            p0 = [230, 15, -5., 5., 0.5]
            labels = ["vflat", "lag", "vz", "h0","sigma"]

        # ThickSandwich 
        elif densmod == ThickSandwich_Density:
            model_name = "Thick Sandwich"
            p0 = [230, 15, -5., 3., 5.]
            labels = ["vflat", "lag", "vz", "hmin","hmax"]

        # RadialVerticalExponential 
        elif densmod == RadialVerticalExponential_Density:
            model_name = "Radial Vertical Exponential"
            p0 = [230, 15, -5., 3., 5.] 
            labels = ["vflat", "lag", "vz", "R0","h0"]

        # VerticalExponential 
        elif densmod == VerticalExponential_Density:
            model_name = "Vertical Exponential"
            p0 = [230, 15, -5., 5.] 
            labels = ["vflat", "lag", "vz", "h0"]

        # Constant 
        elif densmod == Constant_Density:
            model_name = "Constant"
            p0 = [230, 15, -5.] 
            labels = ["vflat", "lag", "vz"]

        else:
            raise NameError('WARNING: invalid density model for initial guesses!')
        ###########################################################################

        # Create directory structure to save output
        dir = './runs/hvc_flag_' + HVC_flag + '/' + densmod.__name__.split("_")[0] + '/'
        # shutil.rmtree(dir, ignore_errors=True) # Remove directory if it already exists
        os.makedirs(dir, exist_ok=True)
        
        # Print info about the model being run
        print ("Running " + densmod.__name__ + " model...")
        
        # Reading in sightlines
        ds = pd.read_table("data/sightlines_flag_" + HVC_flag + ".txt", sep=' ', skipinitialspace=True)

        # We select only the ion we have chosen to fit
        di = ds[ds['ion']==ion]
        # We select only latitudes below 60 deg for the fit
        glon, glat, vl = di['Glon'].values, di['Glat'].values, di['weighted_v_LSR'].values
        glon[glon>180] -= 360
        m = (np.abs(glat)<lat_lim)
        # Just storing the data in a Sightline object for convenience
        data = Sightlines()
        data.add_sightlines(glon[m],glat[m],vl[m],None,None)

        print ( "               Ion: " + ion)
        print (f" Sightlines to fit: {len(data.lon)}")
        print ( "          HVC mask: " + HVC_flag)
        print (f"Start time: {datetime.now().strftime('%H:%M:%S')}")

        nll = lambda *args: -lnprob(*args)
        
        # This is just to minimize the likelihood in a classical way
        soln = minimize(nll, p0, args=(data),method='Nelder-Mead')
        print ("Best-fit parameters:", soln.x)
        
        # Initializing chains and walkers
        ndim, nwalkers, nsteps = len(p0), 500, 500
        pos = p0 + 1e-1*np.random.randn(nwalkers,ndim)

        print ("\n Running MCMC...")
        if run_in_parallel:
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[data], pool=pool)
                start = time.time()
                res = sampler.run_mcmc(pos, nsteps, progress=True)
                multi_time = time.time()-start
                print("Computational time {0:.1f} minutes".format(multi_time/60.))
        else:
            print('Running in serial. This may take a while.')
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[data])
            start = time.time()
            res = sampler.run_mcmc(pos, nsteps, progress=True)
            multi_time = time.time()-start
            print("Computational time {0:.1f} minutes".format(multi_time/60.))

        # Saving samples
        fits.writeto(f"{dir}{ion}_samples.fits",np.float32(sampler.get_chain()),overwrite=True)

        # Plotting chains 
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        fig.savefig(f"{dir}/{ion}_chains.png")

        # Burn-in
        burnin = 100
        thin = 1
        samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

        # Output parameters to terminal and a text file
        print ("\n MCMC parameters:")
        pp = []
        with open(f"{dir}/params_" + densmod.__name__.split("_")[0] + f"_{ion}.txt",'w') as paramfile:
            for i in range(ndim):
                mcmc = np.percentile(samples[:, i], [15.865, 50, 84.135])
                q = np.diff(mcmc)
                txt = "%10s = %10.3f %+10.3f %+10.3f"%(labels[i],mcmc[1], -q[0], q[1])
                print (txt) # Output to Terminal
                paramfile.write(txt.split("=")[0]+txt.split("=")[1]+'\n') # Write to file
                pp.append(mcmc[1])


        # Autocorrelation function
        #tau = sampler.get_autocorr_time(quiet=True)
        #burnin = int(2*np.nanmax(tau))
        #thin = int(0.5*np.nanmin(tau))
        #print(tau,burnin,thin)

        # Save corner plot
        # Levels 
        levels = 1.0-np.exp(-0.5*np.arange(0.5, 2.1, 3.5) ** 2)
        #levels = 1.0-np.exp(-0.5*np.array([1., 2., 3.]) ** 2)

        fig = corner.corner(samples, truths=pp, labels=labels, show_titles=True, title_kwargs={"fontsize": lsize},\
                    truth_color='firebrick') #,fill_contours=True,levels=levels)
        fig.savefig(f"{dir}/{ion}_corner.pdf",bbox_inches='tight')

        # Plot model vs data
        ###########################################################################
        # FlatSandwich
        if densmod == FlatSandwich_Density:
            model = kinematic_model(data.lon,data.lat,velopars=(pp[0],pp[1],0,pp[2]),densmodel=FlatSandwich_Density,\
                                    denspars=(1E-08,pp[3]),useC=True,nthreads=8,getSpectra=False)

        # GaussianSandwich
        elif densmod == GaussianSandwich_Density:
            model = kinematic_model(data.lon,data.lat,velopars=(pp[0],pp[1],0,pp[2]),densmodel=GaussianSandwich_Density,\
                                    denspars=(1E-08,pp[3],pp[4]),useC=True,nthreads=8)

        # ThickSandwich
        elif densmod == ThickSandwich_Density:
            model = kinematic_model(data.lon,data.lat,velopars=(pp[0],pp[1],0,pp[2]),densmodel=ThickSandwich_Density,\
                                    denspars=(1E-08,pp[3],pp[4]),useC=True,nthreads=8)

        # RadialVerticalExponential
        elif densmod == RadialVerticalExponential_Density:
            model = kinematic_model(data.lon,data.lat,velopars=(pp[0],pp[1],0,pp[2]),densmodel=RadialVerticalExponential_Density,\
                                    denspars=(1E-08,pp[3],pp[4]),useC=True,nthreads=8)

        # VerticalExponential
        elif densmod == VerticalExponential_Density:
            model = kinematic_model(data.lon,data.lat,velopars=(pp[0],pp[1],0,pp[2]),densmodel=VerticalExponential_Density,\
                                    denspars=(1E-08,pp[3]),useC=True,nthreads=8)

        # Constant
        elif densmod == Constant_Density:
            model = kinematic_model(data.lon,data.lat,velopars=(pp[0],pp[1],0,pp[2]),densmodel=Constant_Density,\
                                    denspars=(1E-08),useC=False,nthreads=8)
        ###########################################################################
        
        # Plot data vs. model and residual skymaps
        fig, ax = plot_datavsmodel(data,model)
        fig.savefig(f"{dir}/{ion}_comp.pdf",bbox_inches='tight')
        fig, ax = plot_residuals(data,model,model_name)
        fig.savefig(f"{dir}/{ion}_residuals_mollweide.pdf",bbox_inches='tight')

        # Calculate goodness of fit
        try:
            r_squared = r2_score(data.vlsr, model.vlsr, sample_weight=None, multioutput='uniform_average')
        except:
            r_squared = -999
        try:
            # Mean squared error
            MSE = mean_squared_error(data.vlsr, model.vlsr)# Mean squared error
            RMS = np.sign(MSE)*np.sqrt(abs(MSE))
        except:
            RMS = 999
        print (" R-squared = ",round(r_squared,4))
        print (" RMS error = ",round(RMS,4),'\n\n')
        with open(f"{dir}/params_" + densmod.__name__.split("_")[0] + f"_{ion}.txt",'a') as paramfile:
            rsqtxt = "%10s %10.3f %+10s %+10s"%('R_squared',r_squared, -999, -999)
            rmstxt = "%10s %10.3f %+10s %+10s"%('RMS',     RMS,       -999, -999)
            paramfile.write(rsqtxt + '\n')
            paramfile.write(rmstxt + '\n')

        # Output data & model values to a text file
        with open(f"{dir}/"  f"{ion}_" + "data-model-vels_" + densmod.__name__.split("_")[0] + ".txt",'w') as paramfile:
            print ("data_lon", "\t", "data_lat", "\t", "data_v", "\t", "model_v", file=paramfile) # Write header
            for d_lon, d_lat, d_v, m_v in zip(data.lon, data.lat, data.vlsr, model.vlsr):
                print("{: 5.4f} \t {: 5.4f} \t {: 5.4f} \t {: 5.4f}".format(d_lon, d_lat, d_v, m_v), file=paramfile)
