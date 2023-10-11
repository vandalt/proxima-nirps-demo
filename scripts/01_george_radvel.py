# %% [markdown]
# # Activity and RV modelling with george, celerite, and RadVel

# %%
import pprint

import arviz as az
import corner
import emcee
import george
import matplotlib.pyplot as plt
import numpy as np
import radvel
import scipy.optimize as op
from astropy.table import Table
from george import kernels
from george.gp import LinAlgError
from radvel.plot import mcmc_plots, orbit_plots

# %% [markdown]
# ## Loading the data
# We'll use the full (non-binned) preprocessed lbl table produced by `00_quicklook.py`
#
# We'll use the "rjd" for time and remove an extra offset to have time close to 0.
# The median is already subtracted from the RVs.
# We'll use `DTEMP` as an activity indicator, but feel free to change this to another
# quantity to explore how it affects the fit.

# %%
tbl = Table.read("./data/lbl_PROXIMA_PROXIMA_preprocessed.rdb")

rjd_bjd_off = 2457000
extra_off_factor = 100
extra_off = np.floor(tbl["rjd"].min() / extra_off_factor) * extra_off_factor
t_off = rjd_bjd_off + extra_off

tlabel = "t"
tbl[tlabel] = tbl["rjd"] - extra_off

t = tbl[tlabel].data
t_mod = np.linspace(t.min(), t.max(), num=1000)
vrad = tbl["vrad"].data
svrad = tbl["svrad"].data
act = tbl["DTEMP"].data
sact = tbl["sDTEMP"].data
act -= np.median(act)

# %% [markdown]
# ## Activity modelling

# %%
plt.figure(figsize=(10, 6))
plt.errorbar(t, act, yerr=sact, fmt="k.")
plt.ylabel("$\Delta$T [K]")
plt.xlabel(f"Time [BJD - {t_off:.0f}]")
plt.show()

# %% [markdown]
# ### Defining a GP model
# We'll define our first GP model with `george`. The package is in maintenance mode and
# there are more efficient and modern alternatives (including some by the same authors),
# but it is still extensively used and provides a simple interface to understand how GPs
# work in practice.
#
# We first need to define a covariance function. We'll use a quasi-periodic kernel made
# of a square exponential and a periodic kernel. This is one of the first kernels that
# were used for this type of work.

# %%
amp = np.std(act)
length_scale = 100.0
gamma = 1.0
mean_val = np.mean(act)
log_wn_var = np.log(0.1**2)
log_period = np.log(91.0)
ker_sqexp = kernels.ExpSquaredKernel(metric=length_scale**2)  # note the **2 here
ker_per = kernels.ExpSine2Kernel(gamma=gamma, log_period=log_period)
kernel = amp**2 * ker_sqexp * ker_per

# %% [markdown]
# Now, we define a GP object which will take the kernel function as an input.
# We don't expect a "mean" model in the activity, so we'll just use a constant here.
# Also, the `white_noise` argument enables specifying an extra noise term applied to
# the diagonal of the GP (to inflate the error bars in a constant manner, note we could
# also add this to the errors directly in a separate likelihood function if we wanted).

# %%
gp = george.GP(
    kernel,
    mean=mean_val,
    fit_mean=True,
    white_noise=log_wn_var,
    fit_white_noise=True,
    # fit_white_noise=False,
)
# This pre-compute the covariance matrix
gp.compute(t, yerr=sact)
# We can then compute the marginal GP likelihood based on our data
print(gp.log_likelihood(act))
pprint.pprint(gp.get_parameter_dict())
parameter_guess = np.array(
    [
        mean_val,
        log_wn_var,
        amp,
        np.log(length_scale**2),
        gamma,
        log_period,
    ]
)

# %%
mu, var = gp.predict(act, t_mod, return_var=True)
std = np.sqrt(var)

# %%
plt.figure(figsize=(10, 6))
plt.errorbar(t, act, yerr=sact, fmt="k.")
plt.plot(t_mod, mu)
plt.fill_between(t_mod, mu - std, mu + std, alpha=0.2, color="C0")
plt.ylabel("$\Delta$T [K]")
plt.xlabel(f"Time [BJD - {t_off:.0f}]")
plt.show()

# %% [markdown]
# Not bad for a first guess! Remember: the GP adapts its fit based on the data and the
# hyperparameters.
#
# Try varying the guessed hyperparameter values to see how they affect the fit.
# What does lambda do? What about gamma?
#
# ### Hyperparameter optimization
# We can now optimize the hyperparameters with scipy to get the best-fit model.


# %%
def nll(p: np.ndarray[float]) -> float:
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(act, quiet=True)
    return -ll if np.isfinite(ll) else 1e25


gp.compute(t, sact)

print(f"Initial log-likelihood: {gp.log_likelihood(act)}")
print("Initial parameters:")
pprint.pprint(gp.get_parameter_dict())

# %%
p0 = gp.get_parameter_vector()
results = op.minimize(nll, p0, method="Powell")

# %%
results

# %%
gp.set_parameter_vector(results.x)
print(f"Final log-likelihood: {gp.log_likelihood(act)}")
print("Final parameters:")
pprint.pprint(gp.get_parameter_dict())

# %%
mu, var = gp.predict(act, t_mod, return_var=True)
std = np.sqrt(var)

# %%
plt.figure(figsize=(10, 6))
plt.errorbar(t, act, yerr=sact, fmt="k.")
plt.plot(t_mod, mu)
plt.fill_between(t_mod, mu - std, mu + std, alpha=0.2, color="C0")
plt.ylabel("$\Delta$T [K]")
plt.xlabel(f"Time [BJD - {t_off:.0f}]")
plt.show()

# %% [markdown]
# Note how the gamma parameter is larger than our initial guess for the best fit.
# This means the likelihood is improved by giving more flexibility to the model
# to handle short term variations. There is still a periodicity around 90 days.
#
# ### Hyperparameter sampling
# Here, we'll use `emcee` to sample the posterior distribution of our simple model.
# We could also use `george` with a nested sampling package, but it would require
# implementing a prior transform, which I won't get into here.
#
# First, we'll need to define our prior distribution.
# I'll print the order of parameters in the next cell for convenience,
# then define distributions and the  model prior.

# %%
pprint.pprint(gp.get_parameter_dict())


# %%
def gaussian_logp(x: float, mu: float, sigma: float) -> float:
    # Copied from radvel for convenience
    return -0.5 * ((x - mu) / sigma) ** 2 - 0.5 * np.log((sigma**2) * 2.0 * np.pi)


def uniform_logp(x: float, minval: float, maxval: float) -> float:
    # Copied from radvel for convenience
    if x <= minval or x >= maxval:
        return -np.inf
    else:
        return -np.log(maxval - minval)


def jeffreys_logp(x: float, minval: float, maxval: float) -> float:
    # Copied from radvel for convenience
    normalization = 1.0 / np.log(maxval / minval)
    if x < minval or x > maxval:
        return -np.inf
    else:
        return np.log(normalization) - np.log(x)


def mod_jeffreys_logp(x: float, minval: float, maxval: float, kneeval: float) -> float:
    normalization = 1.0 / np.log((maxval - kneeval) / (minval - kneeval))
    if (x > maxval) or (x < minval):
        return -np.inf
    else:
        return np.log(normalization) - np.log(x - kneeval)


def log_prior(p: np.ndarray) -> float:
    log_prob = 0.0
    # Mean with wide prior around 0
    log_prob += gaussian_logp(p[0], 0.0, 5.0)
    # Log White noise: Uniform
    log_prob += uniform_logp(p[1], -5.0, 5.0)
    # Log Variance: Uniform
    log_prob += uniform_logp(p[2], -10.0, 10.0)
    # Log metric (lambda**2): Uniform
    log_prob += uniform_logp(p[3], 6.0, 20.0)
    # Gamma: Jeffreys prior
    log_prob += jeffreys_logp(p[4], 1.0, np.exp(10.0))
    # Log Period: Uniform
    log_prob += uniform_logp(p[5], np.log(60), np.log(120))

    return log_prob


def log_post(p: np.ndarray) -> float:
    log_prob = log_prior(p)
    if np.isfinite(log_prob):
        gp.set_parameter_vector(p)
        return log_prob + gp.log_likelihood(act, quiet=True)
    return -np.inf


# %% [markdown]
# Now that we have defined our probabilistic model, we can do inference on the
# parameters.
#
# First, let's do a quick check that our prior is what we expect and gives models that
# look OK. This is called a prior predictive check.
#
# We'll use corner and Arviz to visualize our posterior.

# %%
nwalkers, ndim = 32, len(gp)
num_warmup = 200
num_prior_samples = 10_000
sampler_prior = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    log_prior,
)

p0 = gp.get_parameter_vector() + 1e-4 * np.random.randn(nwalkers, ndim)

# %%
print("Running prior predictive check")
sampler_prior.run_mcmc(p0, num_prior_samples + num_warmup, progress=True)
print("Done")

# %%
prior_samples = sampler_prior.get_chain(discard=num_warmup)

# %%
converted_prior_samples = {
    gp.get_parameter_names()[i]: np.swapaxes(prior_samples[..., i], 0, 1)
    for i in range(len(gp))
}
plt.show()
prior_inf_data = az.from_dict(converted_prior_samples)

# %%
corner.corner(prior_inf_data)
plt.show()

# %%
az.plot_trace(prior_inf_data)
plt.show()

# %%
num_display = 50
maxiter = 100
niter = 0
model_prior_samples = []
while niter < maxiter and len(model_prior_samples) < num_display:
    w = np.random.randint(nwalkers)
    n = np.random.randint(num_prior_samples)
    gp.set_parameter_vector(prior_samples[n, w])
    try:
        model_prior_samples.append(gp.sample_conditional(act, t_mod))
    except LinAlgError:
        print("LinAlgError")
        print(f"iteration {niter}")
        print("GP hyperparameters:")
        pprint.pprint(gp.get_parameter_dict())
    niter += 1

# %%
model_prior_samples = np.array(model_prior_samples)

# %%
plt.figure(figsize=(10, 6))
plt.plot(t_mod, model_prior_samples.T, alpha=0.5)
plt.errorbar(t, act, yerr=sact, fmt="k.")
plt.ylabel("$\Delta$T [K]")
plt.xlabel(f"Time [BJD - {t_off:.0f}]")
plt.show()

# %% [markdown]
# The prior models don't all look great, but they widely encompass the data and they do
# not raise linear algebra errors.
#
# We can now sample the posterior distribution with `emcee`. I'm using the `DEMove()` to
# propose steps as it seems to help with the sampling in this case

# %%
nwalkers, ndim = 32, len(gp)
num_warmup = 600
num_samples = 10_000
sampler_act = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    log_post,
    # moves=[emcee.moves.StretchMove()]
    moves=[emcee.moves.DEMove()],
)

gp.compute(t, yerr=sact)

gp.set_parameter_vector(results.x)
p0 = gp.get_parameter_vector() + 1e-4 * np.random.randn(nwalkers, ndim)

# %%
print("Running sampling")
sampler_act.run_mcmc(p0, num_samples + num_warmup, progress=True)
print("Done")

# %%
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler_act.get_chain()
labels = gp.get_parameter_names()
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number")
plt.show()

# %% [markdown]
# Arviz can convert emcee posterior samples to an InferenceData object, which
# facilitates summarizing, analyzing and plotting the results, and it provides
# a common interface for multiple samplers. The next two lines extract samples
# from emcee and remove warm-up steps.

# %%
inf_data = az.from_emcee(sampler_act, var_names=gp.get_parameter_names())
inf_data = inf_data.sel(draw=slice(num_warmup, None))

# %%
inf_data.to_netcdf("inf_data_act.nc")

# %%
inf_data = az.from_netcdf("inf_data_act.nc")

# %%
az.summary(inf_data)

# %%
az.plot_trace(inf_data)
plt.show()

# %%
corner.corner(inf_data, show_titles=True, quantiles=[0.16, 0.5, 0.84])
plt.show()

# %% [markdown]
# Looking at the plots above, we see that this combination of GP kernel and
# hyperparameter might not be the best suited for this problem: the R-hat
# statistic is not very close to 1 and the chains appear to get stuck in some
# places. For now I will keep this model and continue with the RV modelling, but
# here are a few ideas to explore:
# - Try another sampler (nested sampling, No-U-Turn-Sampling with PyMC+exoplanet, etc.)
# - Explore emcee options (e.g. Moves) to see if it improves the sampling
# - Explore if other priors yield better/different results.
# - Try another kernel, either by changing the structure of your george model above or
#   by using celerite, celerite2, tinygp or another package.
#   - Celerite almost has the same interface as george, but supports a specific class
#     of kernels which provide more efficient calculation. Most kernels there are quasi-periodic which
#     is great for our use-case.
#   - `tinygp` and `celerite2` can be seen as more modern versions of `george` and `celerite`.
#     They support autodifferentiation frameworks (gradients "for free"!).
#
# ## RV Modelling
# Now that we have a model for the activity, we can use this as prior knowledge for the
# RV modelling.
#
# As mentioned above, we will use RadVel in this section: it is well tested, provides
# a simple interface, and supports GPs out of the box.
#
# The flexibility of the RadVel GP models is somewhat limited, but new kernels can
# technically be added by implementing custom classes.

# %%
pprint.pprint(radvel.gp.KERNELS)

# %%
# We need to declare parameters for our GP likelihood a few cells below
hnames = [
    "gp_amp",  # eta_1; GP variability amplitude
    "gp_explength",  # eta_2; GP non-periodic characteristic length
    "gp_per",  # eta_3; GP variability period
    "gp_perlength",  # eta_4; GP periodic characteristic length
]

# %% [markdown]
# ### Converting samples from the activity model
# The activity model we used above used the default George parametrization.
# In practice we would probably want to implement GP kernels with the same
# parameterization for activity indicators and RV. However, we can still
# use our activity model as a prior by converting posterior samples.
#
# The equation for the RadVel QP kernel can be found [in the docs](https://radvel.readthedocs.io/en/latest/gp.html).
# The same is true for [george kernels](https://george.readthedocs.io/en/latest/user/kernels/).
# From the equations there, we can derive posteriors corresponding to RadVel parameters. This
# is simple with Arviz InferenceData

# %%
inf_post = inf_data.posterior
inf_post["gp_explength"] = np.sqrt(2 * np.exp(inf_post["kernel:k1:k2:metric:log_M_0_0"]))
inf_post["gp_per"] = np.exp(inf_post["kernel:k2:log_period"])
inf_post["gp_perlength"] = np.sqrt(0.5 / inf_post["kernel:k2:gamma"])

# %%
corner.corner(
    inf_data,
    var_names=["gp_explength", "gp_per", "gp_perlength"],
    show_titles=True,
    quantiles=[0.16, 0.5, 0.84],
)
plt.show()

# %% [markdown]
# ### Defining the RV model
# A RadVel model contains the following components:
# - A set of parameters
# - A `Model` object defining the forward model
# - A `Likelihood` object (chi2 or GP)
# - A `Posterior`, which combines the likelihood with priors.
#
# Let us start by initializing the parameters

# %%
nplanets = 1
# The basis determines the orbit parametrization
params = radvel.Parameters(nplanets, basis="per tc secosw sesinw k")

# Orbit parameters
p_orb_guess = 12.0  # Not too far from real value
tc_guess = 202  # Selected to avoid phase-wrapping issues
params["per1"] = radvel.Parameter(value=p_orb_guess)
params["tc1"] = radvel.Parameter(value=tc_guess)
params["sesinw1"] = radvel.Parameter(value=0.0, vary=False)  # fix e = 0 for now
params["secosw1"] = radvel.Parameter(value=0.0, vary=False)
params["k1"] = radvel.Parameter(value=1.3)

# GP parameters
params["gp_amp"] = radvel.Parameter(value=25.0)
# Length scale parameters are derived from activity indicator
params["gp_explength"] = radvel.Parameter(
    value=inf_post.quantile(0.5)["gp_explength"].item()
)
params["gp_per"] = radvel.Parameter(value=inf_post.quantile(0.5)["gp_per"].item())
params["gp_perlength"] = radvel.Parameter(
    value=inf_post.quantile(0.5)["gp_perlength"].item()
)

# The RV model is built from the parameters, but adds
# a trend and a quadratic variation to the model. We fix  them to 0.
mod = radvel.RVModel(params)
mod.params["dvdt"] = radvel.Parameter(value=0.0, vary=False)
mod.params["curv"] = radvel.Parameter(value=0.0, vary=False)

# %% [markdown]
# When the model includes GPs, we use `GPLikelihood` instead of `Likelhood`.
# Here we combine the model, the data, and the GP info.
#
# The "jit" parameter is a white noise term. Gamma is the RV offset.

# %%
like = radvel.likelihood.GPLikelihood(
    mod, t, vrad, svrad, hnames, kernel_name="QuasiPer"
)
like.params["gamma"] = radvel.Parameter(value=0.0, vary=True)
like.params["jit"] = radvel.Parameter(value=1.0)

# %% [markdown]
# We can now create a posterior object from the likelihood and combine
# it with priors.
#
# For orbit parameters, we use weakly informative priors.
# For the GP length scales, however, we use gaussian distributions
# derived from the activity posteriors.
# Note that this is not ideal: we would want a more flexibile distribution
# in to capture non-gaussianity. The extreme case of this is to use a kernel
# density estimate as the prior (`NumericalPrior` in radvel). However this is
# a bit more computationally expensive and, in our case, there was a strong
# bimodality in the period posterior, which we might not want to force on the RV fit.

# %%
post = radvel.posterior.Posterior(like)

# %%
post.priors += [radvel.prior.HardBounds("per1", 10.0, 20.0)]
post.priors += [radvel.prior.HardBounds("tc1", tc_guess - 5, tc_guess + 5)]
post.priors += [radvel.prior.HardBounds("k1", 0.0, 5.0)]
post.priors += [radvel.prior.Gaussian("gamma", 0.0, 10.0)]
post.priors += [radvel.prior.Jeffreys("gp_amp", np.exp(-5), np.exp(5))]
post.priors += [radvel.prior.Jeffreys("jit", np.exp(-5), np.exp(5))]
derived_unc = np.mean(np.abs(inf_post.quantile([0.16, 0.84]) - inf_post.quantile(0.5)))
post.priors += [
    radvel.prior.Gaussian(
        "gp_explength",
        inf_post.quantile(0.5)["gp_explength"].item(),
        derived_unc["gp_explength"],
    )
]
post.priors += [
    radvel.prior.Gaussian(
        "gp_per", inf_post.quantile(0.5)["gp_per"].item(), derived_unc["gp_per"]
    )
]
# post.priors += [
#     radvel.prior.NumericalPrior(
#         "gp_per", inf_post.quantile(0.5)["gp_per"].item(), derived_unc["gp_per"]
#     )
# ]
post.priors += [
    radvel.prior.Gaussian(
        "gp_perlength",
        inf_post.quantile(0.5)["gp_perlength"].item(),
        derived_unc["gp_perlength"],
    )
]

# %%
print(post)

# %% [markdown]
# ### Parameter Optimization
# That's it! We have a working RadVel model.
# We can plot the model with its initial parameter values below.
# Then we will optimize the model with scipy and plot the updated result.

# %%
GPPlot = orbit_plots.GPMultipanelPlot(
    post,
    subtract_gp_mean_model=False,
    plot_likelihoods_separately=False,
    subtract_orbit_model=False,
)
GPPlot.plot_multipanel()
plt.show()

# %%
res = op.minimize(
    post.neglogprob_array,
    post.get_vary_params(),
    method="Powell",
    # options=dict(maxiter=200, maxfev=100000, xatol=1e-8)
)
print(res)

# %%
post.set_vary_params(res.x)
print(post)

# %%
GPPlot = orbit_plots.GPMultipanelPlot(
    post,
    subtract_gp_mean_model=False,
    plot_likelihoods_separately=False,
    subtract_orbit_model=False,
)
GPPlot.plot_multipanel()
plt.show()

# %% [markdown]
# ### Sampling the posterior
# Radvel includes a built-in MCMC sampler based on emcee.
# It is also possible to use `emcee` directly with the `post.logprob_array()` method.
# The built-in sampler is nice, but it is not as flexible as `emcee`.
#
# RadVel does not support nested sampling yet, but

# %%
chains = radvel.mcmc(
    post, nwalkers=32, nrun=10_000, ensembles=4, savename="rawchains.h5"
)

# %%
chains.to_csv("radvel_chains.csv")

# %%
Corner = mcmc_plots.CornerPlot(post, chains)  # posterior distributions
Corner.plot()

# %%
quants = chains.quantile(
    [0.159, 0.5, 0.841]
)  # median & 1sigma limits of posterior distributions
for par in post.params.keys():
    if post.params[par].vary:
        med = quants[par][0.5]
        high = quants[par][0.841] - med
        low = med - quants[par][0.159]
        err = np.mean([high, low])
        err = radvel.utils.round_sig(err)
        med, err, errhigh = radvel.utils.sigfig(med, err)
        print("{} : {} +/- {}".format(par, med, err))

# %% [markdown]
# We're done!
#
# Feel free to modify the code above. Here are a few ideas:
# - Try a different GP kernel (both in radvel and for the activity indicator).
# - Try building your own GP+planets plots instead of using the RadVel ones
# - Try different priors
# - Try a differnt orbit parametrization in RadVel
# - Test it on another dataset
#
# Ohter frameworks exist to perform the tasks above. Here are a few:
# - [Juliet](https://juliet.readthedocs.io/en/latest/): less easy to customize and play
#   with the model, but simpler interface and more samplers.
# - [exoplanet](https://docs.exoplanet.codes/en/latest/): built on top of PyMC3. This
#   makes creating flexible models very easy, PyMC computes gradients for your model
#   automatically. This enables using more advanced samplers such as Hamiltonian Monte
#   Carlo (HMC) easily.
# - [jaxoplanet](https://jax.exoplanet.codes/en/latest/): A version of `exoplanet`
#   built on top of `jax` and `numpyro`. These tools cover a similar niche to PyMC,
#   but the Jax interface is a bit more intuitive for Jax users, and it can run on GPUs.
#   However, `jaxoplanet` is still a WIP and it's not really documented yet.
# - Your own code! All of this is fairly simple to implement, you could try implementing
#   a code that does it from sratch, or use one that you already have. All the tools
#   I mentioend above are open source, so you can look at the code to understand
#   the details.
