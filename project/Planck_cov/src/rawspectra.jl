#
# ```@setup rawspectra
# # the example command line input for this script
# ARGS = ["example.toml", "P143hm1", "P143hm2"] 
# ``` 


# # Raw Spectra (rawspectra.jl)
# The first step in the pipeline is simply to compute the pseudo-spectrum between the 
# maps ``X`` and ``Y``. 
# We define the pseudo-spectrum ``\widetilde{C}_{\ell}`` as the 
# result of the estimator on spherical harmonic coefficients of the *masked* sky,
# ```math
# \widetilde{C}_{\ell} = \frac{1}{2\ell+1} \sum_m a^{i,X}_{\ell m} a^{j,Y}_{\ell m}.
# ```
# Since we mask the galaxy and point sources, this is a biased estimate of the underlying
# power spectrum. The mask couples modes together, and also removes power from parts of the 
# sky. This coupling is described by a linear operator ``\mathbf{M}``, the mode-coupling 
# matrix. 
# For more details on spectra and mode-coupling, please refer to the [documentation for 
# PowerSpectra.jl](https://xzackli.github.io/PowerSpectra.jl/dev/spectra/).
# If this matrix is known, then one can perform a linear solve to obtain an unbiased
# estimate of the underlying power spectrum ``C_{\ell}``,
# ```math
# \langle\widetilde{C}_{\ell}\rangle = 
# \mathbf{M}^{XY}(i,j)_{\ell_1 \ell_2} \langle {C}_{\ell} \rangle,
# ```
# 
# This script performs this linear solve, *without accounting for beams*. The noise spectra
# are estimated from the difference of auto- and cross-spectra, 
# The command-line syntax for using this component to compute mode-coupled spectra is 
# 
# ```@raw html
# <pre class="shell">
# <code class="language-shell hljs">$ julia rawspectra.jl global.toml [map1] [map2]</code></pre>
# ```
# `[map1]` and `[map2]` must be names of maps described in global.toml. 
#
#
# ## File Loading and Cleanup

# This 
# page shows the results of running the command
# ```@raw html
# <pre class="shell">
# <code class="language-shell hljs">$ julia rawspectra.jl example.toml P143hm1 P143hm2</code></pre>
# ```

# We start by loading the necessary packages.

using TOML
using Healpix
using PowerSpectra
using Plots

# The first step is just to unpack the command-line arguments, which consist of the 
# TOML config file and the map names, which we term channels 1 and 2.

configfile, mapid1, mapid2 = ARGS[1], ARGS[2], ARGS[3]
#
config = TOML.parsefile(configfile)

# Next, we check to see if we need to render plots for the Documentation.

if config["general"]["plot"] == false
    Plots.plot(args...; kwargs...) = nothing
    Plots.plot!(args...; kwargs...) = nothing
end

# For both input channels, we need to do some pre-processing steps.
# 1. We first load in the ``I``, ``Q``, and ``U`` Stokes vectors, which are FITS 
#    columns 1, 2, and 3 in the Planck map files. These must be converted from nested to ring
#    ordered, to perform SHTs.
# 2. We read in the corresponding masks in temperature and polarization.
# 3. We zero the missing pixels in the maps, and also zero the corresponding pixels in the 
#    masks.
# 4. We convert from ``\mathrm{K}_{\mathrm{CMB}}`` to ``\mu \mathrm{K}_{\mathrm{CMB}}``.
# 5. Apply a small polarization amplitude adjustment, listed as `poleff` in the config.
# 6. We also remove some noisy pixels with ``\mathrm{\sigma}(Q) > 10^6\,\mu\mathrm{K}^2`` 
#    or ``\mathrm{\sigma}(U) > 10^6\,\mu\mathrm{K}^2``, or if they are negative. This 
#    removes a handful of pixels in 
#    the 2018 maps at 100 GHz which interfere with covariance estimation.
# 7. Estimate and subtract the pseudo-monopole and pseudo-dipole.


"""Return (polarized map, maskT, maskP) given a config and map identifier"""
function load_maps_and_masks(config, mapid, maptype=Float64)
    ## read map file 
    mapfile = joinpath(config["dir"]["map"], config["map"][mapid])
    polmap = PolarizedMap(
        nest2ring(readMapFromFITS(mapfile, 1, maptype)),  # I
        nest2ring(readMapFromFITS(mapfile, 2, maptype)),  # Q
        nest2ring(readMapFromFITS(mapfile, 3, maptype)))  # U

    ## read maskT and maskP
    maskfileT = joinpath(config["dir"]["mask"], config["maskT"][mapid])
    maskfileP = joinpath(config["dir"]["mask"], config["maskP"][mapid])
    maskT = readMapFromFITS(maskfileT, 1, maptype)
    maskP = readMapFromFITS(maskfileP, 1, maptype)

    ## read Q and U pixel variances, and convert to μK
    covQQ = nest2ring(readMapFromFITS(mapfile, 8, maptype)) .* 1e12
    covUU = nest2ring(readMapFromFITS(mapfile, 10, maptype)) .* 1e12

    ## go from KCMB to μKCMB, and apply polarization factor
    poleff = config["poleff"][mapid]
    scale!(polmap, 1e6, 1e6 * poleff)  # apply 1e6 to (I) and 1e6 * poleff to (Q,U)

    ## identify missing pixels and also pixels with crazy variances
    missing_pix = (polmap.i .< -1.6e30)
    missing_pix .*= (covQQ .> 1e6) .| (covUU .> 1e6) .| (covQQ .< 0.0) .| (covUU .< 0.0)
    allowed = (~).(missing_pix)

    ## apply the missing pixels to the map and mask for T/P
    mask!(polmap, allowed, allowed)
    mask!(maskT, allowed)
    mask!(maskP, allowed)

    ## fit and remove monopole/dipole in I
    monopole, dipole = fitdipole(polmap.i, maskT)
    subtract_monopole_dipole!(polmap.i, monopole, dipole)
    
    return polmap, maskT, maskP
end

# We'll use this function for the half-missions involved here.

m₁, maskT₁, maskP₁ = load_maps_and_masks(config, mapid1)
m₂, maskT₂, maskP₂ = load_maps_and_masks(config, mapid2)
plot(m₁.i, clim=(-200,200))  # plot the intensity map

#  
plot(maskT₁)  # show the temperature mask

#
# # Computing Spectra and Saving
#
# Once you have cleaned up maps and masks, you compute the 
# calculation is described in [PowerSpectra - Mode Coupling](https://xzackli.github.io/PowerSpectra.jl/dev/spectra/#Mode-Coupling-for-EE,-EB,-BB).
# That package has a utility function [`master`](https://xzackli.github.io/PowerSpectra.jl/dev/module_index/#PowerSpectra.master-Tuple{Healpix.PolarizedMap,%20Healpix.Map,%20Healpix.Map,%20Healpix.PolarizedMap,%20Healpix.Map,%20Healpix.Map})
# that performs the full MASTER calculation on two ``IQU`` maps with associated masks.
#

Cl = master(m₁, maskT₁, maskP₁,
            m₂, maskT₂, maskP₂)
nside = maskT₁.resolution.nside  # get the resolution from any of the maps
lmax = nside2lmax(nside)
print(keys(Cl))  # check what spectra were computed 

# The PowerSpectra.jl package has the Planck bestfit theory and beams as utility functions,  
# for demo and testing purposes. We can use it that for plotting here.

spec = :TE
bl = PowerSpectra.planck_beam_bl("100", "hm1", "100", "hm2", spec, spec; lmax=lmax)
ell = eachindex(bl)
prefactor = ell .* (ell .+ 1) ./ (2π)
plot( prefactor .*  Cl[spec] ./ bl.^2, label="\$D_{\\ell}\$", xlim=(0,2nside) )
theory = PowerSpectra.planck_theory_Dl()
plot!(theory[spec], label="theory $(spec)", ylim=(-200,300))

# Now we save our spectra. 

using CSV, DataFrames
run_name = config["general"]["name"]
spectrapath = joinpath(config["dir"]["scratch"], "rawspectra")
mkpath(spectrapath)

## assemble a table with the ells and spectra
df = DataFrame()
df[!,:ell] = ell
for spec in keys(Cl)
    df[!,spec] = parent(Cl[spec])
end

CSV.write(joinpath(spectrapath, "$(run_name)_$(mapid1)x$(mapid2).csv"), df)
