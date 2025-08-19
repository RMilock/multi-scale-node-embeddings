from plotting_functions import *
from utils import *

def _compute_hist2d(x, y, bins = 30):
    # obtain the 2D density of the pmatrix
    H, xedges, yedges = np.histogram2d(x, y, bins)

    # Histogram does not follow Cartesian convention (see numpy docs), transpose H for visualization purposes.
    H = H.T / x.size

    return H, xedges, yedges

def _inset_pmatrix(ax, H, xedges, yedges, width=2, height=2, bbox_to_anchor = (.67, .05), axis_scales = "linear", title = ""):    
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from dependencies import cmap

    # inset plot
    # Note: bbox_to_anchor = (x, y) fix the position of the upper left corner of the inset axis --> width and height will provide the size of the inset axis
    inax = inset_axes(ax, width=width, height=height, loc = "lower left", bbox_to_anchor = bbox_to_anchor, bbox_transform=ax.transAxes)
    
    im = inax.imshow(H , cmap = cmap, norm = "log", aspect = "auto", origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    # coords = ax.transAxes.inverted().transform(inax.get_tightbbox())
    # print(f'-coords: {coords}',)
    # border = 0#0.02
    # w, h = coords[1] - coords[0] + 2*border
    ax.add_patch(plt.Rectangle((.65,.02), 1, .45, fc="white", alpha = 0.7, 
                            transform=ax.transAxes, zorder=4))
    
    # if the pmatrix is very small, use linear scales
    # set the axis labels, scales and title
    inax.grid(False)
    inax.set(ylabel=r'$p_{sum}$', xlabel=r'$p_{fit}$',
                title=title, yscale = axis_scales, xscale = axis_scales)
    
    # set it externally as they are not working inside the inax.set
    inax.set_xticks([]) 
    inax.set_yticks([])

def _plot_inset_pmatrices(fig, ax, obs_net, sum_model, ref_model, level = None):
    """
    Create a new (squared) inset axis for the pmatrix plot
    On top of this, we have computed the abs error since for small values the relative error may lead to huge values spoiling the comparison
    """

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from plotting_functions import save_fig 

    # select the width of the inner axis, the markersize and the alpha
    width = 100
    ms, alpha = 1, 0.3

    triu_idx = np.triu_indices(sum_model.n_nodes, k = 1)
        # to have really good spaciated dashed line, do plot([min, max], [min, max], "--", linewidth = 0.3, color = "black")

    triu_ref_zl_pmatrix = ref_model.zl_pmatrix[triu_idx]
    triu_sum_zl_pmatrix = sum_model.zl_pmatrix[triu_idx]

    axis_scales = "linear"
    H, xedges, yedges = _compute_hist2d(x = triu_ref_zl_pmatrix, y = triu_sum_zl_pmatrix, bins = 30)

    _inset_pmatrix(ax, H, xedges, yedges, axis_scales = axis_scales)
    
    
    plot_full_path = f"{sum_model.plots_dir}/pmatrix/level{obs_net.level:g}.pdf"
    if not os.path.exists(plot_full_path):
        fig, axs = plt.subplots(1, 2, figsize = (12,5))
        fig.subplots_adjust(top = 0.8)

        # plot the 2D density of the pmatrix
        im = axs[0].imshow(H, cmap = cmap, norm = "log", aspect = "auto", origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        fig.colorbar(im)

        # set the axis as before
        axs[0].grid(False)
        axs[0].set(ylabel=f'Summed', xlabel=f'Fitted', title=r"Density of $p_{ij}$", yscale = axis_scales, xscale = axis_scales)

        # this is the relative error but it is not working properly as we have small values that are creating a huge error
        delta_p = sum_model.pmatrix - ref_model.pmatrix
        im = axs[1].imshow(delta_p, cmap='RdBu', aspect = "auto")
        
        im = axs[1].imshow(delta_p, cmap='RdBu', aspect = "auto", norm = "symlog", vmin = np.round(delta_p.min(),2), vmax = np.round(delta_p.max(),2))
        norm_delta_p = np.round(np.linalg.norm(np.triu(delta_p,1)),1)
        axs[1].set(yscale = axis_scales, xscale = axis_scales,)
        axs[1].set_title(f"(Sum - Fit) Pmatrices ~ {norm_delta_p}", fontdict = dict(fontsize = 13))
        axs[1].grid(False)
        fig.colorbar(im)

        # since LPCA has B and C, create a dim string
        # title_dimBCX = ""
        # if sum_model.name.endswith("LPCA"):
        #     title_dimBCX = f", dimB: {sum_model.dimB}, dimC: {sum_model.dimB}"
        # elif sum_model.name.endswith("maxlMSM"):
        #     title_dimBCX = f", dimX: {sum_model.dimX}"
        title_dimBCX = ""
        if ref_model.objective == "NodeEmb":
            title_dimBCX = f", dimB: {sum_model.dimB}, dimC: {sum_model.dimB}" if sum_model.name.endswith("LPCA") else f", dimX: {sum_model.dimX}"
        
        fig.suptitle(f"Summed VS Fitted pmatrix for {ref_model.plots_name} -- level {obs_net.level}{title_dimBCX}")
        save_fig(fig = fig, full_path = plot_full_path)
        plt.close()

def ax_scatter_errorbar(ax, meas3, meas4, meas5, meas6, err_meas3, err_meas4, err_meas5, err_meas6, ref_color, sum_color, ref_ms, sum_ms, ref_label, sum_label):
    """
    Dumb Function to switch among ax.scatter and ax.errorbar 
    as ax.errorbar generates always errorbars even if the error is 0
    Note: err_meas3/4/5/6 = [(lb, ub), ...] for every point
    """

    if np.array_equiv(err_meas4, 0):
        ax.scatter(meas3, meas4, marker = ref_model_marker, c = ref_color, s = ref_ms**2, label = ref_label)
        ax.scatter(meas5, meas6, marker = sum_model_marker, c = sum_color, s = sum_ms**2, label = sum_label)
    else:
        # for + error-bars, uncomment this (comment the other)
        alpha_conf_int = 0.5
        _, caps, bars = ax.errorbar(meas3, meas4, xerr = err_meas3, yerr = err_meas4, fmt = ref_model_marker, elinewidth = 1, capsize = 3, color = ref_color, markersize = ref_ms, label = ref_label)

        # set the bar and cap to an alpha =  alpha_conf_int
        [bar.set_alpha(alpha_conf_int) for bar in bars]
        [cap.set_alpha(alpha_conf_int) for cap in caps]
        
        _, caps, bars = ax.errorbar(meas5, meas6, xerr = err_meas5, yerr = err_meas6, fmt = sum_model_marker, elinewidth = 1, capsize = 3, color = sum_color, ms = 1.2*sum_model_ms, label = sum_label)

        # set the bar and cap to an alpha =  alpha_conf_int
        [bar.set_alpha(alpha_conf_int) for bar in bars]
        [cap.set_alpha(alpha_conf_int) for cap in caps]
        
        # # to use fill_between(...) one has to sort the x-axis before plotting
        # alpha_conf_int = 0.3
        # arg_sort_meas = np.argsort(meas3)
        # ax.scatter(meas3[arg_sort_meas], meas4[arg_sort_meas], marker = ref_model_marker, c = ref_color, s = ref_ms**2, label = ref_label)
        # err_lb = meas4 - err_meas4[0]
        # err_ub = meas4 + err_meas4[1]
        # ax.fill_between(meas3[arg_sort_meas], err_lb[arg_sort_meas], err_ub[arg_sort_meas], color = ref_color, alpha = alpha_conf_int)

        # # repeat the same procedure for the err_meas5
        # arg_sort_meas = np.argsort(meas5)
        # ax.scatter(meas5[arg_sort_meas], meas6[arg_sort_meas], marker = sum_model_marker, c = sum_color, s = sum_ms**2, label = ref_label)
        # err_lb = meas6 - err_meas6[0]
        # err_ub = meas6 + err_meas6[1]
        # ax.fill_between(meas5[arg_sort_meas], err_lb[arg_sort_meas], err_ub[arg_sort_meas], color = sum_color, alpha = alpha_conf_int)

def _ccdf_vs_deg(dict_deg):
    def normalized_ccdf(arr):
        ccdf = np.cumsum(arr)[::-1]
        # Problems with ccdf/ccdf[0]. Therefore, use np.divide
        ccdf = np.divide(ccdf, ccdf[0])
        return ccdf

    obs_sort_deg, sum_model_sort_deg, ref_model_sort_deg = np.sort(dict_deg["obs"]), np.sort(dict_deg["sum_model"]), np.sort(dict_deg["ref_model"])
    ccdf_vs_deg = {"obs": obs_sort_deg, "obs_ccdf": normalized_ccdf(obs_sort_deg), 
                    "sum_model" : sum_model_sort_deg, "sum_model_ccdf" : normalized_ccdf(sum_model_sort_deg),
                    "ref_model" : ref_model_sort_deg, "ref_model_ccdf" : normalized_ccdf(ref_model_sort_deg)}

    return ccdf_vs_deg

def inset_plot_ccdf(ax, dict_meas,
                        obs_color = obs_color, sum_model_color = sum_model_color, ref_model_color = ref_model_color,
                        obs_net_label = "observed", sum_model_label = "sum_model", ref_model_label = "sum_model",
                        inset_title = "none",):
    """Function to plot the ccdf of the observed, sum_model and ref_model"""
    # load in ccdf all the ccdf obtained from dict_meas
    
    lw = 7
    ccdf = _ccdf_vs_deg(dict_meas)
    ax.step(ccdf["obs"], ccdf["obs_ccdf"], color = obs_color, lw = 7, label = obs_net_label)
    ax.step(ccdf["ref_model"], ccdf["ref_model_ccdf"], color = ref_model_color, lw = 7, label = ref_model_label)
    ax.step(ccdf["sum_model"], ccdf["sum_model_ccdf"], color = sum_model_color, lw = 7, label = sum_model_label)

    ax.set(ylabel=f'CCDF', xlabel= 'DEG', title=f'{inset_title}', xscale = "log", yscale= "linear")
    ax.legend(loc = 0)

def save_dict_meas(meas_dict, obs_dict_dir, sum_dict_dir):
    from pickle import dump
    import os

    # extract from meas_dict the observed dict. Therefore, what remains are the expected measures
    obs_dict = {}
    for k in meas_dict:
        if meas_dict[k] != None:
            obs_dict[k] = meas_dict[k]["obs"]
            del meas_dict[k]["obs"]
    
    # check and save the observed measurements
    if not os.path.exists(obs_dict_dir):
        with open(obs_dict_dir, "wb") as f:
            dump(obs_dict, f)

    # check and save the expected measurements
    if not os.path.exists(sum_dict_dir):
        with open(sum_dict_dir, "wb") as f:
            dump(meas_dict, f)

def _set_dict_deg_annd_cc(obs_net, sum_model, ref_model = None, confidence = 0, n_samples = 10):
    """ Fill deg, annd, cc with the values for each model ("obs", "sum_model", "ref_model") computed (and saved) ens_avg_lbci_ubci_{netmeas}_conf{confidence}
    1) if reduced_by != None, there will be the average of the structural equivalent nodes;
    2) if confidence > 0, for ref_model and sum_model, the n_sample-ensemble will be generated and the enseble averages +- conf.int. will be stored

    Overall, an e.g. for the degree is: 
    - deg["obs"] carries the observed degree sequence, 
    - deg["sum_model"] the one for the sum-model
    - deg["ref_model"] the fitted model one
    """

    # create the file_names to check if they already exists
    reduced_label = f"reduced_by_{ref_model.reduced_by}_" if ref_model.reduced_by else ""
    obs_dict_dir = f"{obs_net.model_dir}/strineq_meas_dicts/{reduced_label}deg_annd_cc.pkl"
    
    # if conf > 0, save also errors
    conf_label = f"_conf{confidence}" if confidence > 0 else ""
    sum_dict_dir = f"{sum_model.model_dir}/strineq_meas_dicts/{reduced_label}deg_annd_cc{conf_label}.pkl"
    
    # create the folders to contain the variables
    os.makedirs(os.path.dirname(obs_dict_dir), exist_ok = True)
    os.makedirs(os.path.dirname(sum_dict_dir), exist_ok = True)

    if not (os.path.exists(obs_dict_dir) and os.path.exists(sum_dict_dir)):

        if ref_model.reduced_by:
            # find the idx of structural inequivalent nodes. 
            # For e.g. deg contains all the deg even of structural equivalent nodes. Thus, we want to plot only the strineq ones.
            idx, avg_obs_deg, avg_obs_annd, avg_obs_cc = avg_streq_nodes(obs_net, ref_model.streq_nodes.copy())

            # since ens_err_** contains [lower_bound, upper_bound] we need to index it [0,1] x idx
            err_idx = np.ix_([0,1], idx)
            deg = {"obs": avg_obs_deg}
            annd = {"obs": avg_obs_annd}
            cc = {"obs": avg_obs_cc}
        else:
            # select all the entries of idx and err_idx
            idx = [True]*obs_net.n_nodes
            err_idx = [True, True]
            deg = {"obs": obs_net.deg}
            annd = {"obs": obs_net.annd}
            cc = {"obs": obs_net.cc}

        if confidence > 0:
            # TO DO: load the measures if n_sample is meet!
            sum_model._set_ens_deg_annd_cc(n_samples=n_samples, confidence=confidence)
            ref_model._set_ens_deg_annd_cc(n_samples=n_samples, confidence=confidence)
            
            # load the measures and if average over deg is needed the proper indexes will be used
            deg.update({"sum_model": sum_model.ens_avg_deg[idx], "ref_model" : ref_model.ens_avg_deg[idx]})
            err_deg = {"obs": None, "sum_model": sum_model.ens_err_deg[err_idx], "ref_model" : ref_model.ens_err_deg[err_idx],}
            annd.update({"sum_model": sum_model.ens_avg_annd[idx], "ref_model" : ref_model.ens_avg_annd[idx],})
            err_annd = {"obs": None, "sum_model": sum_model.ens_err_annd[err_idx], "ref_model" : ref_model.ens_err_annd[err_idx],}
            cc.update({"sum_model": sum_model.ens_avg_cc[idx], "ref_model" : ref_model.ens_avg_cc[idx],})
            err_cc = {"obs": None, "sum_model": sum_model.ens_err_cc[err_idx], "ref_model" : ref_model.ens_err_cc[err_idx],}

        else:
            deg.update({"sum_model": sum_model.deg[idx], "ref_model" : ref_model.deg[idx],})
            annd.update({"sum_model": sum_model.annd[idx], "ref_model" : ref_model.annd[idx],})
            cc.update({"sum_model": sum_model.cc[idx], "ref_model" : ref_model.cc[idx],})
            err_deg, err_annd, err_cc = None, None, None

        # load all the measurements into meas_dict dictionary
        meas_dict = {"deg" : deg.copy(), "annd" : annd.copy(), "cc" : cc.copy()}
        
        # add the errors if confidence > 0
        if confidence > 0:
            meas_dict.update({"err_deg" : err_deg.copy(), "err_annd" : err_annd.copy(), "err_cc" : err_cc.copy()})
        
        # save the meas_dict by splitting what was from the observed net, ref_model and sum_model
        save_dict_meas(meas_dict, obs_dict_dir, sum_dict_dir)

    else:
        # meas_dict = {"deg" : {}, "annd" : {}, "cc" : {}, "err_deg" : {}, "err_annd" : {}, "err_cc" : {}}
        from pickle import load

        # insert the values of the obs_dict according to the structure above
        deg, annd, cc = {}, {}, {}
        with open(obs_dict_dir, 'rb') as f:
            obs_dict = load(f)
            deg["obs"], annd["obs"], cc["obs"] = obs_dict["deg"], obs_dict["annd"], obs_dict["cc"]

        # insert the values of the sum_models + ref meas into meas_dict
        with open(sum_dict_dir, 'rb') as f:
            ref_sum_dict = load(f)
            deg.update(ref_sum_dict["deg"])
            annd.update(ref_sum_dict["annd"])
            cc.update(ref_sum_dict["cc"])

        err_deg, err_annd, err_cc = None, None, None
        if confidence > 0:
            err_deg, err_annd, err_cc = {"obs" : None}, {"obs" : None}, {"obs" : None}
            err_deg.update(ref_sum_dict["err_deg"]), err_annd.update(ref_sum_dict["err_annd"]), err_cc.update(ref_sum_dict["err_cc"])

    return deg, annd, cc, err_deg, err_annd, err_cc

def avg_streq_nodes(net, streq_nodes):
    '''
    Given the observed class, it returns the average degree sequence, average annd and average cc for the same-degree nodes
    '''

    def _check_and_fill(arr, streq_dict):   
        """Check if there are deterministic nodes left-out and fill only with the representative one"""
        if np.atleast_1d(arr).size:
            streq_dict[arr[0]] = arr

    # fill the stre_nodes with the deterministic nodes as the obs_net_meas will regard all the nodes
    all_streq_nodes = streq_nodes.copy()

    # fill fc, fd, only2fc nodes
    _check_and_fill(net.fc_nodes, all_streq_nodes)
    _check_and_fill(net.fd_nodes, all_streq_nodes)
    if net.get("only2fc_nodes"):
        _check_and_fill(net.only2fc_nodes, all_streq_nodes)

    # sort them since we want that the first avg_deg will be liked with the expected sum_model
    # Otherwise, we shoudl change the ordering the expected sum_model
    all_strineq_nodes = np.sort(list(all_streq_nodes.keys()))

    # function to average over the deg, annd, cc of the net / observed
    avg_str_var = lambda net_meas: np.array(list(map(lambda i: np.mean(net_meas[i]), all_strineq_nodes)))

    return all_strineq_nodes, avg_str_var(net.deg), avg_str_var(net.annd), avg_str_var(net.cc)

def _perm_scalar_formatter(a, b):
    """ Find if two numbers are in the same Order Of Magnitude, i.e. OOM, is the power of 10 of a number. 
    'Same' means it is <= 1"""
    
    import math

    # safe way of accounting for the case of zero clustering coefficient
    if a == 0:
        oom_a = 0
    else:
        # order of magnitude i.e. log_10(x) // 1
        oom_a = math.floor(math.log10(abs(a)))
    
    if b == 0:
        oom_b = 0
    else:
        oom_b = math.floor(math.log10(abs(b)))

    # increase by one the oom if the rel_err with the next oom is < 1
    # print(f"-a,b,oom_a,oom_b: {a}, {b}, {oom_a}, {oom_b}")
    rel_err_next_oom = lambda x, oom: (10**(oom + 1) - x) / x if x > 0 else 0

    # set the oom to 0 if the number is 1 otherwise the relative error would increase it by one, i.e. 0 -> 1
    if a == 1:
        oom_a = 0
    elif rel_err_next_oom(a, oom_a) < 1:
        oom_a += 1
    
    # do the same for the upper bound b
    if b == 1:
        oom_b = 0
    elif rel_err_next_oom(b, oom_b) < 1:
        oom_b += 1

    # print(f"-a,b,oom_a,oom_b: {a}, {b}, {oom_a}, {oom_b}")
    # print(f'-abs(oom_a - oom_b): {abs(oom_a - oom_b)}',)
    return abs(oom_a - oom_b) <= 1

def remove_NaN(arr):
    """Remove NaN values from the array"""
    return arr[~np.isnan(arr)]

def set_scalarformatter(ax, x_obs_meas, y_obs_meas, x_meas, y_meas, style = 'sci'):
    """ 
    Set the scalar formatter for the axis x or y if both exp_XY_meas and obs_meas are agreeing in being represented by a scalar (scientific) scale. 
    E.g. if '220 221', the ticks will be '2.2, 2.1' and at the end 'x 1e2'
    """
    import matplotlib.ticker as ticker

    # it may happen in the sampled measures that we have NaN values. Therefore, we need to remove them
    x_obs_meas, x_meas, y_meas = remove_NaN(x_obs_meas), remove_NaN(x_meas), remove_NaN(y_meas)

    # print(f'-x_obs_meas.min(), x_obs_meas.max(), y_obs_meas.min(), y_obs_meas.max(): {x_obs_meas.min(), x_obs_meas.max(), y_obs_meas.min(), y_obs_meas.max()}',)
    # print(f'-x_meas.min(), x_meas.max(), y_meas.min(), y_meas.max(): {x_meas.min(), x_meas.max(), y_meas.min(), y_meas.max()}',)

    # find the permission given by the obs x and y measures
    x_min_obs, x_max_obs, y_min_obs, y_max_obs = x_obs_meas.min(), x_obs_meas.max(), y_obs_meas.min(), y_obs_meas.max()
    x_min, x_max, y_min, y_max = x_meas.min(), x_meas.max(), y_meas.min(), y_meas.max()

    # find the permission for x-axis
    x_obs_perm = _perm_scalar_formatter(x_min_obs, x_max_obs)
    x_perm = _perm_scalar_formatter(x_min, x_max)

    # find the permission for y-axis
    y_obs_perm = _perm_scalar_formatter(y_min_obs, y_max_obs)
    y_perm = _perm_scalar_formatter(y_min, y_max)
    # print(f'-x_obs_perm, y_obs_perm, x_perm, y_perm: {x_obs_perm, y_obs_perm, x_perm, y_perm}',)

    # set the x-axis and y-axis based on the permissions
    # minor_scalar_Formatter = ticker.ScalarFormatter()
    # minor_scalar_Formatter.set_powerlimits((0, 1))
    scalarFormatter = ticker.ScalarFormatter()

    if x_perm and x_obs_perm:
        ax.xaxis.set_major_formatter(scalarFormatter)
        ax.xaxis.set_minor_formatter(scalarFormatter)
        ax.ticklabel_format(style = 'plain' if x_max_obs < 100 else style, axis = 'x', scilimits=(0,0))

    if y_perm and y_obs_perm:
        ax.yaxis.set_major_formatter(scalarFormatter)
        ax.yaxis.set_minor_formatter(scalarFormatter)
        ax.ticklabel_format(style = 'plain' if y_max_obs < 100 else style, axis = 'y', scilimits=(0,0))

def inset_plot_rec_meas(ax, dict_meas, dict_meas2 = None, dict_err = None, dict_err2 = None,
                        obs_net_label = "observed", sum_model_label = "sum_model", ref_model_label = "summed_sum_model",
                        inset_title = "none", netmeas_label = "none", kind = "rec",
                        **kwargs):
    """
    Inset Function To Generate the and the plots:
    
    1)  ``exp VS obs netmeas'': use only of the dict_meas and dict_err (confidence is provided iff dict_err != None);
    2) ``(exp_netmeas, exp_deg) VS (obs_netmeas, obs_deg)'': use dict_meas, dict_meas2 and dict_err, dict_err2 (confidence is provided iff dict_err != None).
    
    """
    from dependencies import sum_model_color, ref_model_color
    
    # set errors = 0 not to have errorbars
    err_obs_meas, err_meas2, err_meas3, err_meas4, err_meas5, err_meas6 = np.array([0]*6)

    # arrange properly the measurements we can use only one function, that may be tweaked, to plot rec and vs_deg measures
    obs_meas = dict_meas["obs"]
    if kind == "rec":

        # the plot will be: 
        # 1) identity (obs vs obs) : obs_meas vs meas2
        # 2) obs vs rec_model (y) : meas3 vs meas4
        # 3) obs vs sum_model (y) : meas5 vs meas6
        meas2, meas3, meas5 = obs_meas, obs_meas, obs_meas
        meas4, meas6 = dict_meas["ref_model"], dict_meas["sum_model"]
        xlabel, ylabel = f'Observed', f'Expected'
        
        # load the errors if dcit_err != None
        if dict_err != None:
            err_meas4, err_meas6 = dict_err["ref_model"], dict_err["sum_model"]

    elif kind == "vs_deg":
        # the plot will be: 
        # 1) identity (deg vs net_meas): obs_meas vs meas2
        # 2) ref_model-net_meas vs ref_model deg : meas3 vs meas4 
        # 3) sum_model-net_meas vs sum_model deg: meas5 vs meas6
        meas2, meas3, meas4, meas5, meas6 = dict_meas2["obs"], dict_meas["ref_model"], dict_meas2["ref_model"], dict_meas["sum_model"],  dict_meas2["sum_model"], 
        xlabel, ylabel = 'DEG', f'{netmeas_label}'

        if dict_err != None:
            err_meas2, err_meas3, err_meas4, err_meas5, err_meas6 = dict_err2["obs"], dict_err["ref_model"], dict_err2["ref_model"], dict_err["sum_model"], dict_err2["sum_model"]

    # observed trends (identity line for the "rec" and behavior "vs_deg" for the other ones)
    ax.scatter(obs_meas, meas2, marker = obs_marker, c = obs_color, s = obs_ms, label = obs_net_label)


    # among ax.scatter and ax.errorbar if the err_meas3 = 0 as in that case we would be in the "analytical case where we don't approximate the error"
    ax_scatter_errorbar(ax, meas3, meas4, meas5, meas6, err_meas3, err_meas4, err_meas5, err_meas6, ref_model_color, sum_model_color, ref_model_ms, sum_model_ms, ref_model_label, sum_model_label)
    
    axis_scale = "log" if np.min(obs_meas) > 0 else "linear"
    ax.set(xlabel = xlabel, ylabel = ylabel, title=f'{inset_title}', xscale = axis_scale, yscale = axis_scale)

    loc = 2 if netmeas_label == r"$k_i$" else 0
    
    ax.legend(loc = loc)

    # set scalar formatter for the axis x and y if they in the same order of magnitude
    set_scalarformatter(ax, obs_meas, meas2, meas3, meas4)

    # seet the grid to the background
    ax.set_axisbelow(True)

def plots_bin_meas_vs_deg(obs_net, sum_model, ref_model = None, suptitle = None, save = False, confidence = 0, n_samples = 100):
    '''
    plot the binary network measures
    upper triptych: deg (+ p_ij inset), knn, cc;
    lower triptych: ccdf / knn / cc VS deg
    '''
    from plotting_functions import save_fig
    import matplotlib.ticker as ticker
    
    # easy first part only to check if it is already existing the plot we are requiring
    plot_full_path = f"{sum_model.plots_dir}/bin_meas_vs_deg/level{obs_net.level:g}.pdf"
    if confidence > 0:
        plot_full_path = f"{sum_model.plots_dir}/bin_meas_vs_deg_conf{confidence}/level{obs_net.level:g}.pdf"

    if not os.path.exists(plot_full_path):
            
        fig, axs = plt.subplots(2,3, figsize = (30,17))

        deg, annd, cc, err_deg, err_annd, err_cc = _set_dict_deg_annd_cc(obs_net, sum_model, ref_model, confidence = confidence, n_samples = n_samples)
        ccdf_vs_deg = _ccdf_vs_deg(dict_deg = deg)

        # plotting part in axs[0,0]
        inset_plot_rec_meas(axs[0,0], dict_meas = deg, dict_err = err_deg,
                            inset_title = f"DEG", netmeas_label = r"$k_i$", 
                            obs_net_label = obs_net.name, 
                            sum_model_label = "Summed" + r" ($z_L = $" + f"{z_score_edg(obs_net, sum_model):.0e})", 
                            ref_model_label = ref_model.plots_name + r" ($z_L = $" + f"{z_score_edg(obs_net, ref_model):.0e})")
        
        # add the inset axis for the connection probability
        _plot_inset_pmatrices(fig, axs[0,0], obs_net, ref_model=ref_model, sum_model=sum_model)

        # plot annd_i and cc_i
        inset_plot_rec_meas(axs[0,1], dict_meas = annd, dict_err = err_annd,
                            inset_title = f"ANND", netmeas_label = r"$k^{nn}_i$", 
                            obs_net_label = obs_net.name,
                            sum_model_label = "Summed", ref_model_label = ref_model.plots_name)
        
        inset_plot_rec_meas(axs[0,2], dict_meas = cc, dict_err = err_cc,
                            inset_title = f"CC", netmeas_label = "CC", 
                            obs_net_label = obs_net.name,
                            sum_model_label = "Summed", ref_model_label = ref_model.plots_name)


        # Behavior of measurments VS degrees -- observed, sum_model, ref_model
        # NEW plotting function inset_plot_ccdf since it is really different from the other ones 
        inset_plot_ccdf(axs[1,0], dict_meas = ccdf_vs_deg,
                            inset_title = f"CCDF VS DEG",
                            obs_net_label = obs_net.name, sum_model_label = "Summed", ref_model_label = ref_model.plots_name)
        inset_plot_rec_meas(axs[1,1], 
                            dict_meas = deg, dict_err = err_deg,
                            dict_meas2 = annd, dict_err2 = err_annd, kind = "vs_deg",
                            obs_net_label = obs_net.name, sum_model_label = "Summed", ref_model_label = ref_model.plots_name,
                            inset_title = "ANND VS DEG", netmeas_label = "ANND")
        inset_plot_rec_meas(axs[1,2], dict_meas = deg, dict_err = err_deg,
                            dict_meas2 = cc, dict_err2 = err_cc,
                            kind = "vs_deg",
                            obs_net_label = obs_net.name, sum_model_label = "Summed", ref_model_label = ref_model.plots_name,
                            inset_title = "CC VS DEG", netmeas_label = "CC")
        
        if suptitle == None:
            suptitle = f"Binary Plots Summed - year: {sum_model.year}"
        
        fig.suptitle(suptitle, x = 0.5, y = 0.93)
        save_fig(fig, full_path = plot_full_path)
        plt.close()

def plot_deg_raw(sum_model = None, obs_deg = None, exp_deg = None, title = None):
    _, ax = plt.subplots(figsize = (7,7))
    if np.array_equal(exp_deg, None):
        exp_deg = sum_model.deg
    ax.scatter(obs_deg, obs_deg, s = 5, color = "navy", label = "observed")
    label = ""
    if sum_model != None:
        f"Summed.deg"
    ax.scatter(obs_deg, exp_deg, s = 40, marker = "x", color = "red", label = label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("observed degrees")
    ax.set_ylabel("expected degrees")
    ax.grid()
    ax.legend()
    if title == None:
        title = "Degrees"
    ax.set_title(title)
    plt.show()

def z_score_edg(obs_net, sum_model):
    '''
    z-score for the density of the observed VS expected adjacency matrix.
    Since the density is only a recaling of the number of edges. Just calculate the z_score for the n_edges. 
    '''
    std = lambda cl: np.sqrt(np.triu(cl.zl_pmatrix*(1-cl.zl_pmatrix), k = 1).sum())
    return (obs_net.n_edges - sum_model.n_edges) / std(sum_model)