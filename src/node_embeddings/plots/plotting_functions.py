from ..utils.helpers import *
from ..lib import *
from ..plots.obs_vs_rec_meas import *
from ..utils.helpers import set_name_for_plots
from ..utils import matplotlib_settings as mpls
from ..utils.matplotlib_settings import ref_model_ms, ref_model_ms

def plots_rec_bin_vs_deg_cm_roc_prc(obs_net, sum_model, ref_model = None, confidence = 0, n_samples = 100):
	""" 
	Master Function which wraps all the plotting functions for the 
	- network measures;
	- Binary ROC / PR scores;
	- Parameters Plotting;
	- Saving the values of nll and n_params_X
	"""
	# create title_dimBCX since for LCPA and maxlMSM we want to report the dimX else for ConfMat, ... None
	title_dimBCX = ""
	if ref_model.objective == "NodeEmb":
		title_dimBCX = f", dimB: {sum_model.dimB}, dimC: {sum_model.dimB}" if sum_model.name.endswith("LPCA") else f", dimX: {sum_model.dimX}"
		
	tech_details = f"{sum_model.dataset_name} Network -- level {obs_net.level}{title_dimBCX}"
	
	netmeas_suptitle = f"Network Measures on {tech_details} "
	plots_bin_meas_vs_deg(obs_net, sum_model, ref_model, suptitle = "", confidence = confidence, n_samples = n_samples, save = True)

	# Score Functions with loops only if all the matrix is not 1
	if not (np.all(obs_net.bin_adj) or np.all(obs_net.bin_adj == 0)):
		plot_cm_roc_prc(obs_net = obs_net, sum_model = sum_model, ref_model = ref_model, suptitle = f"ConfMat-ROC-PR scores on {tech_details}", save = True)
		if not sum_model.name.endswith("LPCA"):
			plot_grained_fitn(sum_model, ref_model, save_std = True)
	else:
		print(f"\n-Warning: only one class {obs_net.bin_adj[0,0]} --> Skipping the plot_cm_roc_prc / plot_grained_fitn")    

	# save the negative loglikelihood and the n of parameters (we will use them for the AIC and BIC scores)
	for e_class in [sum_model, ref_model]:
		
		# save NLL
		if e_class.name.endswith("LPCA"):
			nll = e_class.nll_LPCA(X = e_class.X.ravel(), A = tc.from_numpy(obs_net.bin_adj))
		elif e_class.name.endswith("maxlMSM"):
			nll = e_class.nll(X = e_class.X, A = obs_net.bin_adj)
		
		if not os.path.exists(f"{e_class.model_dir}/nll.csv") and e_class.name.endswith(("LPCA","maxlMSM")):
			np.savetxt(f"{e_class.model_dir}/nll.csv", np.array([nll]), fmt = "%.18f")
		
		
		# save the n_params_X for AIC and BIC scores only for fitted model
		if e_class.name in ["LPCA", "maxlMSM"]:
			
			# number of observation if all the matrix is needed
			n_observations = e_class.n_nodes * (e_class.n_nodes - 1) / 2
			
			# rescale aic and bic with n * (n-1) for an average aic, bic per pair that may be compared across scales
			aic = 2*(nll + e_class.n_params_X)
			bic = 2*nll + e_class.n_params_X * np.log(n_observations)

			aic /= n_observations
			bic /= n_observations
			if not os.path.exists(f"{e_class.model_dir}/n_params.csv"):
				np.savetxt(f"{e_class.model_dir}/n_params.csv", np.array([e_class.n_params_X]), fmt = "%i")
			
			if not os.path.exists(f"{e_class.model_dir}/avg_aic.csv"):
				np.savetxt(f"{e_class.model_dir}/avg_aic.csv", np.array([aic]), fmt='%.18e')
				np.savetxt(f"{e_class.model_dir}/avg_bic.csv", np.array([bic]), fmt='%.18e')

# Score functions
def expected_confusion_matrix(true_mat, sum_model):
	'''
	Calcualte the expected cm for the true_mat and the sum_model.
	Discard the self-loops
	'''
	# Normalization
	prob_mat = sum_model.pmatrix
	N = true_mat.shape[0]

	# number of pairs without self-loops
	tot_pairs = N*(N-1) / 2

	# since we are discard the self-loops in the newtork measures discard it even here with k = 1
	sum_triu = lambda A: np.sum(np.triu(A, k = 1))
	obs_edges = sum_triu(true_mat)
	sum_model_edges = sum_model.n_edges 

	# Multiplication for < TP > = sum_triu(A*P)
	sum_model.tp = sum_triu(true_mat*prob_mat)

	# Multiplication for < TN > = N(N+1)/2 - (L^* + <L>) + < TP >
	sum_model.tn = tot_pairs - (obs_edges+sum_model_edges) + sum_model.tp

	# Multiplication for < FP > = sum_triu( (1-A) * P ) = < L > - < TP >
	sum_model.fp = sum_model_edges - sum_model.tp

	# Multiplication for < FN > = sum_triu( A * (1-P) ) = L^* - < TP >
	sum_model.fn = obs_edges - sum_model.tp

	# EXPECTED confusion matrix
	sum_model_cm = np.array([[sum_model.tn, sum_model.fp],[sum_model.fn, sum_model.tp]])

	# normalized by the number of entries
	return sum_model_cm / tot_pairs

def plot_cm_roc_prc(obs_net, sum_model, ref_model, suptitle = "", save = False):
	from ..utils.helpers import cmap
    
	plot_full_path = f"{sum_model.plots_dir}/cm_roc_prc/level{sum_model.level:g}.pdf"
	if not os.path.exists(plot_full_path):

		from matplotlib import pyplot as plt
		import pandas as pd
		import numpy as np
		from sklearn.metrics import roc_curve, ConfusionMatrixDisplay
		from sklearn.metrics import RocCurveDisplay, roc_auc_score
		from sklearn.metrics import precision_recall_curve, auc
		from sklearn.metrics import PrecisionRecallDisplay 

		# ravel with triu_idx to get the final density for sum_precision[0]
		triu_idx = np.triu_indices_from(obs_net.bin_adj, k = 1)
		y_true = obs_net.bin_adj[triu_idx]
		sum_y_score = sum_model.pmatrix[triu_idx]
		ref_y_score = ref_model.pmatrix[triu_idx]

		sum_model_cm = expected_confusion_matrix(obs_net.bin_adj, sum_model)
		ref_model_cm = expected_confusion_matrix(obs_net.bin_adj, ref_model)

		sum_fpr, sum_tpr, _ = roc_curve(y_true, sum_y_score)
		ref_fpr, ref_tpr, _ = roc_curve(y_true, ref_y_score)

		sum_precision, sum_recall, _ = precision_recall_curve(y_true, sum_y_score)
		ref_precision, ref_recall, _ = precision_recall_curve(y_true, ref_y_score)

		# the first values of PR is 0/0 and sci-kit learn sets it to 1. 
		# I decide to discard it, setting it to the 2nd-to-last value
		sum_precision[-1], sum_recall[-1] = sum_precision[-2], 0
		ref_precision[-1], ref_recall[-1] = ref_precision[-2], 0

		# consider only the AUC but above the no skill-identity line
		# normalize it s.t. the AUC-ROC the values span from -1 to 1. Thus, the best value is 1
		normalize_roc = lambda auc: 2*(auc - 0.5)
		auc_roc = normalize_roc(roc_auc_score(y_true, sum_y_score))
		auc_roc2 = normalize_roc(roc_auc_score(y_true, ref_y_score))

		# consider only the AUC but above density line / horizontal line at the lowest threshold
		# normalize it s.t. the AUC-PR the values span from -1 to 1. Thus, the best value is 1
		density = obs_net.n_edges / (obs_net.n_nodes * (obs_net.n_nodes - 1) / 2)
		normalize_pr = lambda pr: (pr - density) / (1 - density)
		# auc_prc = normalize_pr(average_precision_score(y_true, sum_y_score))
		# auc_prc2 = normalize_pr(average_precision_score(y_true, ref_y_score))
		auc_prc = normalize_pr(auc(sum_recall, sum_precision))
		auc_prc2 = normalize_pr(auc(ref_recall, ref_precision))

		# save auc_roc, auc_prc as evalutation metrics
		sum_full_path = f"{sum_model.model_dir}/auc_roc_prc.csv"
		if not os.path.exists(sum_full_path):
			np.savetxt(sum_full_path, np.array([auc_roc, auc_prc]), fmt = "%.18f", delimiter=",")
		
		ref_full_path = f"{ref_model.model_dir}/auc_roc_prc.csv"
		if not os.path.exists(ref_full_path):
			np.savetxt(ref_full_path, np.array([auc_roc2, auc_prc2]), fmt = "%.18f", delimiter=",")

		fig = plt.figure(figsize=(25, 8))
		# Expected cm for the refitted sum_model
		ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=1)

		# change the ytick.major.size to 2 and restore it 
		# This is a global change since the cmap is not tunable as for ax0 with, e.g., ax0.set_ytick_params
		old_ytick_major_size = mpl.rcParams['ytick.major.size']
		mpl.rcParams['ytick.major.size'] = 2

		# Expected cm for the coarse-grained ref_model
		ConfusionMatrixDisplay(confusion_matrix=ref_model_cm).plot(cmap=cmap, ax = ax0)
		ax0.set_title(f"{ref_model.plots_name} " + r'$\langle$' + 'ConfMat' + r'$\rangle$')
		ax0.tick_params(size=0, which = 'both')
		ax0.grid(False)

		# Expected cm for the coarse-grained sum_model
		ax1 = plt.subplot2grid((2, 3), (1, 0), colspan=1)
		ConfusionMatrixDisplay(confusion_matrix=sum_model_cm).plot(cmap=cmap, ax = ax1)
		ax1.set_title(f"Summed " + r'$\langle$' + 'ConfMat' + r'$\rangle$')
		ax1.tick_params(size=0, which = 'both')	
		ax1.grid(False)

		# restore the ytick.major.size
		mpl.rcParams['ytick.major.size'] = old_ytick_major_size
		lw_ref, lw_sum = 4, 3
		zorder = 3
		
		# Plot ROC
		ax2 = plt.subplot2grid((2, 3), (0, 1), rowspan=2)
		RocCurveDisplay(fpr=ref_fpr, tpr=ref_tpr).plot(ax2, label = f"{ref_model.plots_name} - area: {auc_roc2:.3}", color = mpls.ref_model_color, lw = lw_ref, zorder = zorder)
		RocCurveDisplay(fpr=sum_fpr, tpr=sum_tpr).plot(ax2, label = f"Summed - area: {auc_roc:.3}", color = mpls.sum_model_color, lw = lw_sum, zorder = zorder)

		ax2.plot([0, 1], [0, 1], color = "black", linestyle = "--", label = 'No Skill', zorder = zorder-1)

		ax2.set_title("Receiver Operating Curve")
		ax2.set_xlabel(r"FPR($\epsilon$) := FP($\epsilon$) / (N(N-1)/2 - L)")
		ax2.set_ylabel(r"TPR($\epsilon$) := TP($\epsilon$) / L")
		ax2.legend(loc = 0)

		# Plot PR
		ax3 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
		PrecisionRecallDisplay(precision=ref_precision, recall=ref_recall).plot(ax=ax3, label = f"{ref_model.plots_name} - area: {auc_prc2:.3}",  color = mpls.ref_model_color, lw = lw_ref, zorder = zorder)
		PrecisionRecallDisplay(precision=sum_precision, recall=sum_recall).plot(ax=ax3, label = f"Summed - area: {auc_prc:.3}", color = mpls.sum_model_color, lw = lw_sum, zorder = zorder)
		
		# find the minimum to start the PR
		no_skill = len(y_true[y_true==1]) / len(y_true)
		ax3.plot([0, 1], [no_skill, no_skill], color = "black", linestyle='--', label=f'No Skill', zorder = zorder-1)
		ax3.set_title("Precision Recall Curve")
		ax3.set_xlabel(r"Recall($\epsilon$)   := TP($\epsilon$) / L = TPR($\epsilon$)")
		ax3.set_ylabel(r"Precison($\epsilon$) := TP($\epsilon$) /" + r'$\langle$' + 'L' + r'$\rangle(\epsilon)$')
		ax3.legend(loc = 0)
		# fig.suptitle(suptitle)

		# Show the graph
		plt.tight_layout()

		save_fig(fig, full_path = plot_full_path, save = save)
		plt.close()

def clf_sum_model_scores(obs_net, sum_model):
	
	'''
	Expected Scores used for Classification Task
	'''
	def h_mean(x,y):
		return np.round(2*x*y / (x+y),6)

	sum_model.n_loops = np.diagonal(sum_model.zl_pmatrix).sum()
	N = obs_net.n_nodes

	# positives
	sum_model_pos = np.sum(np.triu(sum_model.zl_pmatrix, k = 1))
	obs_pos = obs_net.n_edges

	# negatives
	all_pairs = N*(N-1)/2
	sum_model_neg = all_pairs - sum_model_pos
	obs_neg = all_pairs - obs_pos
	
	# composed scores
	sum_model.tpr = sum_model.tp / sum_model_pos # recall
	sum_model.fpr = sum_model.fp / sum_model_neg 
	sum_model.spc = sum_model.tn / sum_model_neg # tnr
	sum_model.ppv = sum_model.tp / (sum_model.tp+sum_model.fp) 
	sum_model.acc = (sum_model.tp + sum_model.tn) / all_pairs
	sum_model.npv = sum_model.tn / (sum_model.tn+sum_model.fn)
	mcc_1 = sum_model.tp*sum_model.tn-sum_model.fp*sum_model.fn
	mcc_2 = (sum_model.tp+sum_model.fp)*(sum_model.tp+sum_model.fn)*(sum_model.tn+sum_model.fp)*(sum_model.tn+sum_model.fn)

	df = pd.DataFrame(
		[
		("Pos", sum_model_pos),
		("Neg", sum_model_neg),
		("Loops", sum_model.n_loops),
		("True Negative", sum_model.tn),
		("False Positive", sum_model.fp),
		("FN", sum_model.fn),
		("TP", sum_model.tp),
		("TPRate", sum_model.tpr), 
		("FPR", sum_model.fpr), 
		("SPeCificity", sum_model.spc), 
		("Predicted Positive Value", sum_model.ppv), 
		("ACCuracy/micro-f1", sum_model.acc),
		("f1(cl=0,cl=1)", (h_mean(sum_model.npv, sum_model.spc), h_mean(sum_model.ppv,sum_model.tpr))),
		("MCC", mcc_1/np.sqrt(mcc_2)), 
		],
		columns=("Expected Scores", f"Summed values"),
	).set_index("Expected Scores")

	return df

def rescale_aic_bic_df(ref_model, models, dims, total_levels, aic_bic_flag = "aic"):

	""" Function to Obtain a DataFrame with the rescaled AIC and BIC values for the models.
	The goal is:
	1) Find min and max values over LPCA and maxlMSM;
	2) color green the min and max. The best are the mins;
	
	In needed,
	3) display the relative error for the non-minimal entries, i.e. AIC_i - AIC_min / AIC_min * 100 (the same for BIC)
	Wheras the mins have been shown to the original values.
	"""

	from os.path import dirname as dirname
	# import dataframe_image as dfi 

	def levels_min_max(models, ravel = False):
		max2color = []
		min2color = []

		for name in set(models):
			min2color.append(df.loc[name].min().to_numpy())
			max2color.append(df.loc[name].max().to_numpy())

		if ravel:
			min2color = np.ravel(min2color)
			max2color = np.ravel(max2color)

		return min2color, max2color

	# Step 3: Define a function to highlight max and min values
	def highlight_max_min(val, min2color, max2color):
		color = ''
		if val in max2color:
			color = 'color: red'
		elif val in min2color:
			color = 'color: green; font-weight: bold'
		return color

	# Custom formatting function for scientific notation
	def scientific_notation(val):
		n_digits = 4
		if val > 10**2:
			return f"{val:.4e}"
		return np.round(val, n_digits)

	if aic_bic_flag == "aic":
		# find the aic list
		aic_of_at_ = lambda level, name, str_dimXBC: load_meas(ref_model, level, name, str_dimXBC = str_dimXBC, meas = "avg_aic")
		aic_bic_list = lambda name, str_dimXBC: np.array([aic_of_at_(lvl, name, str_dimXBC) for lvl in range(total_levels)])

	else:
		# find the bic list
		bic_of_at_ = lambda level, name, str_dimXBC: load_meas(ref_model, level, name, str_dimXBC = str_dimXBC, meas = "avg_bic")
		aic_bic_list = lambda name, str_dimXBC: np.array([bic_of_at_(lvl, name, str_dimXBC) for lvl in range(total_levels)])

	# Convert dims to a uniform format (string or tuple)
	dims = [str(d) if isinstance(d, int) else d for d in dims]

	str_dimXBC = lambda name, dim: f"dimB{dim[0]}/dimC{dim[1]}" if name.endswith("LPCA") else f"dimX{dim}"
	aic_bics = np.array([aic_bic_list(name, str_dimXBC(name, dim)) for name, dim in zip(models, dims)])

	# Create a DataFrame from the lists
	level_aics = {f"Level {i}" : aic_bics[:, i] for i in range(total_levels)}

	data = {
		"Model": models,
		"Dimension": dims,
		**level_aics
		}

	df = pd.DataFrame(data)

	# Set the multi-index
	df.set_index(["Model", "Dimension"], inplace=True)


	# this part of the code is to
	# 1) find the minimum and max
	# 2) obtain the relative error
	# 3) replace the real minimum (instead of retaining zeros)
	# 4) color the table with green for real min and rel.error max

	# mins, _ = levels_min_max(models)
	# print(f'-min/level AVG {aic_bic_flag.upper()}: {mins}',)
	# print(f'-max/level AVG {aic_bic_flag.upper()}: {maxs}',)

	# Subtract the minimum value within each group from all the values in the group
	# df = df.groupby(level=0).transform(lambda x: (x - x.min()) / x.min() * 100)

	# # replace the zeros with the original values since the zeros are not informative rather the min values are as a benchmark
	# for i, name in enumerate(set(models)):
	#     for level in range(total_levels):
	#         df.loc[name, f"Level_{level}"].replace(0, mins[i][level], inplace = True)

	# create the min and max values such that we may color the table
	mins, maxs = levels_min_max(models, ravel = True)

	# Apply the highlighting function
	styled_df = df.style.map(lambda val: highlight_max_min(val, mins, maxs))\
				.format(scientific_notation)\
				.set_caption(f"{aic_bic_flag.upper()}s by class")\
				
				
				# .set_table_styles([
				#     {'selector': 'th', 'props': [('vertical-align', 'top')]},
				#     dict(selector="caption", props=[("text-align", "center"),("font-size", "150%"),("color", 'black')])])\
				#.format("{:.3e}")

	return styled_df

def plot_loss_gradn(self, save = True):
	seed_str = ""
	if self.__dict__.get('seed') != None: seed_str = f"seed{self.__dict__.get('seed'):g}_"
	plot_full_path = f"{self.plots_dir}/plot_loss_gradn/level{self.level:g}_{seed_str}plot_loss_gradn.pdf"

	if not os.path.exists(plot_full_path):
	
		fig, ax1 = plt.subplots(figsize = (12, 7))
		ax2 = ax1.twinx()
		colors = ["tab:red", "tab:blue"]
		labels = [f"Losses", f"Jac Norms"]
		leg2 = ax2.plot(self.nit_list, self.gradn_list, label = labels[1]+f" - lastv: {self.gradn_list[-1]:.2e}", color = colors[1]);
		leg1 = ax1.plot(self.nit_list, self.losses_list , label = labels[0]+ f" - lastv: {self.losses_list[-1]:.2e}", color = colors[0], zorder = 1)
		#leg3 = ax1.plot(self.nit_list, np.array(self.mse_deg_list)*np.max(self.losses_list), color = mpls.ref_model_color, label = "mse_deg", zorder = 1)

		idx_nm = [np.where(self.losses_list == loss)[0][0] for loss in self.opt_final_loss]
		its_nm = [self.nit_list[i] for i in idx_nm]
		ax1.scatter(its_nm, self.opt_final_loss, marker = "*", s = 13**2, label = "opt_final_loss", color = mpls.ref_model_color, zorder = 2)

		fontsize = 16
		for i, ax in enumerate([ax1, ax2]):
			ax.set_yscale("log")
			ax.tick_params(axis='y', which = "both", labelcolor = colors[i], labelsize = fontsize)
			ax.tick_params(axis='x', which = "both", labelsize = fontsize)
			ax.set_ylabel(labels[i], color = colors[i], fontsize = fontsize)

		for xi, yi, text in zip(its_nm, self.opt_final_loss, self.opt_names):
			ax1.annotate(text, xy=(xi, yi), xycoords='data', xytext=(3, 3), 
						fontsize = fontsize,
						textcoords='offset points', color = mpls.ref_model_color,)


		ax1.legend(handles=leg1+leg2, loc="best", fontsize = fontsize)
		ax1.set_title("Losses and Jacobian Norms", fontsize = fontsize)
		ax1.set_xlabel("Num of Iterations", fontsize = fontsize)
		ax1.set_zorder(ax2.get_zorder()+1)  # default zorder is 0 for ax1 and ax2
		ax1.set_frame_on(False)  # prevents ax1 from hiding ax2
		ax1.set_xlim(-its_nm[-1]*2e-2,its_nm[-1]*1.13)
		#ax1.set_xscale("log")

		save_fig(fig, full_path = plot_full_path)
		plt.close()

# plot fitnesses VS GDP
def plot_grained_fitn(sum_model, ref_model, save_std = False, xscale = "log", yscale = "log", color = mpls.sum_model_color):
	if sum_model.name.startswith("fg"):
		dir_appendix = "fine_vs_fit"
	else:
		dir_appendix = "sum_vs_fit"
	
	plot_full_path = sum_model.plots_dir+f"/sum_VS_fit_X/level{ref_model.level}.pdf"
	if not os.path.exists(plot_full_path):
		if sum_model.dimX == 1:
			fig, ax = plt.subplots(figsize = (7, 7))

			refit_X = ref_model.X.reshape(-1,ref_model.dimX)
			sum_X = sum_model.X.reshape(-1,sum_model.dimX)

			# if fc_nodes discard those entries
			if ref_model.fc_nodes.size > 0:
				sel_nodes = refit_X < np.max(refit_X)
				refit_X = refit_X[sel_nodes]
				sum_X = sum_X[sel_nodes]
			
			ax.plot(refit_X, refit_X, '--', color='grey', label = "identity")
			ax.plot(refit_X, sum_X, 'x', ms = 6, color= color, label = ref_model.name)

			# find the relative error array and avg, std
			rel_error = (sum_X - refit_X) / refit_X #(np.log(sum_X) - np.log(refit_X)) / np.log(refit_X)

			ax.set_axisbelow(True)
			ax.grid(True)
			# set the axis ax[0], ax[1]
			ax.set(ylabel = r'Summed  $x_I$', 
					xlabel= r'Fitted  $\hat{x}_{I}$',
					# title=f"Summed VS Fitted Params",
					#obsl: {ref_model.level}, pickl: {ref_model.pick_l}, fittedl: {ref_model.top_l}",
					xscale = xscale,
					yscale = yscale)

			ax.legend()

			plt.tight_layout()
			if save_std:
				save_fig(fig, full_path = plot_full_path)
			plt.close()

def level_exoX_vs_topoX(obs_net, sum_model = None, name_mark_color = None, colors = None, sum_model_ms = 5):

	""" 
	After training all the models, check the correlation among the GDP / Strenghts / Exogenous Variables (exox) VS the topological Fitnesses of the UMSM 
	The inputs must be a sum_model = maxlMSM for dimX = 1 as the rest is hard coded for fitnMSM, fitnMSM and CM.
	No LPCA as we have two fitnesses per node
	"""
	def normalize(model_name):
		""" If model_name is a string, load the x from the directory, otherwise normalize the x """
		
		x_dir = lambda model_name: full_path_retriever(sum_model, obs_net.level, model_name, str_dimXBC = "dimX1", meas = "X")
		if type(model_name) == str:
			x = np.genfromtxt(x_dir(model_name), delimiter=',', skip_header=False)

		return x / np.mean(x)

	def plot_exox_vs_topox_model(model_name, ms, ref_label = "", sum_label = "", zorder = 1):
		""" 
		Plot the normalized exogenous (observable) variables and the fitted parameter for each model
		Note: name_mark_color is set from outside as a lambda function
		"""
		
		x_dir = lambda model_name: full_path_retriever(sum_model, obs_net.level, model_name, str_dimXBC = "dimX1", meas = "X")
		
		if os.path.exists(x_dir(model_name)):
			# set the labels for loading the parameters
			sum_model_name = "sum-" + model_name
			
			# set the marker and color
			marker, color = name_mark_color[sum_model_name+"-1"]
			
			# plot it
			# fc = "None" if obs_net.level > 0 else color
			alpha = 0.5 if "fitn" in model_name else 1
			
			# don't plot the summed model if level 0 (fitted level the two models will coincide)
			if obs_net.level == 0:
				ax.scatter(norm_exox_l, normalize(model_name), marker = marker, c = color, s = ms, lw = .01, ec = 'k', alpha = alpha, label = model_name, zorder = zorder)
			else:
				ax.scatter(norm_exox_l, normalize(model_name), marker = "x", c = color, s = ms, alpha = alpha, label = model_name, zorder = zorder)
				ax.scatter(norm_exox_l, normalize(sum_model_name), marker = marker, c = color, ec = 'k', s = ms, lw = .01, label = "Summed", zorder = zorder)
		else:
			print(f'-{model_name} not existing @ {x_dir(model_name)}!',)

	level = obs_net.level
	# plots_dir = os.path.dirname(os.path.dirname(obs_net.plots_dir))
	path_file_name = sum_model.plots_dir_multi_models + f"/exoX_vs_topX/level{level:g}.pdf"

	if not os.path.exists(path_file_name):
		
		if obs_net.name.endswith("Gleditsch"):
			x_label = r"$GDP_I \, / \,  \overline{GDP}$"
		
		elif obs_net.name.endswith("ING"):
			x_label = r"$s_I \, / \, \overline{s}$"

		# now plotting if it does not exists
		fig, ax = plt.subplots(figsize = (10,10))
		
		# load the exogenous reference value, i.e. s_i / mean(s_i)
		cog_or_fine_label = sum_model.name.split("-")[0]
		norm_exox_l = normalize("fitnMSM")

		# since we take the normalized values, fitnMSM would be the same as of the observed normalized strengths
		ms = sum_model_ms**3 if obs_net.n_nodes > 200 else sum_model_ms**4
		plot_exox_vs_topox_model(model_name = "fitnMSM", ms = ms)

		# norm model parameters
		plot_exox_vs_topox_model(model_name = "CM", zorder = 1, ms = ms)
		plot_exox_vs_topox_model(model_name = "maxlMSM", zorder = 2, ms = ms)
		plot_exox_vs_topox_model(model_name = "degcMSM", zorder = 2, ms = ms)
		
		ax.set(xscale = "log", yscale = "log", xlabel = x_label, ylabel = r"$x_I \, / \, \overline{x}$", )
		ax.set_axisbelow(True)
		ax.grid(True)
		
		# rescale the markers and set alpha = 1 for fitn model
		leg = ax.legend(markerscale=3 if obs_net.n_nodes > 200 else 1)
		for lh in leg.legend_handles:
			lh.set_alpha(1)

		# save and close
		save_fig(fig, path_file_name)
		plt.close()

def plot_triangles_at_c(obs_net, ref_model, name_mark_color, n_points, only_summed = True):
	""" 
	Plot the triangles at c only for the Summed models
	"""
	from pathlib import Path

	# plot_full_path = plots_dir_(sum_model) + f"/multi_models/triangles_at_c/level{obs_net.level:g}_triangles_at_c.pdf"
	plot_full_path = ref_model.plots_dir_multi_models + f"/triangles_at_c/level{obs_net.level:g}.pdf"

	if not os.path.exists(plot_full_path):
		
		# enhance the names for the summed models
		names_dims = name_mark_color.keys()

		# import the directory
		pmatrix_name_dim = lambda name, str_dimXBC: load_meas(ref_model, obs_net.level, name, str_dimXBC, meas = "pmatrix")

		# select the cut-offs degrees
		lin_space_c = np.linspace(obs_net.deg.min(), obs_net.deg.max(), n_points)

		# funcion: triangles computation
		tri_dens = lambda A: (ref_model.frmv_diag(A @ A) @ A).diagonal().sum() / (2 * obs_net.n_nodes)
		obs_net.triangle_c = []

		# dictionary where to save the numb of triangles
		sum_model_triangle_c = dict(zip(names_dims, [[] for _ in names_dims]))

		# now start computing the triangles
		# select only the subgraph with the nodes with obs nodes with observed degree k_i > c
		full_path = obs_net.model_dir + "/triangles_at_c.csv"
		if os.path.exists(full_path):
			obs_net.triangle_c = np.genfromtxt(full_path, delimiter = ",")
		else:
			for c in lin_space_c:
				obs_subnodes = np.where(obs_net.deg >= c)[0]
				sub_A = obs_net.zl_bin_adj[obs_subnodes][:,obs_subnodes]
				obs_net.triangle_c.append(tri_dens(sub_A))
			np.savetxt(full_path, np.array(obs_net.triangle_c), fmt = "%.18f", delimiter=",")

		# for name, d, sum_name_dim in zip(models, dims, names_dims):
		for sum_name_dim in names_dims:
			
			# split sum_name_dim according to the name, e.g. "sum-LPCA-(1,1)" or "sum-maxlMSM-1"
			name, d = split_name_dim(sum_name_dim)
			str_dimXBC = f"dimB{d[0]}/dimC{d[1]}" if name.endswith("LPCA") else f"dimX{d}"
   
			full_path = full_path_retriever(ref_model, level = obs_net.level, name = name, str_dimXBC = str_dimXBC, meas = "triangles_at_c",)
			if not os.path.exists(full_path):
				sum_model_zl_pmatrix = ref_model.frmv_diag(pmatrix_name_dim(name, str_dimXBC))

				# select only the subgraph with the nodes with obs nodes with degree k_i > c
				for c in lin_space_c:
					obs_subnodes = np.where(obs_net.deg >= c)[0]
					sub_P = sum_model_zl_pmatrix[obs_subnodes][:,obs_subnodes]
					sum_model_triangle_c[sum_name_dim].append(tri_dens(sub_P))
				
				np.savetxt(full_path, np.array(sum_model_triangle_c[sum_name_dim]), fmt = "%.18f", delimiter=",")
			else:
				sum_model_triangle_c[sum_name_dim] = np.genfromtxt(full_path, delimiter = ",")

		# Plotting the Triangles
		fig, ax = plt.subplots(figsize = (12,7))

		# plot as first entry the observed triangles
		ax.scatter(lin_space_c, obs_net.triangle_c, marker = mpls.obs_marker, c = mpls.obs_color,
					s = mpls.obs_ms // 2, label = f"{obs_net.name}", zorder = len(names_dims)+1)

		# plot expected numb of triangles for each model
		for model_name_d in name_mark_color:
			lw_ampl = 1.2

			# replace the sum- in the label and remove -dimension if model_name_d starts with "fit", "CM", "degc"
			prefix = "sum-" if ref_model.fc_direction == "cg" else "fine-"
			label = model_name_d.replace(prefix, "") if only_summed else model_name_d

			mark, color = name_mark_color[model_name_d]
			
			# if ssm model for NetRec, remove the -1 e.g. sum-CM-1 -> CM
			# if ssm model for NodeEmb, remove the "maxl" if it exists, e.g. maxlMSM -> MSM but LPCA remains LPCA
			label = label.replace("-1", "") if ref_model.objective == "NetRec" else label.replace("maxl", "")
			
			# actual plot of the 
			ax.plot(lin_space_c, sum_model_triangle_c[model_name_d],
						marker = mark, 
						markerfacecolor = "none", 
						ms = 10 * lw_ampl, color = color,
						linestyle='dashed', linewidth=2,
						label = label)
   
		ax.set(xscale = "linear", yscale = "linear")
		# ax.set_title(f'Rich-Club-Triangle Density @ level {obs_net.level} - Summed models')
		ax.set_xlabel('CUT-OFF c')
		ax.set_ylabel('Triangle-Density at c')
		ax.legend()
		plt.tight_layout()

		save_fig(fig, full_path = plot_full_path, save = True)
		plt.close()

# plot the number of loops evolution
def plot_n_self_loops(obs_net, ref_model, models, dims, colors, markers):
	"""
	ref_model must be maxlMSM dimX 1 since we will plot the other coarse-grained models
	Only maxlMSM and degcMSM are plotted since the other models are not considering the self-loops
	"""
	# plots_dir = os.path.dirname(os.path.dirname(obs_net.plots_dir))
	full_path = ref_model.plots_dir_multi_models + f'/n_self_loops.pdf'
	if not os.path.exists(full_path):
		fig, ax = plt.subplots(figsize = (12,7))

		# how big to make the scatter plot points
		ms = 10
		total_levels = 4

		# create functions for number of nodes, obs and expected number of self-loops
		def exp_n_self_loops_at_(name, level, str_dimXBC = "dimX1"):
			pij = load_meas(ref_model = ref_model, level = level, name = name, str_dimXBC = str_dimXBC, meas = "pmatrix")
			return np.diag(pij).sum()
		
		# functions: observed nodes and self-loops
		n_nodes_at_ = lambda level: np.genfromtxt(obs_net.model_dir.replace(f"level{obs_net.level}", f"level{level}") + f"/deg.csv", delimiter = ",").size
		n_self_loops_at_ = lambda level: np.diag(np.genfromtxt(obs_net.model_dir.replace(f"level{obs_net.level}", f"level{level}") + f"/bin_adj.csv", delimiter = ",")).sum()

		# obtain the number of nodes and self-loops at each level
		n_nodes = [n_nodes_at_(level) for level in np.arange(total_levels)]
		n_self_loops = [n_self_loops_at_(level) / n_nodes[level] * 100 for level in np.arange(total_levels)] # 

		# function: expected number of self-loops at each level
		exp_n_self_loops = lambda name, str_dimXBC: [exp_n_self_loops_at_(name, level, str_dimXBC) / n_nodes[level] * 100 for level in np.arange(total_levels)]

		# function: plotting functions for the models
		plot_exp_n_loops = lambda model_name, str_dimXBC, marker: ax.loglog(n_nodes, exp_n_self_loops("sum-"+model_name, str_dimXBC), marker = marker, linestyle = "dashed", lw = 2, mfc = "None", color = colors[model_name], ms = ms, label = model_name.replace("sum-", "") + f"{str_dimXBC.replace('dimX','-')}")
		
		# plot the obs number of self-loops
		ax.loglog(n_nodes, n_self_loops, 'o', mfc = None, color = mpls.obs_color, label = 'Observed')

		for name, d, marker in zip(models, dims, markers):
			str_dimXBC = f"dimB{d[0]}/dimC{d[1]}" if name.endswith("LPCA") else f"dimX{d}"
			plot_exp_n_loops(name, str_dimXBC, marker)

		ax.set(xlabel = 'Number of Nodes', ylabel = 'Percentage of Self-Loops', 
				title = 'Percentage of Self-Loops for Summed Models', 
				xscale = "log", yscale = "log")
		ax.invert_xaxis()
		ax.legend()
		plt.close()

		save_fig(fig, full_path)

# Plot the evolution of AIC and BIC
def plot_AIC_BIC(obs_net, models, dims, markers, ref_model, colors, total_levels = 4):
	"""
	Plotting the AIC and BIC scores for every model and for each number of nodes
	"""
	# plots_dir = os.path.dirname(os.path.dirname(obs_net.plots_dir))
	# full_path = f'{plots_dir}/multi_models/avg_aic_bic_scores.pdf'
	full_path = ref_model.plots_dir_multi_models + f'/avg_aic_bic_scores.pdf'
	if not os.path.exists(full_path):
		print('-Plotting AIC and BIC scores',)

		fig, axs = plt.subplots(1, 2, figsize = (20,7))

		# how big to make the scatter plot points
		ms = 14

		# create functions for number of nodes, obs and expected number of self-loops
		n_nodes_at_ = lambda level: np.genfromtxt(obs_net.model_dir.replace(f"level{obs_net.level}", f"level{level}") + f"/deg.csv", delimiter = ",").size

		# create the aic and bic for a specific model
		aic_of_at_ = lambda level, name, str_dimXBC: load_meas(ref_model, level, name, str_dimXBC = str_dimXBC, meas = "avg_aic")
		bic_of_at_ = lambda level, name, str_dimXBC: load_meas(ref_model, level, name, str_dimXBC = str_dimXBC, meas = "avg_bic")
		
		# x-axis is the number of nodes for each level
		n_nodes = [n_nodes_at_(level) for level in np.arange(total_levels)]
		aics_of = lambda name, str_dimXBC: [aic_of_at_(level, name, str_dimXBC) for level in np.arange(total_levels)]
		bics_of = lambda name, str_dimXBC: [bic_of_at_(level, name, str_dimXBC) for level in np.arange(total_levels)]
		
		# y-axis is the aic and bic for each model to be split in 2 subplots
		lw = 2
		for mod, d, m in zip(models, dims, markers):
			str_dimXBC = f"dimB{d[0]}/dimC{d[1]}" if mod.endswith("LPCA") else f"dimX{d}"
			# colors are cycled throught the matplotlib ones. If custom, add color = colors[mod]

			axs[0].plot(n_nodes, aics_of(mod, str_dimXBC), linestyle = "dashed", marker = m, markerfacecolor = "none", lw = lw, ms = ms, label = mod+ f"-{d}")
			axs[1].plot(n_nodes, bics_of(mod, str_dimXBC), linestyle = "dashed", marker = m, markerfacecolor = "none", lw = lw, ms = ms, label = mod+ f"-{d}")

		axs[0].set(xlabel = 'Number of Nodes', ylabel = 'AIC', title = 'Numb of Nodes VS AIC', xscale = "log", yscale = "log")
		axs[0].invert_xaxis()
		axs[0].legend()
		
		axs[1].set(xlabel = 'Number of Nodes', ylabel = 'BIC', title = 'Numb of Nodes VS BIC', xscale = "log", yscale = "log")
		axs[1].invert_xaxis()
		axs[1].legend()
		
		fig.suptitle('Multiscale averaged AIC and BIC for Fitted Models')
		save_fig(fig, full_path)
		plt.close()
		
def plots_label(ref_model, model_name, d):
	mod_no_cgd = model_name.replace("sum-", "")
	label = mod_no_cgd.replace(f"-{d}", "") if ref_model.objective == "NetRec" else mod_no_cgd.replace("maxl", "") #+ f"-{d}"
	return label.replace(" ", "")

def plots_label_add_dim(ref_model, model_name, d):
	mod_no_cgd = model_name.replace("sum-", "")
	label = mod_no_cgd.replace(f"-{d}", "") if ref_model.objective == "NetRec" else mod_no_cgd.replace("maxl", "") + f"-{d}"
	return label.replace(" ", "")

def split_name_dim(sum_name_dim):

	# split sum_name_dim according to the name, e.g. "sum-LPCA-(1,1)" or "sum-maxlMSM-1"
	minus_splitting = sum_name_dim.split("-")
	name = "-".join(minus_splitting[:2])
	d = minus_splitting[-1]
	if "LPCA" in sum_name_dim:
		d = d[1:-1].split(",")

	return name, d


# plot the AUC ROC and AUC PRC scores increasing the numnbers of nodes
def plot_auc_roc_prc(obs_net, name_mark_color, ref_model, total_levels = 4, yscale = "log", xlabels = "levels"):
	"""
	Multi-Scale plotting of AIC and BIC scores for every model and for each number of nodes
	"""
	plt.rcParams.update({"font.size" : '21'})
	
	full_path = ref_model.plots_dir_multi_models + f'/auc_roc_prc_scores.pdf'

	if not os.path.exists(full_path):
		print('-Plotting the AUC ROC/PR for the summed model',)

		fig, axs = plt.subplots(1, 2, figsize = (14,8))

		# create functions for number of nodes, obs and expected number of self-loops
		if xlabels == "levels":
			xticks = np.arange(total_levels)
			xscale = 'linear'
		else:
			n_nodes_at_ = lambda level: np.genfromtxt(obs_net.model_dir.replace(f"level{obs_net.level}", f"level{level}") + f"/deg.csv", delimiter = ",").size
			xticks = [n_nodes_at_(level) for level in np.arange(total_levels)]
			xscale = 'log'


		# create the aic and bic for a specific model
		auc_roc_at_ = lambda level, name, str_dimXBC: load_meas(ref_model, level, name, str_dimXBC = str_dimXBC, meas = "auc_roc_prc")[0]
		auc_prc_at_ = lambda level, name, str_dimXBC: load_meas(ref_model, level, name, str_dimXBC = str_dimXBC, meas = "auc_roc_prc")[1]
		
		# x-axis is the number of nodes for each level
		auc_roc_of = lambda name, str_dimXBC: [auc_roc_at_(level, name, str_dimXBC) for level in np.arange(total_levels)]
		auc_prc_of = lambda name, str_dimXBC: [auc_prc_at_(level, name, str_dimXBC) for level in np.arange(total_levels)]
		
		# y-axis is the aic and bic for each model to be split in 2 subplots
		lw, ms = 0, 14
		for name_dim in name_mark_color: #mod, d, m, c in zip(models, dims, markers, colors):
			
			# how thick will be the ms
			# lw = 4 if mod.endswith("CM") else 2
			
			# find the label to plot
			# label = plots_label(ref_model, mod, d)

			name, d = split_name_dim(name_dim)
			m, c = name_mark_color[name_dim]

			# colors are cycled throught the matplotlib ones. If custom, add color = colors[mod_no_cgd]
			str_dimXBC = f"dimB{d[0]}/dimC{d[1]}" if name.endswith("LPCA") else f"dimX{d}"
			print(f'-auc_roc_of({name}, {str_dimXBC}): {auc_roc_of(name, str_dimXBC)}',)
			
			# remove "sum-" and add dim if not Net Rec
			_plot_label = plots_label(ref_model, name_dim, d)
			axs[0].scatter(xticks, auc_roc_of(name, str_dimXBC), marker = m, color = c, label = _plot_label, ec = "black", lw = 0.5, s = 3*ref_model_ms**2)
			axs[1].scatter(xticks, auc_prc_of(name, str_dimXBC), marker = m, color = c, label = _plot_label, ec = "black", lw = 0.5, s = 3*ref_model_ms**2)

		str_x_label = "Number of Nodes" if xlabels == "n_nodes" else "Levels"
		axs[0].set(xlabel = str_x_label, ylabel = 'AUC-ROC', xscale = xscale, yscale = yscale)
		axs[0].set_ylim(None, 1)
		axs[0].legend()
		
		axs[1].set(xlabel = str_x_label, ylabel = 'AUC-PR', xscale = xscale, yscale = yscale)
		axs[1].set_ylim(None, 1)
		# axs[1].legend()
		
		# invert the axis such that the number of nodes increases from left to right
		# as the levels 0, 1, ...
		if xlabels == "n_nodes":
			axs[0].invert_xaxis()
			axs[1].invert_xaxis()
		else:
			axs[0].set_xticks(xticks)
			axs[1].set_xticks(xticks)

		axs[0].set_axisbelow(True)
		axs[1].set_axisbelow(True)
		
		# fig.suptitle('Multiscale normalized ROC and Prec Rec for Summed Models')
		save_fig(fig, full_path)
		plt.close()
	plt.rcParams.update({"font.size" : mpls.global_font_size})

# Plot Cross Comparison among the models for the Conference Plots
def plots_dir_(ref_model, n_parents = 2):
	from pathlib import Path
	return str(Path(ref_model.plots_dir).parents[n_parents])

def strineq_meas(ref_model, name = None, level = None, str_dimXBC = None, net_meas = None, reduced_by = None):
	""" Load the observed and expected reduced measurement 
	Only works for summed models, e.g. "sum-CM" since the dict is stored in that folder
	Note: the dictionary is of the kind 
	{"deg" : {"sum_model" : [], ref_model : []},
	"annd" : {"sum_model" : [], ref_model : []},
	"cc" : {"sum_model" : [], ref_model : []},}
	"""
	from pickle import load

	# if you specify a name, load only sum_model. Otherwise, load sum_model and ref_model
	if name == None:
		name = ref_model.name
	if level == None:
		level = ref_model.level

	# find the reduced label
	if reduced_by != "None":
		reduced_by = f"reduced_by_{reduced_by}_"
	else:
		reduced_by = ""

	if ref_model.kind == 'exp':
		# load the strienq_meas_dicts directory for the name model, e.g. sum-maxlMSM
		strineq_dir = full_path_retriever(ref_model, level, name, str_dimXBC, meas = None) + "/strineq_meas_dicts"

		# load the mod_dict
		dict_path = f"{strineq_dir}/{reduced_by}deg_annd_cc.pkl"
		with open(dict_path, "rb") as f:
			mod_dict = load(f)

		# get the model name with prefix
		fc_name = "sum_model" if ref_model.fc_direction == 'cg' else "fine_model"

		# the nest for-loops return "deg", "annd" --> [[deg_ref_model], [deg_sum_model], [annd_ref_model], [annd_sum_model], [cc_ref_model], [cc_sum_model]]
		return np.squeeze([mod_dict[m][model_name] for m in net_meas for model_name in [fc_name,"ref_model"]])

	# do the same for observed measurement
	elif ref_model.kind == 'obs':

		obs_dict_path = f"{ref_model.model_dir.replace(f'level{ref_model.level}', f'level{level}')}/strineq_meas_dicts/{reduced_by}deg_annd_cc.pkl"
		with open(obs_dict_path, "rb") as f:
			obs_dict = load(f)
		
		# when "deg", squeeze the [[deg]]
		obs_meas = np.squeeze([obs_dict[m] for m in net_meas])

		return obs_meas
	

def net_meas_plot(ax, obs_net, ref_model, models, dims, colors, markers, net_meas = "cc"):
	"""
	Plot CC for LPCA and MSM models at a fixed scale, e.g. level 0
	ref_model must be maxlMSM dimX 1 since we will plot the other coarse-grained models
	Only maxlMSM and degcMSM are plotted since the other models are not considering the self-loops
	"""
	from ..plots.obs_vs_rec_meas import _inset_pmatrix, _compute_hist2d
	from matplotlib import ticker


	# how big to make the scatter plot points
	ms = 40
	total_levels = 4
	axis_scale = "linear"

	
	# add sum- to highlight the coars-graining procedure
	sum_flag = "sum-" if obs_net.level > 0 else ""

	# function: plotting functions for the models
	# label_ = lambda mode_name: "MSM" if mode_name.endswith("MSM") else mode_name

	plot_cc_of_ = lambda ax, obs_meas, exp_meas, marker, model_name, d: ax.scatter(obs_meas, exp_meas, marker = marker, c = colors[model_name], s = ms, label = f"{sum_flag}"+plots_label_add_dim(ref_model, model_name, d))
	
	# upper-triangular indices and sum_flag to plot the "sum" label only for model.level > 0
	triu_idx = np.triu_indices(obs_net.n_nodes, k = 1)			
	load_pmatrix = lambda model_name, str_dimXBC: load_meas(ref_model, obs_net.level, model_name, str_dimXBC, meas = "pmatrix")[triu_idx]
	load_mod_meas = lambda model_name, str_dimXBC: load_meas(ref_model, obs_net.level, model_name, str_dimXBC, meas = net_meas)
	load_reduced_by = lambda model_name, str_dimXBC: load_meas(ref_model, obs_net.level, model_name, str_dimXBC, meas = "reduced_by.txt") 

	# find the reduced label
	# reduced_by = load_meas(ref_model, 0, name, str_dimXBC, meas = "reduced_by.txt")
	
	# plot the meas and the inset
	for i, (name, d, marker) in enumerate(zip(models, dims, markers)):
		
		# load the observed and expected net_meas
		str_dimXBC = f"dimB{d[0]}/dimC{d[1]}" if name.endswith("LPCA") else f"dimX{d}"
		
		# load the observed and expected measurements without reducing
		reduced_by = load_reduced_by(name, str_dimXBC)
		if reduced_by == 'None':
			
			if net_meas == "deg":
				obs_meas = obs_net.deg 
			elif net_meas == "annd":
				obs_meas = obs_net.annd
			else:
				obs_meas = obs_net.cc
			
			# load the model one
			exp_meas = load_mod_meas("sum-"+name, str_dimXBC)
		
		# for all the other models load the reduced measurements
		else:
			obs_meas = strineq_meas(obs_net, net_meas = [net_meas], reduced_by=reduced_by)
			exp_meas = strineq_meas(ref_model, name = "sum-"+name, level = obs_net.level, str_dimXBC = str_dimXBC, net_meas=[net_meas], reduced_by=reduced_by)[0]
		
		# plot the identity line
		ax[i].scatter(obs_meas, obs_meas, marker = 'o', s = ms, c = mpls.obs_color, label = 'Observed')
		
		# plot the expected VS the observed
		plot_cc_of_(ax[i], obs_meas, exp_meas, marker, name, d)
		
		# create the histogram for the pmatrix and plot it
		axis_scales = "linear"
		upper_zlP = load_pmatrix(name, str_dimXBC)
		upper_sum_zlP = load_pmatrix("sum-"+name, str_dimXBC)

		# plot the insets with the pmatrix
		# H, xedges, yedges = _compute_hist2d(x = upper_zlP, y = upper_sum_zlP, bins = 30)
		# _inset_pmatrix(ax[i], H, xedges, yedges, axis_scales = axis_scales, bbox_to_anchor = (.74, .05))


		ax[i].set(xlabel = net_meas.upper(), ylabel = r'$\langle$' +  f"{net_meas.upper()}" + r'$\rangle$', 
				title = f'Level {obs_net.level}', xscale = axis_scale, yscale = axis_scale)
		ax[i].legend(loc = 0), ax[i].grid(False)
		
		
		# change x- and y-ticks if needed
		# find the minimum and maximum values of the observed and expected measurements
		x_min, x_max = np.min([obs_meas, exp_meas]), np.max([obs_meas, exp_meas])

		# if the difference lays one order of magnitude and the axis_scale is log, erase the minor ticks and place n_bins
		if (x_max - x_min) / x_min < 1 and axis_scale == 'log':
			locator = ticker.MaxNLocator(nbins=3)
			ax[i].xaxis.set_major_locator(locator)
			ax[i].yaxis.set_major_locator(locator)
			ax[i].xaxis.set_minor_locator(ticker.NullLocator())
			ax[i].yaxis.set_minor_locator(ticker.NullLocator())
		
		# set the minimum of the plot as the minimum value greater than zero
		if name.endswith("MSM"):
			MIN_exp_meas = min(obs_meas[obs_meas > 0] * 0.95)

		# To have a quantitative comparison, we will dispaly the RE among the ||sum_P|| and the ||A||
		upper_zlA = obs_net.zl_bin_adj[triu_idx]
		norm_diff = np.linalg.norm(upper_sum_zlP - upper_zlA) / np.linalg.norm(upper_zlA)
		print(f'-name, level, norm_diff: {name, obs_net.level, norm_diff}',)
			
	# for i in range(len(models)):
	#     if obs_net.level == 0:
	#         ax[i].set_ylim(MIN_exp_meas, 1.05)
	#     else: ax[i].set_ylim(MIN_exp_meas, 1.01)

def plot_cross_comparison(ref_model, models, dims, markers, colors, levels, kwargs, net_meas = "cc"):
	""" This function provides the comparison of the two SUMMED models over the net_meas (e.g. the Clustering Coefficient) """

	def model_dims(i,dims):
		""" 
		Convert the dimension tuple, or int, to string for a well-formatted title.
		Examples:
		LPCA -> (1,1) -> 11
		MSM -> 2 -> 2
		"""
		model_i = models[i]
		if model_i.endswith(("LPCA", "maxlMSM")):
			dim_i = dims[i]
			# this is for LPCA dims
			if not isinstance(dim_i, int):
				return model_i + "_" + ''.join(map(str, dim_i))
			# for MSM dims
			else:
				return model_i + "_" + str(dim_i)
		else:
			return model_i
	
	from ..graphs.Undirected_Graph import Undirected_Graph

	# define the directory for the plots
	fig_dir = ref_model.plots_dir_multi_models #plots_dir_(ref_model) + "/multi_models"
	os.makedirs(fig_dir, exist_ok = True)

	# create a string by joining the models and the dims if models are not LCPA and maxlMSM
	str_models_d = '_'.join([model_dims(i,dims) for i in range(len(models))])
	path_levels = "".join([str(i) for i in levels])
	full_path = f'{fig_dir}/cross_comparison/levels{path_levels}/{net_meas.upper()}_{str_models_d}.pdf'

	# start with plots
	if not os.path.exists(full_path):
		fig, axs = plt.subplots(len(levels),len(models),figsize = (24,14)) # (20,12)
		
		# for every fixed level at row 0,1,2,... plot on the columns the different models
		for i, level in enumerate(levels):
			kwargs.update({
						"level" : level,
						})

			net = Undirected_Graph(**kwargs)

			# plot in the i-th row the net_meas VS the expected net_meas for the different models
			net_meas_plot(axs[i], net, ref_model, models, dims, colors, markers, net_meas = net_meas)


		# save it
		save_fig(fig, full_path)
		plt.close()

def norm_rec_acc(obs_net_meas, lbub_ci): #model, levels, str_dimXBC, meas = "deg"):
	"""Function to count how many times obs_net_meas is included in the ensemble disp.interval of the network measurement, e.g. deg"""

	# select only the lower-bound and upper-bound from the enseble disp.interval the network measurement, e.g. deg
	# then, sum only the true values in the interval
	return np.sum((obs_net_meas >= lbub_ci[0]) & (obs_net_meas <= lbub_ci[1])) / obs_net_meas.size

def get_rec_acc_models(obs_net, ref_model, name_mark_color, total_levels, confidence, net_meas = ["deg", "annd", "cc"], ):
	""" Fill the dictionary with the rec_acc (RA) for the different models and levels
	The dictionary is {"CM" : [RA_deg_0, RA_deg_1, ...], [RA_annd_0, RA_annd_1, ...], [RA_CC_0, RA_CC_1, ...]}
	where 0, 1, ... are the levels
	"""
	from pathlib import Path
 
	dims = [x.split("-")[-1] for x in name_mark_color.keys()]

	# set the directory to save the dictionary
	par_jumps = 4 if ref_model.name.endswith("LPCA") else 3
	directory = Path(ref_model.model_dir).parents[par_jumps] / "multi_models"
	directory.mkdir(parents=True, exist_ok=True)
	full_path = directory / f"rec_acc_{ref_model.objective}_conf{confidence}.pkl"
	if not full_path.exists():

		# container of avg_rec_accuracy
		dict_rec_acc = {}
		levels = np.arange(total_levels)

		for meas in net_meas:	
			
			# create a dictionary with name and dims to host rec_accuracy
			model_norm_rec_acc = dict(zip(name_mark_color.keys(), [[]]*len(name_mark_color)))
			# model_norm_rec_acc = { name+f"-{dim}" : [] for name, dim in zip(models, dims)}
			
			# cycle over the levels such that the obs_net_meas is the same for all the models
			for level in levels:
				
				# create the obs_net_meas to be tested, e.g. DEG
				obs_net_meas = load_meas(obs_net, level, name = obs_net.name, meas = meas, ensemble_avg = False)
				
				# for every model check if it is included in the induced ci
				for model_name_d in name_mark_color:
					model_name, dim = split_name_dim(model_name_d)

					# load the mean and errors (wrt to the mean)
					str_dimXBC = f"dimB{dim[0]}/dimC{dim[1]}" if model_name.endswith("LPCA") else f"dimX{dim}"
					_, lb_ci, ub_ci = load_meas(ref_model, level, name = model_name, str_dimXBC = str_dimXBC, meas = meas + f"_conf{confidence}", ensemble_avg = True)
					
					# create the lower and higher dispersion intervals
					lbub_ci = np.vstack((lb_ci, ub_ci))

					# fixed a level append the rec_acc for every model
					model_norm_rec_acc[model_name_d].append(norm_rec_acc(obs_net_meas, lbub_ci))

					# update the dict_rec_acc with the rec_acc for the specific model and levels
					if level == levels[-1]:
						dict_rec_acc.setdefault(model_name_d, []).append(model_norm_rec_acc[model_name_d])

		# save the dictionary
		from pickle import dump
		with open(full_path, "wb") as f:
			dump(dict_rec_acc, f)
	
	# if already existing import it
	else:
		from pickle import load
		with open(full_path, "rb") as f:
			dict_rec_acc = load(f)

	return dict_rec_acc

def _plot_inset_pmatrices_in_net_meas(fig, ax, ref_model, name, dim, level, bbox_to_anchor = (1.025, .65)):
	from ..plots.obs_vs_rec_meas import _inset_pmatrix, _compute_hist2d
	""" Plot the inset pmatrices in the net_meas by level grid """
	# select the width of the inner axis, the markersize and the alpha
	from ..utils.helpers import str_dimXBC_
	width = 100
	ms, alpha = 1, 0.3

	str_dimXBC = str_dimXBC_(name, dim)

	dir_ref_pmatrix = full_path_retriever(ref_model, level, name = name.replace("sum-", ""), str_dimXBC = str_dimXBC, meas = "pmatrix.csv") #f"{os.path.dirname(ref_model.model_dir)}/level{level:g}/zl_pmatrix.csv"
	dir_sum_pmatrix = full_path_retriever(ref_model, level, name = name, str_dimXBC = str_dimXBC, meas = "pmatrix.csv") #f"{os.path.dirname(sum_model.model_dir)}/level{level:g}/zl_pmatrix.csv"

	# calculate the triu pmatrices
	ref_zl_pmatrix = np.genfromtxt(dir_ref_pmatrix, delimiter = ",")
	n_nodes = ref_zl_pmatrix.shape[0]
	triu_idx = np.triu_indices(n_nodes, k = 1)

	triu_ref_zl_pmatrix = ref_zl_pmatrix[triu_idx]
	triu_sum_zl_pmatrix = np.genfromtxt(dir_sum_pmatrix, delimiter = ",")[triu_idx]

	axis_scales = "linear"
	H, xedges, yedges = _compute_hist2d(x = triu_ref_zl_pmatrix, y = triu_sum_zl_pmatrix, bins = 30)

	_inset_pmatrix(ax, H, xedges, yedges, width=1.4, height=1.4, bbox_to_anchor = bbox_to_anchor, axis_scales = axis_scales)

def plots_net_meas_by_level(obs_net, ref_model, levels, name, dim):
	"""
	Plot the network measurements for the observed and expected measurements for the different levels 
	"""

	directory = ref_model.plots_dir_multi_models+f'/net_meas_by_level'
	os.makedirs(directory, exist_ok = True)
	full_path = directory + f'/{name}.pdf'
	if not os.path.exists(full_path):

		# measurements to be plotted
		measurements = ["deg", "annd", "cc"]

		# dimensions
		str_dimXBC = f"dimB{dim[0]}/dimC{dim[1]}" if name.endswith("LPCA") else f"dimX{dim}"

		# create the figure and the grid
		fig = plt.figure(figsize = (23,15))
		gs = fig.add_gridspec(len(measurements), len(levels), hspace=0.2, wspace=0.2)
		axs = gs.subplots(sharex='col') 
		
		# plot the measurements
		for j, level in enumerate(levels):

			# load the reduce_by label
			reduced_by = load_meas(ref_model, level, name, str_dimXBC, meas = "reduced_by.txt")

			# load the observed degrees
			obs_meas = strineq_meas(obs_net, obs_net.name, level, net_meas = measurements, reduced_by = reduced_by)
			
			norm = lambda x: x

			# normalize the measurements
			# normalized = True
			# if normalized:
			# 	n_nodes = load_meas(ref_model, level, name, str_dimXBC, meas = "deg.csv").size
			# 	div_n_nodes = np.array([n_nodes - 1, n_nodes - 1, 1])[:, None]
			# 	norm = lambda x: x / div_n_nodes #/ np.mean(x, axis = 1)[:,None]

			# normalize the observed measurements
			norm_obs = norm(obs_meas)

			# load the summed model measurements
			# ref_model.name
			exp_meas = strineq_meas(ref_model, name, level, net_meas = measurements, reduced_by = reduced_by)
			norm_fit = norm(exp_meas[1::2])
			norm_sum = norm(exp_meas[::2])

			for i, meas in enumerate(measurements):

				axis_scale = 'log'

				# load the meas from the observed obs_network
				# i = 0, deg; i = 1, annd; i = 2, cc --> norm_obs[0] = obs deg
				axs[i,j].scatter(norm_obs[0], norm_obs[i], marker = 'o', color = mpls.obs_color, s = mpls.obs_ms, label = 'Observed')
				axs[i,j].scatter(norm_fit[0], norm_fit[i], marker = '+', color = mpls.ref_model_color, s = mpls.obs_ms, label = f'{set_name_for_plots(name, ref_model)}')
				axs[i,j].scatter(norm_sum[0], norm_sum[i], marker = 'x', color = mpls.sum_model_color, s = mpls.obs_ms, label = 'Summed')
				
				if j == 0:
					# add name of measurements for the y-axis only in the first row
					axs[i,0].set_ylabel(meas.upper(), labelpad=30)


				if i == 0:
					# plot as insets the summed VS fit probabilities
					_plot_inset_pmatrices_in_net_meas(fig, axs[0,j], ref_model, name, dim, level, bbox_to_anchor=(.59, .07))
					# first row add the title level 0 -> 0
					axs[0, j].set_title(f"Level {level}") #rf'Level 0 $\to$ {j}')
					if j == 0:
						# first plot (0,0) --> add the legend
						axs[i,j].legend(fontsize = 12)


				# in the last row add the x-axis label
				if i == len(measurements) - 1:
					axs[i,j].set_xlabel(measurements[0].upper())

				# remove only the x-ticks since one has only the obs_deg on the x-axis
				# the y-axis changes from level 0 to higher levels
				if i < len(measurements) - 1:
					axs[i,j].xaxis.set_visible(False)
				
				# set x-axis and y-axis scales
				axs[i,j].set(xscale = axis_scale, yscale = axis_scale)
				axs[i,j].grid(False)
				
				# set scalar formatter with style 'sci' (as option)
				# if i == 2 and j == 3: print(f'\n-Here: {level, meas}',)
				set_scalarformatter(axs[i,j], norm_obs[0], norm_obs[i], norm_sum[0], norm_sum[i], style = 'sci')

		# fig.suptitle(f'MultiScale Net. Meas. for {name} @ {obs_net.dataset_name}', y = .95)

		# save the figure
		save_fig(fig, full_path)

		# close the figure
		plt.close(fig)


def plot_rec_acc_by_meas(obs_net, ref_model, name_mark_color, total_levels, confidence):
	"""
	Function to plot the accuracy of the network measurements for the different models and levels
	The average will be the average and c.i. are over the reconstruction accuracy for the different levels: 
	the lower the avg, the worse the model but also the bigger the c.i. the worse the model.
	"""

	names_no_dims = ["-".join(x.split("-")[:-1]) for x in name_mark_color.keys()]
	dims = [x.split("-")[-1] for x in name_mark_color.keys()]


	# full_path = plots_dir_(ref_model) + f"/multi_models"
	full_path = ref_model.plots_dir_multi_models+f"/rec_acc/conf{confidence}_by_meas.pdf"

	width = 0.03  # the width of the bars
	sep_among_meas = 10 * width # separation among the network measurements
		
	if not os.path.exists(full_path):
		print(f'-Plotting the Reconstruction Accuracy for the summed model',)

		import numpy as np

		# evaluate only the summed MSM models
		n_models = len(names_no_dims)

		net_meas = ["deg", "annd", "cc"]
		n_meas = len(net_meas)
		x = np.arange(n_meas)
		# consider all this network measurements
	
		# get the dictionary with the rec_acc (RA) for the different models and levels
		dict_rec_acc = get_rec_acc_models(obs_net, ref_model, name_mark_color, total_levels, confidence)
		
		# plot them          
		fig, ax = plt.subplots(figsize = (12,7))

		# create the linspace which depends on model idx (idx) and the net_idx
		width_per_model = width * total_levels
		nlspace_idx_nidx = lambda idx, net_idx: \
							sep_among_meas * net_idx + \
							np.linspace(
							width_per_model * (idx + net_idx * n_models), 
							width_per_model * (idx + net_idx * n_models + 1), 
							total_levels + 1
							)[:total_levels]

		for idx, (name, rec_acc) in enumerate(dict_rec_acc.items()):
			# it can be that name has spaces
			name = name.replace(" ", "")
   
			# ravel rec_acc
			rec_acc = np.array(rec_acc).ravel()
			x = np.concatenate([nlspace_idx_nidx(idx, j) for j in range(n_meas)])

			viz_name = set_name_for_plots(name, ref_model)
			print(f'-{name}, rec_acc: {rec_acc}',)
			_ = ax.bar(x, rec_acc, width, label = viz_name, align = "edge", edgecolor = "black", color = name_mark_color[name][1], linewidth = 1)
			# colors[idx % len(models)]

		# for the x-axis ticks for-loops over the separation
		label_centers = []
		label_center = 0
		label_center = width_per_model * n_models / 2
		for i in range(n_meas):
			label_centers.append(label_center)
			label_center += sep_among_meas + width_per_model * n_models
		ax.set_xticks(label_centers, [str_.upper() for str_ in net_meas])
		
		# Add some text for labels, title and custom x-axis tick labels, etc.
		ax.set_ylabel('Rec. Accuracy')
		# ax.set_title('Reconstruction Accuracy for the Summed models')

		# set axes properties
		ax.legend(loc='upper left', ncol=3)
		ax.set_ylim(0, 1.4)
		ax.grid(False)

		save_fig(fig, full_path)
		plt.close()

def plot_rec_acc_by_level(obs_net, ref_model, name_mark_color, total_levels, confidence):
	""" Plots the Rec. Accuracy ordered by level """

	full_path = ref_model.plots_dir_multi_models + f"/rec_acc/conf{confidence}_by_level.pdf"
	os.makedirs(os.path.dirname(full_path), exist_ok = True)
	if not os.path.exists(full_path):

		# load the rec.accuracies to be plotted
		dict_rec_acc = get_rec_acc_models(obs_net, ref_model, name_mark_color, total_levels, confidence)
		
		n_models = len(name_mark_color)
		n_meas = 3
		
		# esthetic parameters
		width_bar = 0.2
		width_meas = width_bar * 2
		width_level = width_meas * 3
		width_models = n_models * width_bar
		starting_baricenter = 0

		# find the shifts of the baricenter across measurements
		# from starting baricenter add the total meas shift, i.e. width_models + width_meas
		bari_meas =  [starting_baricenter + j * (width_models + width_meas) for j in np.arange(n_meas)]

		# as soon as the the bari reached the last meas it has to be shifted by width models and wdith_levels
		shift_bulk_model_levels = width_models + width_level

		# simply multiply by the needed level l the overall occupied space, i.e. bari_meas[-1] + shift_bulk_model_levels
		bari_level = np.ravel([bari_meas + l * (bari_meas[-1] + shift_bulk_model_levels) for l in np.arange(total_levels)])

		# center in 0, the first_bar starts at (n_models - 1)//2 jumps (long width_bar) on the left, e.g. 5 models + 0.2 width_bar --> -0.4
		idx_center_models = (n_models - 1)//2 * width_bar
		center_first_bar = starting_baricenter - idx_center_models

		fig, ax = plt.subplots(figsize = (24,7))
		for i, model_name_d in enumerate(dict_rec_acc):
			init_idx = center_first_bar + i * width_bar

			lev_idx = bari_level + init_idx

			# load dict_rec_acc
			model_rec_acc = np.array(dict_rec_acc[model_name_d]).ravel(order = 'F')

			# remove the "-1" if objective is NetRec
			viz_name = set_name_for_plots(model_name_d, ref_model)

			# plot the bars
			ax.bar(lev_idx, model_rec_acc, width_bar, label = viz_name, align = "center", facecolor = name_mark_color[model_name_d][1], edgecolor = "black", linewidth = 1)


			ax.set_xticks(bari_level, labels = ["DEG", "ANND", "CC"]*total_levels)
			# ax.set_xticklabels()
			ax.tick_params(axis='x', which='both', length=0)

		# Add another x-labels grouping
		secax = ax.secondary_xaxis(-0.1)
		secax.set_ticks([bari_level[i * n_meas:(i + 1) * n_meas].mean() for i in range(total_levels)])
		secax.set_xticklabels([f'Level {i}' for i in range(total_levels)])
		
		# remove x_ticks and fake the appearance of the x-axis in white color
		secax.tick_params(axis='x', which='both', length=0)
		secax.spines['bottom'].set_color('white')

		# Add some text for labels, title and custom x-axis tick labels, etc.
		ax.set_ylabel('Rec. Accuracy')

		# set axes properties
		ax.legend(loc='upper left', ncol=3)
		ax.set_ylim(0, 1.3)
		ax.grid(False)

		save_fig(fig, full_path)
		plt.close()


def plot_sum_vs_cg_pmatrix(obs_net, ref_model, level, class_models = "fitn_models"):
	'''Save plot for cg_pmatrix vs sum_pmatrix for all models for all the levels'''

	full_path = ref_model.plots_dir_multi_models + f"/sum_vs_cg_pmatrix/{class_models}/level{level}.pdf"
	
	if not os.path.exists(full_path):
		str_dimXBC = lambda name, d: f"dimB{d[0]}/dimC{d[1]}" if name.endswith("LPCA") else f"dimX{d}"
		pij_extraction = lambda name, d: load_meas(ref_model = ref_model, level = 0, name = name, str_dimXBC = str_dimXBC(name, d), meas = "pmatrix")

		def prod_pij(pij, I, J, ref_model):
			
			i = ref_model.toplab2botlab[I]
			j = ref_model.toplab2botlab[J]

			# Use broadcasting to get the Cartesian product of indices
			i, j = np.meshgrid(i, j, indexing='ij')

			# Now I and J are arrays that contain all combinations of iinI and jinJ
			# Flatten I and J to use them as indices
			i = i.flatten()
			j = j.flatten()

			return np.prod(1 - pij[i,j])

		ref_model.toplab2botlab = ref_model.isource_2_itarget(obs_net, lsour = level, ltar = 0)

		
		colors = [mpls.ref_model_color, mpls.sum_model_color]
		dims = [1,1]
		if class_models == "fitn_models":
			model_names = ["fitnCM", "fitnMSM"]
		elif class_models == "degc_models":
			model_names = ["CM", "degcMSM"]
		else:
			model_names = ["LPCA", "maxlMSM", ]
			dims = [(1,1),1,]
			# colors = ["tab:orange", "#0714b1" , "#2294a4"]

		markers = ["."]*len(model_names)
		n_nodes = load_meas(ref_model = ref_model, level = level, name = model_names[0], str_dimXBC = str_dimXBC(model_names[0], dims[0]), meas = "X").shape[0]
		name2cg_pmatrix = {name : np.zeros((n_nodes, n_nodes)) for name in model_names}
		uptri = np.triu_indices(n_nodes, k=1)

		for name, d in zip(model_names, dims):    
			pij = pij_extraction(name, d)
			for I, J in zip(*uptri):
				name2cg_pmatrix[name][I,J] = 1 - prod_pij(pij, I, J, ref_model)

		fig, ax = plt.subplots(figsize = (6,6))
		for name, d, marker, col in zip(model_names, dims, markers, colors):

			# select the already summed version of the proability matrix
			sum_pIJ = load_meas(ref_model = ref_model, level = level, name = "sum-"+name, str_dimXBC = str_dimXBC(name, d), meas = "pmatrix")
			
			# scatter the plot realizing that fitnMSM is the identity line. Thus, no need for other references
			# label = f'{name}-{d}' if name == "LPCA" else f'{name}'
			label = f'{name.replace("maxl", "")}-{d}'.replace(" ", "") if ref_model.objective == "NodeEmb" else f'{name}' #plots_label(ref_model, name, d)
			ax.scatter(name2cg_pmatrix[name][uptri], sum_pIJ[uptri], marker = marker, label = label, color = col, zorder = 2, alpha=0.3)

			# if name == "fitnMSM": break

		axis_scale = 'linear'
		ax.set(xscale = axis_scale, yscale = axis_scale, xlabel = 'Coarse-Grained P', ylabel = 'Summed P', )
		# title = f'Summed VS Coarse-Grained Pmatrix @ level {level}')
		# ax.set_xlim(sum_pIJ[uptri].min(), 1.4)
		# ax.set_ylim(sum_pIJ[uptri].min(), 1.4)

		leg = ax.legend(loc = 0, markerscale = 3)
		for lh in leg.legend_handles:
			lh.set_alpha(1)
		ax.grid(False)


		# inset Axes....
		# Uncomment if needed
		# x1, x2, y1, y2 = .3, 1.01, .3, 1.01 # subregion of the original image
		# axins = ax.inset_axes(
		#     [0.1, 0.65, .5, .3],
		#     xlim=(x1, x2), ylim=(y1, y2), xticks = [], yticks = [], xscale = axis_scale, yscale = axis_scale)

		# for name, d, marker, col in zip(model_names, dims, markers, colors):

		#     # select the already summed version of the proability matrix
		#     sum_pIJ = load_meas(ref_model = ref_model, level = level, name = "sum-"+name, str_dimXBC = str_dimXBC(name, d), meas = "pmatrix")

		#     # scatter the plot realizing that fitnMSM is the identity line. Thus, no need for other references
		#     axins.scatter(name2cg_pmatrix[name][uptri], sum_pIJ[uptri], marker = marker, color = col, label = f'{name}-dim{d}', zorder = 2, alpha=0.1)

		# axins.set_xticks([], minor = True)
		# axins.set_yticks([], minor = True)
		# ax.indicate_inset_zoom(axins, edgecolor="black", alpha = 0.3)
		
		save_fig(fig, full_path)
		plt.close()

def plots_rel_err_n_edges_across_levels(sum_model, name_mark_color, dims, levels):
	""" 
	Plot the relative error across the levels either for the summed and fined models 
	NB: model_names must already carry "sum-" or "fine-" prefix
	"""
	plt.rcParams.update({"font.size" : '21'})

	from pathlib import Path

	quantity_name = "rel_err_n_edges_across_levels"
	full_path = sum_model.plots_dir_multi_models + f"/{quantity_name}.pdf"
	names_no_dims = ["-".join(x.split("-")[:-1]) for x in name_mark_color.keys()]

	if not os.path.exists(full_path):
		from ..utils.helpers import fc_title, str_dimXBC_

		lw_ampl = 1.2
	
		# Create subplots
		fig, ax = plt.subplots(figsize=(11, 6))

		scale = 100
		for i, sum_name_dim in enumerate(name_mark_color):
			
			name, d = split_name_dim(sum_name_dim)
			str_dimXBC = f"dimB{d[0]}/dimC{d[1]}" if name.endswith("LPCA") else f"dimX{d}"
			
			# create the function to fish the proper n_edges_across_levels
			if sum_model.objective == "NodeEmb":
				n_parents = 4 if sum_model.name.endswith("LPCA") else 3
				init_cond = sum_model.model_dir.split("/")[-2]
				n_edges_path = str(Path(sum_model.model_dir).parents[n_parents] / name / str_dimXBC / init_cond / (quantity_name + ".csv"))
			else:
				n_edges_path = os.path.dirname(sum_model.model_dir).replace(sum_model.name, name)+f"/{quantity_name}.csv"
			
			n_edges_across_levels = load_array(n_edges_path)
			
			# set the levels and the rel_error on edges
			# edges = n_edges_across_levels(model, dims[i])

			# find the prefix to be removed from the model name
			# fc_label = sum_model.name.split("-")[0]+"-" if "-" in sum_model.name else ""
			_plot_label = plots_label(sum_model, sum_name_dim, d)
			
			# plot the edges
			marker, color = name_mark_color[sum_name_dim]
			ax.scatter(levels, n_edges_across_levels * scale,
						marker=marker, c = color, s = 2 * mpls.obs_ms, #fc = "none", , 
						label=_plot_label)
		

		# Customize the plot
		ax.set(xlabel = 'Levels', ylabel = 'Signed Rel.Err. L (\%)')
		ax.set_xticks(levels)

		# ax.set_title(f'RelErr of the Number of Edges for {fc_title(sum_model)} models', y = 1.03)
		ax.legend(markerscale = 1)
		ax.grid(False)

		if sum_model.fc_direction == "fg":
			ax.invert_xaxis()

		# set horizontal grey lines to inspect the (percentage) relative error
		# use [1:-1] in the fc case to avoid shifted major ticks after the plotting of horizontal lines
		# otherwise try: y_ticks = ax.get_yticks().copy(), (after plotting) --> ax.set_yticks(y_ticks)
		
		for h in ax.get_yticks()[1:-1]:
			ax.hlines(h, 0, len(levels) - 1, color='lightgrey', linestyle='--', linewidth=1, zorder=0)
		

		# Save the plot in multi-models folder
		save_fig(fig, full_path)

		plt.close()

	plt.rcParams.update({"font.size" : mpls.global_font_size})
