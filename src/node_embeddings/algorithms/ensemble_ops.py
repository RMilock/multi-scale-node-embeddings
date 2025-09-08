from ..utils.helpers import *
from utils import *
from Undirected_Graph import *

class ensemble_ops():
    def __init__():
        pass

    def deg_filtered_annd(self):
        """
        When generating an ensemble of network it may happen that the deg is zero.
        Therefore, compute the func only if pos_degs are met and set to zero all the measurments regarding that node.
        Since, the number of networks will be high, statistically there would be a link there.

        The trick to set to np.nan the arr is beacause, while evaluating the ANND for each graph in the ensemble, the np.nanmean will ignore the np.nan values.
        Indeed, we should average only over the nodes with mathematically defined ANND, i.e. the ones without degree > 0.
        """

        idx = self.deg > 0
        if np.any(~idx):
            arr = np.empty(self.n_nodes) * np.nan
            
            filtered_deg = self.deg[idx]
            arr[idx] = self.zl_bin_adj[idx][:, idx] @ filtered_deg / filtered_deg
            return arr
        else:
            return self.zl_bin_adj @ self.deg / self.deg
    
    def deg_filtered_cc(self):
        """
        check if all the deg > 0,1 ---> compute cc.
        Otherwise, 
        1) if i has deg[i] = 1, cc[i] = 1 to preserve the dataset. Otherwise, we could choose to completely remove the nodes
        2) if deg[i] = 1, same.

        This Warning is likely to happen while sampling with the p_ij. 
        Therefore, there is the hope that the ens_avg_cc is not zero
        """

        idx_e1 = self.deg == 1
        idx_g1 = self.deg > 1

        # enter if at least one of the two was True
        if np.any(~idx_g1):
            arr = np.empty(self.n_nodes) * np.nan
            
            # hard code to cc = 1 the nodes with deg = 1
            arr[idx_e1] = 1

            # compute the cc for the nodes with deg > 1
            filtered_deg = self.deg[idx_g1]
            pos_zl_bin_adj = self.zl_bin_adj[idx_g1][:, idx_g1]
            num_triangles = (self.frmv_diag(pos_zl_bin_adj @ pos_zl_bin_adj) @ pos_zl_bin_adj).diagonal()
            arr[idx_g1] = num_triangles / list(map(lambda k: self.n_wedges(k), filtered_deg))

            return arr
        else:
            num_triangles = (self.frmv_diag(self.zl_bin_adj @ self.zl_bin_adj) @ self.zl_bin_adj).diagonal()
            n_wedges = list(map(lambda k: self.n_wedges(k), self.deg))
            return num_triangles / n_wedges
    
    def _set_ens_deg_annd_cc(self, n_samples = 10, confidence = 0.95):
        """ Set the ENSEMBLE quantities for deg, annd, and cc
        This function is not needed for the Undirected_Graph but will be shared by all the models. Thererfore, it is here.
        """
        
        import scipy.stats as st
        from Undirected_Graph import Undirected_Graph
        
        def find_ens_max_graph_idx(self):
            """
            Find the maximum idx of the already sampled graph.
            Therefore if the n_sampels < ens_max_idx, no need to sample
            Exceptions:
            -If no ensemble folder exists, then return -1
            -If the folder is empty, then return -1
            """

            from os import walk, path
            import re
            if os.path.exists(self.sampled_graph_dir):
                if len(next(os.walk(self.sampled_graph_dir))[1]) > 0: #path.exists(self.ensemble_dir):
                    filenames = next(os.walk(self.sampled_graph_dir))[1]
                    ens_max_idx = sorted([int(re.sub(r'\D', '', x)) for x in filenames], reverse=True)[0]
                    return ens_max_idx
                else:
                    return -1
            else:

                # this helps to create the first graph_0 in the folder. Otherwise, i - max_graph_idx >= 0:
                return -1

        def ens_avg_err_lb_ub(data, confidence=0.95, axis = 0, anal_avg = None, netmeas = None, path_ = None):
            """Calculate the mean and the dispersion interval of the data
            If anal_avg = "..." it will be taken the analytical average
            If the confidence value is and int and >= 1 (e.g. 1,2,3,4,...) it will taken the standard deviation
            """
            a = 1.0 * np.array(data)
            n = a.shape[0]
            # se = st.sem(a, axis = axis)

            # deg, annd, cc will be automatically computed by means of load_or_create func
            if anal_avg == "deg":
                m = self.deg
            elif anal_avg == "annd":
                m = self.annd
            elif anal_avg == "cc":
                m = self.cc
            else:
                # here directly compute the ensemble average of deg, annd, cc
                m = np.nanmean(a, axis = axis)

            # find the lb_ci, ub_ci via quantiles (disp.interval) or standard deviation
            if confidence < 1:
                alpha =  (1 - confidence) / 2
                lb_ci = np.nanquantile(a, alpha, axis = axis, method = "linear")
                ub_ci = np.nanquantile(a, 1 - alpha, axis = axis, method = "linear")
                
                # to plot the error bars one needs to compute the error wrt the mean
                err_lb = m - lb_ci
                err_ub = ub_ci - m

                # if err_lb or err_ub would be negative (they are centered wrt to the median), there would be problems with the plotting procedure. Thus, I am clipping them. No big difference apart some few values
                err_lb = np.clip(err_lb, 0, np.inf)
                err_ub = np.clip(err_ub, 0, np.inf)
            
            else:
                std = np.nanstd(a, axis = axis, ddof = 1)
                # to plot the error bars
                err_lb, err_ub = confidence*std, confidence*std
                # to calculate the reconstruction accuracy
                lb_ci, ub_ci = m - err_lb, m + err_ub
            
            # save the ensemble average and lower/upper dispersion interval for reconstruction accuracy (no more)
            np.savetxt(path_(netmeas), np.array([m, lb_ci, ub_ci]), delimiter = ",")
            
            # return the ensemble average and the errors to plot (not the lb_ci and ub_ci)
            return m, np.array([err_lb, err_ub])                
        
        # HERE STARTS the function

        # check if deg exists
        self.ensemble_dir = self.model_dir + f"/ensemble"
        self.sampled_graph_dir = self.model_dir.replace("vars", "vars/ensemble")
        path_ = lambda netmeas: f"{self.ensemble_dir}/ens_avg_lbci_ubci_{netmeas}_conf{confidence}.csv"
        os.makedirs(self.ensemble_dir, exist_ok = True)
        if not os.path.exists(path_("deg")):

            ens_deg, ens_annd, ens_cc = [], [], []
            max_graph_idx = find_ens_max_graph_idx(self)
            n_zero_deg_nodes = 0

            for i in np.arange(n_samples):

                # here I need to sample the remainging graphs to match the n_samples
                if i - max_graph_idx > 0:
                    if i - max_graph_idx == 1: 
                        print(f"-Sampling the remaining {n_samples - max_graph_idx - 1} graphs -- already in the folder {max_graph_idx + 1 if max_graph_idx >= 0 else 'None'}")
                    sampled_A = sample_from_p(P = self.pmatrix, name = self.name, seed = 2 * i + 63)
                    
                    kwargs = {"kind" : "obs", 
                            "name" : f"graph_{i}", 
                            "model_dir": self.sampled_graph_dir + f"/graph_{i}",
                            "bin_adj" : sampled_A, 
                            "dataset_name" : self.dataset_name,
                            "cg_method" : self.cg_method,
                            }
                            
                    sampled_net = Undirected_Graph(**kwargs)
                    sampled_net.bin_adj = sampled_A
                    # np.savetxt(sampled_net.model_dir + f"/bin_adj.csv", sampled_A, delimiter = ",")

                    ens_deg_i = sampled_net.deg
                    ens_annd_i = sampled_net.annd
                    ens_cc_i = sampled_net.cc
                
                else:
                    if i == 0: print(f'-Loading (already) Sampled Graphs for {self.name}',)
                    ens_deg_i = np.genfromtxt(f"{self.sampled_graph_dir}/graph_{i}/deg.csv", delimiter=',')
                    ens_annd_i = np.genfromtxt(f"{self.sampled_graph_dir}/graph_{i}/annd.csv", delimiter=',')
                    ens_cc_i = np.genfromtxt(f"{self.sampled_graph_dir}/graph_{i}/cc.csv", delimiter=',')
                
                n_zero_deg_nodes += np.sum(ens_deg_i == 0)

                ens_deg.append(ens_deg_i)
                ens_annd.append(ens_annd_i)
                ens_cc.append(ens_cc_i)

            if n_zero_deg_nodes > 0:
                print(f'-There are n_zero_deg_nodes {n_zero_deg_nodes} in the ensemble. Not averaging ANND, CC over them.',)

            # do the averages over the sampled ensembles
            self.ens_avg_deg, self.ens_err_deg = ens_avg_err_lb_ub(ens_deg, confidence = confidence, netmeas = "deg", path_ = path_)
            self.ens_avg_annd, self.ens_err_annd = ens_avg_err_lb_ub(ens_annd, confidence = confidence, netmeas = "annd", path_ = path_)
            self.ens_avg_cc,  self.ens_err_cc = ens_avg_err_lb_ub(ens_cc, confidence = confidence, netmeas = "cc", path_ = path_)

        else:
            
            def conf_int_to_errors(net_meas):    
                # load the ens avg and dispersion interval (lower bound - upper bound) of deg, annd, cc
                m, lb_ci, ub_ci = load_array(path_(net_meas))
                return m, np.vstack((m - lb_ci, ub_ci - m))
            
            # load_arr = load_array(path_("deg"))
            # self.ens_avg_deg, tmp_ens_di_deg = load_arr[0], load_arr[1:]
            self.ens_avg_deg, self.ens_err_deg = conf_int_to_errors("deg")
            load_arr = load_array(path_("annd"))
            self.ens_avg_annd, self.ens_err_annd = load_arr[0], load_arr[1:]
            load_arr = load_array(path_("cc"))
            self.ens_avg_cc,  self.ens_err_cc = load_arr[0], load_arr[1:]

