from ..lib import *
from ..graphs.variables import variables
from ..utils.helpers import load_array

class Graph():

    def __init__(self, **kwargs):
        
        if kwargs["dataset_name"].lower().endswith("undirected"):
            variables.__init__(self)

            # add un underscored in the keys of dict if also present in restricted_var
            restricted_var = [var for var in variables.__dict__.keys() if not var.startswith("_")]
            restrict_und = lambda x: "_" if x in restricted_var and not x.startswith("_") else ""
            kwargs = {restrict_und(k) + k: v for k,v in kwargs.items()}

            # update the __dict__ with the kwargs with underscore
            self.__dict__.update(kwargs)

        elif kwargs["dataset_name"].lower().endswith("directed"):
            print('-No implementation for the directed graph',)

        self._set_model_plots_dirs()
        self.kind = "obs" if self.name.endswith(("ING", "Gleditsch", "net")) or self.name.startswith("graph_") else "exp"
    
    def _set_model_plots_dirs(self):
        Graph.static_set_model_plots_dirs(self)
    
    @staticmethod
    def static_set_model_plots_dirs(self):
        """
        Set the model directories in order to save the observed/expected measurements
        or the plots

        In the multi_models directory, we created two subfolders for the NetRes and Node-Embedding papers.
        Note: use maxlMSM at dimX > 1 to generate the multi_models plots for the Node-Embedding paper.
        """
        # add to plot name the model params
        str_model_params = ""
        if self.get("dimX"): 
            str_model_params = f"/dimX{self.dimX}"
        if self.name.endswith("LPCA") and self.dataset_name.endswith("Undirected"): 
            str_model_params = f"/dimB{self.dimB}/dimC{self.dimC}"

        # set the cg_method
        cg_meth_init_guess = self.cg_method

        if self.get("model_dir"):
            if "ensemble" in self.model_dir:
                self.plots_dir = os.path.dirname(os.path.dirname(self.model_dir)) + f"/plots/{self.name}{str_model_params}/{cg_meth_init_guess}"
        else:
            # base_dir = f"../outputs"
            base_dir = os.path.expanduser('~') + "/Documents/code_local_files/outputs"
            if self.get("corpkey"):
                base_dir = f"../../data/corealgos/rmilocco/outputs"
            base_dir += f'/datasets/{self.dataset_name}'
            # coarse grained method + initial condition path
            
            if self.id_code == "naics_code": self.cg_method = "naics"
            if self.get("cg_method") == None: self.cg_method = "random"
            
            # from now the infos are used only for the models, i.e. kind = "exp"
            # append the init guess
            if self.get("initial_guess"):
                cg_meth_init_guess += f"_ig-{self.initial_guess}"

            # create objective label if self.objective exists
            objective = self.objective + "/" if self.get("objective") else ""
            
            # for fine-graining, define a top_level_dir (fixing the top_level to be fractiones)
            top_level_dir = f"/top_level{self.top_level}" if self.get("top_level") else ""

            # for fine-graining with stripes, define the stripes_level_dir (at least top_level + 1)
            stripes_level_dir = f"/stripes_level{self.stripes_level}" if self.get("stripes_level") else ""

            # create the model_dir and plots_dir where to properly store the data
            self.model_dir = base_dir + f"/vars/{self.name}{str_model_params}/{cg_meth_init_guess}{top_level_dir}{stripes_level_dir}/level{self.level:g}"
            self.plots_dir = base_dir + f"/plots/{objective}{self.name}{str_model_params}/{cg_meth_init_guess}{top_level_dir}{stripes_level_dir}"

            # define a split directory for the train and test sets
            if self.get("n_splits"):
                if self.get("n_splits") > 1:
                    # the train and test procedure will be used only to estimate the ``best D''. Then, the whole network will be embedded
                    # we want to obtain level0/n_splits_.../KFold_mode/balan_loss/dimX.../initial_guess
                    split_specs = f"/n_splits_{self.n_splits}/{self.KFold_mode}/{self.balan_loss}"
                    add_cg_meth_init_guess = f"/{cg_meth_init_guess}" if self.get("initial_guess") else ""
                    self.model_dir = base_dir + f"/vars/{self.name}/level{self.level:g}{split_specs}{str_model_params}{add_cg_meth_init_guess}"
                    self.plots_dir = self.model_dir.replace('vars/', f'plots/{objective}')

            # create the multi_models directory: join the path before "objective" + "/multi_models"
            self.plots_dir_multi_models = "".join(self.plots_dir.partition(str(self.get("objective")))[:2]) + "/multi_models"

        os.makedirs(self.model_dir, exist_ok=True)
        
    def frmv_diag(self, adj_mat = None, rmv_diag = True):
        """
        function useful both adj_mat or a pmatrix coming from the specification of fitnesses.
        -default matrix: pmatrix_gen
        """

        if np.any(adj_mat.diagonal() != 0) and rmv_diag:
            adj_mat = adj_mat * (1 - np.eye(adj_mat.shape[0]))
        
        return adj_mat

    def csr_rmv_diag(self, adj_mat, value=0):
        """Faster remover of diagonal elements for a csr matrix, e.g. the ing one"""
        from scipy import sparse, newaxis
        zl_mat = adj_mat - sparse.dia_matrix((adj_mat.diagonal()[newaxis, :], [0]), shape=adj_mat.shape)
        return zl_mat

    @staticmethod
    def n_nodes_from(pdf, id_code = None):
        """Extract the number of nodes from the pdf"""
        if id_code is None:
            id_code = pdf.columns[0].split("_")[-1]
        return len(pd.unique(pdf.loc[:, [f'payer_{id_code}', f'beneficiary_{id_code}']].to_numpy().ravel('K')))

    def n_nodes_from(self, pdf):
        """Extract the number of nodes from the pdf"""
        return len(pd.unique(pdf.loc[:, [f'payer_{self.id_code}', f'beneficiary_{self.id_code}']].to_numpy().ravel('K')))

    def _set_int2idcode_from(self, pdf):
        """ Extract from the pdf the int2idcode and its number (n_nodes)"""
        full_path = f'{self.model_dir}/mic_{self.id_code}.csv'
        
        #extract from pdtrans the unique int2idcode (and their number but not needed)
        if os.path.exists(full_path):
            self.int2idcode = load_array(full_path)
        else:
            self.int2idcode_from = lambda pdf: pd.unique(pdf.loc[:, [f'payer_{self.id_code}', f'beneficiary_{self.id_code}']].to_numpy().ravel('K'))
            self.int2idcode = self.int2idcode_from(pdf)
            np.savetxt(full_path, self.int2idcode, fmt='%i', delimiter=",")
        
        # save the n_nodes
        self.n_nodes = int(self.int2idcode.size)
        

    def get_graph_basics(self, pdf = None):
        """
        Get the network measures from the (ALREADY coarse-grained) pdf and calculate:
        - nodes and n_nodes
        - idcode2node
        - self_loops and n_self_loops
        - n_edges
        - fully connected and fully disconnected nodes
        - labels and node means integer labels, e.g. label = 0 = node where id_code could be NAICS codes or other numbers
        """

        # check if pdf has been passed as a Dataframe
        if isinstance(pdf, type(None)):
            pdf = self.pdtrans
        
        if self.dataset_name.lower().startswith("Gleditsch"):
            if not os.path.exists(self.model_dir + f"/gdp.csv"):
                # create summed gdp
                gdp = load_array(os.path.expanduser('~') + "/Documents/code_local_files/datasets/Gleditsch-2000-Directed/gdp.csv")
                self.gdp = np.array([sum(gdp[self.mic2mac_int == lab]) for lab in range(self.n_nodes)])

                np.savetxt(self.model_dir + f"/gdp.csv", self.gdp, delimiter=",")
            
        # n_of_self_loops
        mask_self_loops = pdf[f'payer_{self.id_code}'].eq(pdf[f'beneficiary_{self.id_code}'])
        self.self_loop_nodes, self.n_self_loops = None, sum(mask_self_loops)
        if self.n_self_loops > 0:
            self.self_loops = pdf.loc[mask_self_loops].iloc[:, 0].to_numpy()
            self.self_loop_nodes = self.map_idcode2int(self.self_loops)

        # n_of_edges = undirected number of links for undirected and directed ones for directed networks
        self.n_edges = pdf.shape[0] - self.n_self_loops

        if self.dataset_name.lower().endswith("undirected"):
            # def. "->"" as "needs" 
            # self.deg -> self.load_or_create -> zl_bin_adj -> create_wei_bin_adj -> map_idcode2int and self.pdtrans
            # maybe it would be better to do it as in the "directed case" to avoid the need of the zl_bin_adj

            # divide by two if the network is undirected
            self.n_edges /= 2

            # fully-connected and fully-disconnected to prevent every model to fit
            self.fc_nodes = np.where(self.deg == self.n_nodes - 1)[0]
            # self.n_fc_nodes = self.fc_nodes.size
            self.fd_nodes = np.where(self.deg == 0)[0]
            # self.n_fd_nodes = self.fd_nodes.size
            self.n_fcfd_nodes = self.fc_nodes.size + self.fd_nodes.size
        
        elif self.dataset_name.lower().endswith("directed"):

            # obtain deg_out and deg_in
            # self.deg_sseq(pdf)

            # fully-connected and fully-disconnected OUT nodes
            self.fc_nodes_out = np.where(self.deg_out == self.n_nodes - 1)[0]
            self.fd_nodes_out = np.where(self.deg_out == 0)[0]
            self.n_fcfd_nodes_out = self.fc_nodes_out.size + self.fd_nodes_out.size

            # fully-connected and fully-disconnected IN nodes
            self.fc_nodes_in = np.where(self.deg_in == self.n_nodes - 1)[0]
            self.fd_nodes_in = np.where(self.deg_in == 0)[0]
            self.n_fcfd_nodes_in = self.fc_nodes_in.size + self.fd_nodes_in.size

            # overall fully-connected and fully-disconnected nodes
            self.fc_nodes = np.concatenate((self.fc_nodes_out, self.fc_nodes_in))
            self.fd_nodes = np.concatenate((self.fd_nodes_out, self.fd_nodes_in))

            self.n_fcfd_nodes = self.n_fcfd_nodes_out + self.n_fcfd_nodes_in        
            # save the strengths
            self.save_var(var = self.stre_out, var_name = "stre_out")
            self.save_var(var = self.stre_in, var_name = "stre_in")
            # self.save_var(var = self.stre_out[:, None] @ self.stre_in[None, :], var_name = "matrix_stre_out_in")

        # check if the number of edges is an integer and if it is covert it to an integer
        self.n_edges = int(self.n_edges) if not(self.n_edges % 1) else self.n_edges
        
        # save the n_edeges
        self.save_var(var = np.array([self.n_edges]), var_name = "n_edges")

        # # the applied models will use strengths (and stripes). Here, only save the strengths
        # self.stre_out = self.zl_wei_adj.sum(1)
        # self.stre_in = self.zl_wei_adj.sum(0)


    def get_cluster_idxs(self):
        """
        Get the cluster indices
        """
        from math import modf
        n, d = self.n_nodes, self.mode_cluster_size
        fr_, int_ = modf(n / d)
        cluster_idxs = list(range(int(int_)))*d + list(range(int(fr_*d)))
        
        return cluster_idxs

    def get(self, var_name):
        """Return the variable if it exists, otherwise return None.
        Note that for inner variables one should get the _var_name. That is why we check also _name."""
        if np.array_equiv(self.__dict__.get(var_name), None):
            return self.__dict__.get("_"+var_name)
        return self.__dict__.get(var_name)

    def get_integer_node_identifier(self, pdf):
        """
            create a mapping to have the integer indexes of the nodes. E.g. level 0: idcode2int[111180] = 0; level 1: idcode2int[11118] = 0
            - mic2mac_int associates each microscopic to its community, where it belongs. E.g. mic2mac_int --> [0,1,1,2,2,2,...]
            - len(mic2mac_int) = n_nodes at level 0, whereas len(int2idcode) = n_nodes at level (>=0) = len(idcode2int)
        """
        if self.id_code.endswith("int"):
            self.map_idcode2int = lambda x: x
        else:
            # select the id_codes from pdf as they need for coarse_grained_pdtrans
            self._set_int2idcode_from(pdf)

            # map the node naics_code to the index / position in the nodes array
            self.idcode2int = {idcode: int_idx for int_idx, idcode in enumerate(self.int2idcode)}

            # vectorize idcode2node s.t. it may accept a list index in the idcode2node dictionary
            self.map_idcode2int = lambda x: np.array(list(map(self.idcode2int.get, x)))
            self.nodes = self.map_idcode2int(self.int2idcode)
            
            # rescaling by 10**level only to match the keys of the map_idcode2int
            micro_idcode = np.genfromtxt(self.model_dir.replace(f"level{self.level}", "level0") + f"/mic_{self.id_code}.csv", delimiter = ",")
            
            # get the cg_labels for the coarse-graining part            
            if self.cg_method.startswith(("naics")):

                # remove the last digits of the naics code
                self.mic2mac_int = self.map_idcode2int(micro_idcode // 10**self.level)
            
            elif self.cg_method.startswith("random"):
                self.mic2mac_int = self.map_idcode2int(micro_idcode)
            
            # save the mic2mac_int
            if not os.path.exists(self.model_dir + f"/mic2mac_int.csv"):
                np.savetxt(self.model_dir + f"/mic2mac_int.csv", self.mic2mac_int, delimiter=",", fmt='%i')
            

    def _set_cg_pdtrans(self, pdtrans = None, distance_matrix = None, lvl_to_nclust = None):
        """ 
        Setup the coarse-grained pdtrans. 
        If distance_matrix is passed, it means that the coarse-grained pdtrans will be created using the distance_matrix.
        Else if the naics_code are used, the last digit will be cut.
        """
        
        # define the function to group_by by pruned naics codes
        id_code = self.id_code
        pdtrans = pdtrans.copy()

        # load or create the mic2mac_int
        full_path = f'{self.model_dir}/pdtrans.csv'
        if os.path.exists(full_path):
            
            # get integer indexes identifying each node
            # map the node naics_code to the index / position in the nodes array
            # vectorize idcode2node s.t. it may accept a list index in the idcode2node dictionary
            pdtrans = pd.read_csv(full_path)
            self._set_int2idcode_from(pdtrans)
            self.idcode2int = {idcode: int_idx for int_idx, idcode in enumerate(self.int2idcode)}
            self.map_idcode2int = np.vectorize(self.idcode2int.get)

            self.mic2mac_int = np.genfromtxt(self.model_dir + '/mic2mac_int.csv', delimiter = ",", dtype = int)
            self.n_nodes = int(self.mic2mac_int.max() + 1)
            return pdtrans
        
        else:            
            # if ge-dist use distance
            if self.cg_method.endswith("dist"):

                if self.level == 0:
                    # fix n_nodes inside this function
                    self.n_nodes = self.n_nodes_from(pdtrans)
                    self.mic2mac_int = np.arange(self.n_nodes, dtype = int)
                    np.savetxt(f'{self.model_dir}/mic_{id_code}.csv', self.mic2mac_int, fmt='%i', delimiter=",")
                    np.savetxt(f'{self.model_dir}/mic2mac_int.csv', self.mic2mac_int, fmt='%i', delimiter=",")
                
                else:
                    # assign labels which refers to the 0th level 
                    # inside self.n_nodes is set
                    self._assign_labels(distance_matrix = distance_matrix, lvl_to_nclust = lvl_to_nclust)
                    
                    # create a mapping functions from 0th level labels, i.e. increasing integers, to the level-th ones
                    self.map_nodes2mic2mac_int = {node : mic_lab for node, mic_lab in enumerate(self.mic2mac_int)}

                    # map the 0th level pdtrans labels to the ones at level (level_pdtrans)
                    pdtrans.loc[:, [f'payer_{id_code}', f'beneficiary_{id_code}']] = \
                        pdtrans.loc[:, [f'payer_{id_code}', f'beneficiary_{id_code}']].map(self.map_nodes2mic2mac_int.get)
            
            # if naics_code cut the last digit
            elif self.cg_method.startswith("naics"):
                
                if self.level == 0:
                    # save the mic_id_code nodes as for the upper levels they will be used to create the coarse-grained pdtrans
                    self._set_int2idcode_from(pdtrans)
                else:
                    #if level == 1: level_pdtrans = pdtrans.copy()
                    pdtrans.loc[:, [f'payer_{id_code}', f'beneficiary_{id_code}']] = \
                        pdtrans.loc[:, [f'payer_{id_code}', f'beneficiary_{id_code}']].apply(lambda x: x // 10**self.level)
                    # pdtrans = pdtrans.loc[:, [f'payer_{id_code}', f'beneficiary_{id_code}']].apply(lambda x: x // 10**self.level)

                    # get n_nodes
                    self._set_int2idcode_from(pdtrans)
                    del self.int2idcode

            elif self.cg_method.startswith("random"):
                if self.level == 0:
                    # save the mic_id_code nodes as for the upper levels they will be used to create the coarse-grained pdtrans
                    self._set_int2idcode_from(pdtrans)

                    #extract from pdtrans the unique int2idcode (and their number but not needed)
                    np.savetxt(f'{self.model_dir}/mic_{id_code}.csv', self.int2idcode, fmt='%i', delimiter=",")

            
            # group by the new pdtrans as there will be multiple-edges, e.g. (0,1), as multiple nodes will be in the same block
            pdf_groupby = lambda pdf: pdf.groupby(by = [f'payer_{id_code}', f'beneficiary_{id_code}'], sort = False).sum().reset_index()
            pdtrans = pdf_groupby(pdtrans)
            
            # save it s.t. it will be not recalculate the next time
            pdtrans.to_csv(full_path, index= False)

            # save the integer indexes identifying each node derived from pdtrans
            self.get_integer_node_identifier(pdtrans)

            return pdtrans

    def create_wei_bin_adj(self, binary = True):
        from scipy.sparse import csr_matrix
        # For scipy sparse it will be computationally demanding to subtrack the diagonal
        # Create both the matrices at once here
        
        def data_row_col(pdf):
            # create the weighted csr_matrix
            
            # find the index of the nodes
            edges_id = lambda str_name: self.map_idcode2int(pdf.loc[:, f"{str_name}_{self.id_code}"].tolist())
            row = edges_id("payer")
            col = edges_id("beneficiary")
            
            # extract the data and the binary data
            data = pdf["amount_euro"].values
            # bin_data = 1*len(data)

            return data, (row, col)

        # create the weighted adjacency matrix
        self.wei_adj = csr_matrix(data_row_col(self.pdtrans), shape = (self.n_nodes, self.n_nodes))
        self.bin_adj = self.wei_adj.copy()
        self.bin_adj.data = np.ones_like(self.wei_adj.data)
        format_ = "npz"
        
        if self.n_nodes < 1e3:
            self.wei_adj = self.wei_adj.toarray()
            self.bin_adj = self.bin_adj.toarray() #np.where(self.wei_adj, 1, 0)
            format_ = "csv"
        
        # save weigh and bin adj
        self.save_var(var_name = "wei_adj", var = self.wei_adj, format_=format_)
        self.save_var(var_name = "bin_adj", var = self.bin_adj, format_=format_)

        # return what was asked, but as soon as it is accessed, it will not be recalled for the other matrix
        # e.g. if the bin_adj is accessed, the wei_adj will be computed as well and load_or_create will prevent a second computation 
        if binary:
            return self.bin_adj
        else:
            return self.wei_adj

    def load_or_create(self, var_name = None, str_var = None, var = None, save = False, format_ = "csv"):
        """
        load the `{var_name}` variable or creating it using

        1) (preferred) dict of built-in funcs accessed by var_name.
        Note: If only self.var, e.g. self._X, it means the variable has been already created. If it was created in a previous run thenand it will be loaded from the file.

        2) the variable itself. It will evaluate whenever passed as arg. E.g. used when sampling the ensemble;
        dict of func has `lambda x: func() if x else 0` since it is trick to ``pause`` func() which is evaluated only if x is True.

        NOTE: for the directed case, 
        - deg_out, sout 
        - deg_in, sin 
        are in self.deg_sseq function
        """
        from scipy.sparse import save_npz, load_npz
        file_path_full = self.model_dir + f"/{var_name}.{format_}"
        
        # search if the variable is there then load it
        if os.path.exists(file_path_full) and not save:
            if format_ == "npz":
                return load_npz(file_path_full)
            else:
                return np.genfromtxt(file_path_full, delimiter=',', skip_header=False)
                   
        # if the path doesnt exists or we want to force saving the variable
        elif not os.path.exists(file_path_full) or save:
            if self.dataset_name.lower().endswith("undirected"):
                dict_of_fun = {
                    "wei_adj" : lambda x: self.create_wei_bin_adj() if x else 0,
                    "bin_adj" : lambda x: self.create_wei_bin_adj() if x else 0, # x = self.weigh_adj
                    "deg": lambda x: self.sparse_ndarr_net_meas("deg") if x else 0, # A1 was added to return a nd.array instead of a matrix which is in the same format of the .csv file that will be saved
                    "stre": lambda x: self.wei_adj.sum(1) if x else 0,
                    "n_edges" : lambda x: self.zl_pmatrix.sum() / 2 if x else 0,
                    "annd": lambda x: self.sparse_ndarr_net_meas("annd") if x else 0,
                    "cc":  lambda x: self.sparse_ndarr_net_meas("cc") if x else 0,
                    "X" : lambda x: self.X if x else 0,
                    "pmatrix" : lambda x: self.pmatrix if x else 0,
                    "w": lambda x: self.w if x else 0,
                    "delta" : lambda x: self.delta if x else 0,
                    }
            elif self.dataset_name.lower().endswith("directed"):
                dict_of_fun = {
                    "wei_adj" : lambda x: self.create_wei_bin_adj() if x else 0,
                    "bin_adj" : lambda x: self.create_wei_bin_adj() if x else 0, # x = self.weigh_adj
                    "X" : lambda x: self.X if x else 0,
                    "Y" : lambda x: self.Y if x else 0,
                    "deg_in": lambda x: self.sparse_ndarr_net_meas("deg_in") if x else 0,
                    "deg_out": lambda x: self.sparse_ndarr_net_meas("deg_out") if x else 0,
                    "anndoo": lambda x: self.sparse_ndarr_net_meas("anndoo") if x else 0,
                    "anndii" : lambda x: self.sparse_ndarr_net_meas("anndii") if x else 0,
                    "anndoi" : lambda x: self.sparse_ndarr_net_meas("anndoi") if x else 0,
                    "anndio" : lambda x: self.sparse_ndarr_net_meas("anndio") if x else 0,
                    "stre_in": lambda x: self.wei_adj.sum(0) if x else 0,
                    "stre_out": lambda x: self.wei_adj.sum(1) if x else 0,
                    "pmatrix" : lambda x: self.pmatrix if x else 0,
                    "w": lambda x: self.w if x else 0,
                    "delta" : lambda x: self.delta if x else 0,
                    }
                    
            if not np.array_equiv(var, None): var = var
            elif var_name: 
                var = dict_of_fun[var_name](True)
            else: print(f'-No variables found!',)
            
            # save the variable
            self.save_var(path = file_path_full, var = var, format_ = format_)

            return var
        
    def is_integer_array(arr):
        return np.issubdtype(arr.dtype, np.integer)
    
    def save_var(self, path = None, var = None, var_name = "deg",  format_ = "csv", force_save = False):
        '''
            Function which allows to save the var as var_name.format
            The format may be specified both as extension of the var_name or filling the field.
            Highest priority to the ext of the var_name
        '''
        from scipy.sparse import save_npz

        # search if format was in the var_name, e.g. bin_adj.npz --> format_ = "npz"
        if var_name.split(".")[-1] != var_name:
            format_ = var_name.split(".")[-1]
        
        if path == None: 
            path = self.model_dir + f"/{var_name}.{format_}"
        
        if not os.path.exists(path) or force_save:
            if format_ == "npz":
                save_npz(path, var)
            elif format_ == "pt":
                tc.save(var, path)
            elif format_ == "csv":
                fmt='%i'
                if np.any(var % 1.):
                    fmt='%.18e'
                if np.isscalar(var):
                    var = np.array([var])
                np.savetxt(path, var, delimiter=",", fmt=fmt)
            elif format_ == "pkl":
                import pickle
                with open(path, 'wb') as f:
                    pickle.dump(var, f)

    def par_folder(self, start_dir, n_par_dir = 0):
        from pathlib import Path
        return str(Path(start_dir).parents[n_par_dir])

    def track_n_edges(self, list_rel_err, obs_net):
        rel_error = lambda x, y: (x-y)/y
        list_rel_err += [rel_error(self.n_edges, obs_net.n_edges)]

    def _assign_labels(self, distance_matrix = None, lvl_to_nclust = None, save_labels = True):
        '''
        Function to assign labels to the nodes based on the distance_matrix.
        We aritrarly choose to have only 10 levels. Thus, lvl_to_nclust is a dictionary with 10 key-levels and the corrisponding distances

        The labels are the block-node they will belong after coarse-graining the orginal network.
        '''
        from sklearn.cluster import AgglomerativeClustering

        mem_dir = self.model_dir #f"outputs/datasets/{self.dataset_name}/vars/{self.name}/{self.cg_method}/level{self.level:g}"
        
        # if level > 0, the labels are assigned by cutting the dendrogram of minimum distances
        # AgglomerativeClustering with single-linkage and distance_threshold ONLY to get the labels_
        sub_model = AgglomerativeClustering(
                                            n_clusters=lvl_to_nclust[self.level], 
                                            linkage = "single", metric = "euclidean", memory = mem_dir
                                            )
        
        # fit Agglomarative clustering
        sub_model = sub_model.fit(distance_matrix)
        self.mic2mac_int = sub_model.labels_

        # the number of nodes in the cg-adjacency matrix will be just the max label + 1
        self.n_nodes = self.mic2mac_int.max()+1

        # save the labels_ of the prev-level-nodes that are going to be merged into the new coarse-grained nodes
        # if geo-dist, for fine-graining of l = 4, load the l = 5 cg_labels since they are labelling the ones of l = 4
        if save_labels:
            np.savetxt(mem_dir + f"/mic2mac_int.csv", self.mic2mac_int, fmt='%d', delimiter=",")

    def is_missing(self, net_meas):
        """ Check if the net_meas exists in the model directory """
        return not os.path.exists(self.model_dir + f"/{net_meas}.csv")

    def isource_2_itarget(self, obs_net, lsour, ltar):
        '''
        It creates a dict that as key it has the labels of the nodes at level lsour and as values the labels of the nodes at level ltar.
        
        Here we will address to "i" as "lab" as it is crucial the difference abount "labels" at a certain levels (e.g. node 0) and their indexes in the array (e.g. [0,1,2]) which are the microscopic node ending up in the community at that level.
        Since "i" is also the initial of "index", we decided to go for "lab".
        
        For the Coarse-Graining, lsour < ltar and the dict will have multiple keys with the same value.
        For the Fine-Graining, lsour > ltar and the dict will have a list of values for each key.
        Otherwise, one has to map it
        '''
        from pickle import load, dump
        from utils import uu_fun

        from os.path import dirname
        dir_path = dirname(obs_net.model_dir)+'/isource_2_itarget'
        os.makedirs(dir_path, exist_ok = True)
        full_path = f"{dir_path}/from{lsour}_to_{ltar}.pkl"

        if not os.path.exists(full_path):
            if self.__dict__.get("cgl0_l") is None:


                obs_dirl = lambda l: dirname(obs_net.model_dir) + f"/level{l}/mic2mac_int.csv" #f"outputs/datasets/gleditsch/vars/{obs_class_name}/{self.cg_method}/level{l:g}"
                self.cgl0_l = lambda l: np.genfromtxt(obs_dirl(l), dtype = int)

            if lsour == ltar:
                # create the same scale dict
                is_2_it = {i : [i] for i in np.arange(self.cgl0_l(lsour).max() + 1)}
                
                # save the dict and return it
                with open(full_path, "wb") as f:
                    dump(is_2_it, f)
                return is_2_it
            else:
                # find the indexes for which in the source layer you have a value of isour
                # where_isour_l = lambda l, i: np.where(self.cgl0_l(l) == i)
                sour_lab2idx = lambda lab_sour: np.where(self.cgl0_l(lsour) == lab_sour) #where_isour_l(lsour, isour)

                # find the target labels of the sour_lab2idx entries
                lab_sour2tar_pn = lambda lab_sour: self.cgl0_l(ltar)[sour_lab2idx(lab_sour)]
                source_n_nodes = self.cgl0_l(lsour).max()+1

                def single_uu_fun(lab_sour, lsour, ltar):
                    # singe if lsour < ltar there will be multiple ones, take the first
                    if lsour < ltar:
                        return uu_fun(lab_sour2tar_pn(lab_sour))[0]
                    else:
                        return uu_fun(lab_sour2tar_pn(lab_sour))

                is_2_it = {lab_sour : single_uu_fun(lab_sour, lsour, ltar) for lab_sour in np.arange(source_n_nodes)}

                # save the dict and return it
                with open(full_path, "wb") as f:
                    dump(is_2_it, f)
                return is_2_it
        else:
            with open(full_path, "rb") as f:
                return load(f)