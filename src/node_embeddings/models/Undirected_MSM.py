from ..lib import *
from ..graphs.Graph import Graph
from ..graphs.Undirected_Graph import Undirected_Graph
from ..algorithms.top2bot2top import top2bot2top
from ..algorithms.param_rearranger import param_rearranger
from ..algorithms.minimizers import minimizers
from ..utils.helpers import rel_err

class Undirected_MSM(Undirected_Graph, Graph, top2bot2top, param_rearranger, minimizers):
    
    def __init__(self, obs_net, **kwargs):

        # copy ONLY the parameters not in variables.py as they will be erased by __init__
        kwargs.update({"level" : obs_net.level, # level of the multiscale graph
                    "dataset_name" : obs_net.dataset_name,
                    "id_code" : obs_net.id_code, # decide if naics_code pipeline or grid_id one
                    "pdtrans" : obs_net.pdtrans,
                    "year" : obs_net.year,
                    "cg_method" : obs_net.cg_method,
                    "map_idcode2int" : obs_net.map_idcode2int, # convert from pid to idx, i.e. nodes 0,1,2.=,...
                    "obs_deg" : obs_net.deg,
                    "mic2mac_int" : obs_net.mic2mac_int,
                    "self_loop_nodes" : obs_net.self_loop_nodes, # needed for setting the self-loops entries
                    "n_self_loops" : obs_net.n_self_loops, # needed for setting the self-loops entries
                    "fc_nodes" : obs_net.fc_nodes,
                    "n_fcfd_nodes" : obs_net.n_fcfd_nodes,
                    "fd_nodes" : obs_net.fd_nodes, # needed in the plotting procedure of ref_X VS sum_X
                    })
        Undirected_Graph.__init__(self, **kwargs)

        # set the variable of the observed graph excluded from the prev update
        self.__dict__.update({"_n_nodes" : obs_net.n_nodes,})

        self._n_params_X = None
        self._pmatrix = None
        self._zl_pmatrix = None
        self.nit = 0 # as we will need to print the number of iteration during the optimization
        
        # for the NodeEmbedding just display MSM as even LPCA was fitted via maximum likelihood
        self.plots_name = self.name
        if self.objective == "NodeEmb" and self.name.endswith("maxlMSM"):
            self.plots_name = "MSM"

        # if self.get("reduced_by") == None and self.kind != "obs":
        if self.kind != "obs":
            if "fitn" in self.name:
                self.reduced_by = None
            elif self.name.endswith(("degcMSM", "CM")):
                self.reduced_by = "deg"
                self.streq_nodes_deg()
            elif self.name.endswith(("maxlMSM")):
                self.reduced_by = "neigh"
                self.streq_nodes_neigh(A = obs_net.zl_bin_adj)
            elif self.name.endswith("LPCA"):
                self.reduced_by = None
                #self.streq_nodes_neigh(A = obs_net.zl_bin_adj)
            
            # don't go throught all the reduced_by steps if the nodes are all strineq
            if self.n_strineq_nodes == self.n_nodes:
                self.reduced_by = None

            if not os.path.exists(self.model_dir + "/reduced_by.txt"):
                # np.save(, self.reduced_by)
                open(self.model_dir + "/reduced_by.txt", "w").write(str(self.reduced_by))

    def obs_deg_mat(self, obs_net):
        # obtain degree of the super-block
        self._import_funcs_labels_deg(obs_net.name)
        mat_degreesl = lambda l: self.Gleditsch_degl(l)[self.cg_labels_l(l)].reshape(-1,1)
        multis_deg = np.hstack([mat_degreesl(l) for l in np.arange(self.top_l, self.top_l+self.dimX)])
        print(f'-multis_deg.shape: {multis_deg.shape}',)
        return multis_deg
    
    def func_obs_deg(self):
        if self.dimX == 1:
            return self.fit_deg
        else:
            return self.obs_deg_mat()
    
    def _set_initial_guess(self, obs_mat = None): 
        self.available_initial_guess = ['degrees_minor', 'nodes_degrees_minor', 'degrees', 'chung_lu', 'nodes_chung_lu', 'random', 'random_degrees', 'uniform', 'degcMSM.X']
        if type(self.initial_guess) == str:
            
                
            if self.initial_guess == "degrees_minor":
                print('-Check self.n_edges: it should be the observed edges',)
                # self.X0 = self.func_obs_deg() / (np.sqrt(self.n_edges)+1) # for proper CHUNG-LU see degrees_minor
            elif self.initial_guess == "nodes_degrees_minor":
                self.X0 = self.func_obs_deg() / self.n_strineq_nodes

            elif self.initial_guess.startswith("random"):
                seed = self.seed
                np.random.seed(seed)

                if self.initial_guess == "random":    
                    self.X0 = np.random.random(size=self.n_params_X)
            
                elif self.initial_guess == "random_degrees":
                    deg_mat = np.repeat(self.obs_deg.reshape(-1,1), self.dimX, axis = 1)
                    noise = np.random.normal(0, 1, size = (deg_mat.shape))
                    noise[:,-1] = self.dimX-np.sum(noise[:,:-1], axis = 1)
                    self.X0 = deg_mat*noise

            elif self.initial_guess == "uniform":
                self.X0 = 10 * np.ones(shape = self.n_params_X, dtype=np.float64)  # All probabilities will be 1/2 initially
            elif self.initial_guess == "degrees":
                if self.name.startswith("degcMSM"):
                    print("WARNING: the deg-equations for the MSM are pathological")
                self.X0 = self.func_obs_deg().astype(np.float64)
            elif self.initial_guess == "chung_lu":
                self.X0 = self.func_obs_deg().astype(np.float64)/(np.sqrt(self.n_edges) + 1) # This +1 increases the stability of the solutions.
            elif self.initial_guess == "nodes_chung_lu":
                self.X0 = self.func_obs_deg().astype(np.float64)/np.sqrt(self.n_strineq_nodes) # for proper CHUNG-LU see degrees_minor
            elif self.initial_guess == "n_edges_fit":
                self.X0 = np.array([1e-15]) # if pij_arg = gdp_i * gdp_j / d_ij, x0 = 3e-7

            
            # REFINE THIS: problem since I need to generate the degcMSM.X for every layer
            elif self.initial_guess == "degcMSM.X":
                #if self.dimX == 1:
                cgmeth_ig = self.cg_method
                cgmeth_ig += f"_ig-nodes_degrees_minor"
                full_path = f"outputs/datasets/{self.dataset_name}/vars/degcMSM/dimX{self.dimX}/{cgmeth_ig}/level{self.level}/X.csv" # for testing
                if os.path.exists(full_path):
                    self.X0 = np.genfromtxt(full_path, delimiter=',')
                else:
                    print(f"WARNING: {full_path} not found. NOT using degcMSM.X from 'nodes_degrees_minor' as initial guess")
                #else:
                #    print(f"Excluding degcMSM.X @ dimX: {self.dimX} > 1 since will not provide a good initial condition")
            elif self.initial_guess == "pso":
                self._find_best_initial_guess_pso(obs_mat, n_particles = self.n_particles, iters = 50) 
                #pass
            else:
                raise ValueError(f"{self.initial_guess} not in the available initial guesses: {', '.join(self.available_initial_guess)}")
            
        else:
            self.X0 = self.initial_guess

        self.X0 = self.X0.ravel()

    def half_diag(self, X2):
        '''if tc.is_tensor(X2):
            v = torch.diag(X2)/2
            mask = torch.diag(torch.ones_like(v)).bool()
            return tc.where(mask, v, X2)
        else: #numpy array'''
        d = np.einsum('ii->i', X2)
        d /= 2
        return X2

    def pmatrix_gen(self, x = None, set_w_diag = False):
        """pmatrix for the CM, LPCA and MSM models"""
        
        
        if self.name.endswith("MSM"):
            x = self.flat_to_mat(x)
            pmatrix = -np.expm1(-self.half_diag(x@x.T))

            # set the self-loops
            if set_w_diag == True:
                
                # all the nodes don't a self loop
                # set either the w and the diagonal
                self.w = np.ones(self.n_nodes)*np.inf
                
                if self.n_self_loops < self.n_nodes:

                    x_norm_all = np.linalg.norm(x, axis = 1)**2
                    no_self_loops = np.setdiff1d(np.arange(self.n_nodes), self.self_loop_nodes)
                    x_norm_sl = x_norm_all[no_self_loops]
                    self.w[no_self_loops] = - 0.5*x_norm_sl
                    
                    np.fill_diagonal(pmatrix, -np.expm1(-0.5*x_norm_all - self.w))
                
                # if all the nodes have a self loop
                else:
                    np.fill_diagonal(pmatrix, 1)
            
            elif set_w_diag == "diag":
                x_norm_all = np.linalg.norm(x, axis = 1)**2
                np.fill_diagonal(pmatrix, -np.expm1(-0.5*x_norm_all - self.w))

            return pmatrix

        elif self.name.endswith(("CM", "fitnCM")):
            x = self.flat_to_mat(x)
            return x @ x.T / (1 + x @ x.T)

        elif self.name.endswith("LPCA"):
            x = self.X
            if x.ndim > 0:
                x = x.ravel()
            # if x.ndim > 0:
            #     x = x.ravel()

            self.B, self.C = self.flat_to_matBC(x)
            logits = self.B @ self.B.T - self.C @ self.C.T
            return expit(logits)
    
    def fexp_deg(self, x = None, red_class = None):
        """
        Use the adj_mat without fc_ / fd_ nodes to determine the "reduced" degree.
        Attention: it is not possible to use only the representative nodes 
        since the degrees will be mis-estimated as they will be missing the inner connections among the equivalent nodes
        """

        if self.reduced_by.startswith("deg"):
            x = self.repeat_streq_X(red_X = x)
        self.zl_pmatrix = self.frmv_diag(adj_mat = self.pmatrix_gen(x))
        self.deg = np.sum(self.zl_pmatrix, 1)
        # print(f'-self.deg[0]: {self.deg[0]}',)
        return self.deg

    def deg_fit(self):
        """
        Fit the degrees (obs_deg) which are the set of digrees without deterministic nodes, i.e. fc- and fd-.
        We are removing the det nodes in the procedure finding for streq_nodes
        """
        # fsolve is a wrapper of MINIPACK's HYBRD (see https://en.wikipedia.org/wiki/Powell%27s_dog_leg_method)
        # thus, use directly optimize least_squares
        from scipy.optimize import least_squares

        # red_deg = obs_deg[self.strineq_nodes] if self.reduced_by == "deg" else obs_deg
        # fun = lambda x: (self.fexp_deg(x) - 1) * red_deg * np.sqrt(self.n_nodes_per_class),
        
        res = least_squares(
            fun = lambda x: (self.fexp_deg(x) - self.obs_deg) ,
            x0 = self.X0, 
            bounds = (np.finfo(float).eps, np.inf),
            verbose = 0,
            max_nfev = 300,
            ftol = 1e-8,
            )

        print(f'-norm2_deg: {2*res.cost}',)

        # recover the full-model, i.e. replicate the x for streq_nodes and insert the fc_ / fd_ nodes
        self.X = self.repeat_streq_X(red_X = res.x).reshape(-1, self.dimX)
        self.X = self.set_fcfd_X().reshape(-1, self.dimX)

        # May happend that `fexp_deg` will go in overflow --> redo calcs with last self.X 
        self.zl_pmatrix_func(x = self.X, set_w_diag = True)
        # self.dxn = [self.X0.ravel()] #ravel since scipy will return only raveled vectors

        return self.X

    def n_edges_fit(self, obs_net, mic_ext_var, initial_guess = "n_edges_fit"):
        '''
        Find the (global) parameter delta (x) such that the expected number of edges is equal to the observed one.
        The used strengths are the sum of the microscopic ones and, note, they are greater (or equal) than the strengths at that specific level.
        The two strengths are equal if there is no connection among the interal memeber of the two communities.
        '''

        from scipy.optimize import fsolve
        labels = obs_net.mic2mac_int
        if obs_net.name.endswith("ING"):
            self.X0 = 1e-20
        elif obs_net.name.endswith("Gleditsch"):
            self.X0 = 1e-15

        # load the strength or GDP and create the external variable matrix s_i @ s_j.T
        ext_var_I = obs_net.stre.reshape(-1, 1)
        ev_ij = ext_var_I @ ext_var_I.T
        
        if self.name.endswith("fitnMSM"):
            fpmatrix = lambda z: -np.expm1(-z*ev_ij)
        elif self.name.endswith("fitnCM"):
            fpmatrix = lambda z: z * ev_ij / (1 + z * ev_ij)

        fexp_n_edges = lambda A: np.sum(np.triu(A, k = 1)) 
        func = lambda z: fexp_n_edges(fpmatrix(z)) - obs_net.n_edges
        self.delta = fsolve(func, x0 = self.X0)
        self.X = np.sqrt(self.delta)*ext_var_I

        self.pmatrix = fpmatrix(self.delta)
        self.save_var(var = self.delta, var_name = "delta")
        self.save_Xw_pmatrix()
        self.zl_pmatrix = self.frmv_diag(adj_mat = self.pmatrix)

        # relative error
        print(f'-rel_err(x = self.n_edges, y = obs_net.n_edges): {rel_err(x = self.n_edges, y = obs_net.n_edges)}',)
        return self.delta

    
    def _set_w(self):
        """
        set the value of w for the self-loops
        If n_self_loops == n_nodes: w = np.inf --> p_ii = 1
        Otherwise, w = inf if A = 1, w = - diag if A = 0

        Obs: self.n_nodes not self.n_strineq_nodes as we will use the full matrix
        """
        # if self.n_self_loops < self.n_nodes:
        #     x_norm_all = np.linalg.norm(x, axis = 1)**2
        #     np.fill_diagonal(pmatrix, -np.expm1(-0.5*x_norm_all - self.w).astype(int))
        # else:
        #     self.w = np.ones(self.n_nodes)*np.inf
        X = self.flat_to_mat(self.X)
        self.w = np.ones(self.n_nodes)*np.inf

        # if degree-corrected, we are not seeing the self-loops so we will set them to zero in pmatrix, i.e. w = -x_i**2/2
        if self.n_self_loops < self.n_nodes:

            x_norm_all = np.linalg.norm(X, axis = 1)**2
            no_self_loops = np.setdiff1d(np.arange(self.n_nodes), self.self_loop_nodes)
            x_norm_sl = x_norm_all[no_self_loops]
            self.w[no_self_loops] = - 0.5*x_norm_sl

            # np.savetxt(self.model_dir+f"/w.csv", self.w, delimiter=",")
        return self.w
            
    def save_Xw_pmatrix(self):      
        """
        Save the variables x, pmatrix and w.
        Since "w" will be already stored by load_or_create matrix we will omit from here
        """
        self.save_var(var = self.X, var_name = "X")
        self.save_var(var = self.pmatrix, var_name = "pmatrix")
        
        if np.atleast_1d(self.get("w")).size > 1:
            self.save_var(var = self.w, var_name = "w")

    def zl_pmatrix_func(self, x = None, set_w_diag = None, save = True):
        # already present but cancelled (pay attention on this) self.X = res.x[:self.n_params_X].reshape(-1, self.dimX)
        if x is None:
            x = self.X
        # create pmatrix and then save x, pmatrix
        self.pmatrix = self.pmatrix_gen(x, set_w_diag = set_w_diag)
        
        # force the saving of x, pmatrix (not zl_pmatrix)
        if save:
            self.save_Xw_pmatrix()
            self.save_var(var = np.array([self.n_edges]), var_name = "n_edges")
            self.save_var(var = self.deg, var_name = "deg")

        # remove the diagonal
        self.zl_pmatrix = self.frmv_diag(adj_mat = self.pmatrix)
        return self.zl_pmatrix

    def reset_net_meas(self):
        from os import remove
        flag = False
        for i in ["deg", "annd", "cc"]:
            full_path = self.model_dir+"/"+f"{i}.csv"
            if os.path.exists(full_path): 
                flag = True
                remove(full_path)
        if flag: print(f"removed deg, annd, cc")
        self.deg, self._annd, self._cc = None, None, None