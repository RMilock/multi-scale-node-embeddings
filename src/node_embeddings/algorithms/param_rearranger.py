from ..lib import *

class param_rearranger():
    def __init__():
        pass

    def model_dir_level(self, ref_model, level = 0, var_name = "X"):
        """load the directory for the coarse-grained X"""
        
        _model_dir_level = ref_model.model_dir.replace(f"level{ref_model.level:g}", f"level{level}") 
        return _model_dir_level+f"/{var_name}.csv"

    def sum_Xw_prevl(self, obs_net, ref_model):
        """Sum the parameters of the members populating a block-node"""

        # Load the mic_X
        mic_dir = lambda X: self.model_dir_level(ref_model = ref_model, var_name = X)
        mic_X = np.genfromtxt(mic_dir("X"), delimiter=',')

        # recover the microscopic mic2mac_int_ generated from the agglomerative clustering
        # DO this for WTW, but not for the ing net since it is more efficient to sum the one of the previous layer
        mic2mac_int = obs_net.mic2mac_int
        
        # calculate the row-wise sum, since each column is a different feature
        sum_X_ = lambda mic_Xw, mic2mac_int: np.array([np.sum(mic_Xw[mic2mac_int == c], 0) for c in np.unique(mic2mac_int)])

        # find full_sum_X and strineq_X for full network and fitnesses comparison resp.
        dims = self.dimB + self.dimC if self.name.endswith("LPCA") else self.dimX
        self.X = sum_X_(mic_X, mic2mac_int).reshape(-1,dims)

        set_w_diag = False
        if self.name.endswith(("maxlMSM", "degcMSM")):
            set_w_diag = "diag"
            mic_w = np.genfromtxt(mic_dir("w"), delimiter=',')
            self.w = sum_X_(mic_w, mic2mac_int)

        # create the zl_matrix
        self.zl_pmatrix_func(x = self.X, set_w_diag = set_w_diag)

    def streq_nodes_deg(self, sort_dict = False):
        """
        Create the dictionary of nodes with the same degree for the reduced obs_deg, i.e. w/o deterministic nodes.
        Strangely, set(self.deg) sorts the deg
        If sorted (not mandatory), the key may be (as we are using ``set``) in 1-1 correspondence with the self.deg array.

        Calling orig_obs_deg, as we will remove the deterministic nodes from self.obs_deg. 
        Thus, self.obs_deg <-- self.obs_deg[no_det_nodes] - self.n_det_nodes

        !!! n_nodes_per_class: n_nodes per class will be the streq_class remaining after the removal of fc and fd nodes
        """
        
        # if self.reduce == False, setting n_nodes_per_class to 1 for each node
        self.n_nodes_per_class = [1]*self.n_nodes

        # remove the deterministic nodes from the deg and create the mapping from original idx and reduced (w/o det) ones
        # e.g. node 7 --> 3 after det removal
        orig_obs_deg = self._map_red2orig2red_indexes(obs_deg = self.obs_deg) #self.obs_deg.copy()
        
        # self.fc_nodes = np.where(orig_obs_deg == (self.n_nodes - 1))[0]
        # self.fd_nodes = np.where(orig_obs_deg == 0)[0]
        # self.n_fcfd_nodes = self.n_fcfd_nodes + self.fd_nodes.size

        # find the representative of each class of "same degree" nodes s.t. we will optimize only over them
        self.streq_nodes = {}
        self.strineq_nodes = list()
        set_red_obs_deg = set(orig_obs_deg)

        # populate the dictionary with original idx of the streq_nodes
        for k in set_red_obs_deg:
            idx_k = np.where(orig_obs_deg == k)[0] # here I am taking the index of the original deg which is ordered as the nodes not set(obs_deg)
            self.streq_nodes[idx_k[0]] = idx_k
            self.strineq_nodes.append(idx_k[0])
            self.n_nodes_per_class.append(idx_k.size)

        if sort_dict:
            _sort = lambda dict_: {k: dict_[k] for k in sorted(dict_)}
            self.streq_nodes = _sort(self.streq_nodes)
            sorted_idx = np.argsort(self.strineq_nodes)
            self.strineq_nodes = np.array(self.strineq_nodes)[sorted_idx]
            self.n_nodes_per_class = np.array(self.n_nodes_per_class)[sorted_idx]
        
        else:
            # convert the list to a numpy array
            self.strineq_nodes = np.array(self.strineq_nodes)
            self.n_nodes_per_class = np.array(self.n_nodes_per_class)

    def wrappper_streq_copy(self, func):
        def inner(x, A, verbose):
            if self.get("reduced_by"):
                # remember that the x is not flat_X
                x = self.repeat_streq_X(x)
                return func(x, A, verbose)
            else:
                return func(x, A, verbose)
        return inner

    def repeat_streq_X(self, red_X):
        """
        This function has 2 important properties:
        1) Repeating the streq_X only for the not-deterministic nodes. Remember that the model is the reduced one w/o fc and fd nodes

        2) it also put the red_X at the original nodes we have in the self.obs_deg.

        More precisely, in the streq_nodes we are not sorting the dictionary. Therefore the first element could be different from the node 0, e.g. 54.
        Hence, even the entries of red_X MUST be redirected to the proper degree they want to fit: reass_X[54] = x[0]
        In this way, we can compare it this self.obs_deg which has the degrees ordered as in the observed network
        """

        # select only the not fc_/fd_ nodes to reassign the fitnesses
        # old: n_notdet_nodes = self.strineq_nodes.max()+1
        reass_X = np.zeros((self.n_notdet_nodes, self.dimBCX))

        red_X = self.flat_to_mat(red_X)
        for red_i, x_i in zip(self.streq_nodes, red_X):

            streq_nodes_red_i = self.streq_nodes[red_i]
            reass_X[streq_nodes_red_i] = x_i
        return reass_X

    def set_fcfd_X(self):
        """
        Set the fitnesses for the complete model, i.e. with fc and fd nodes
        Note: I did not used ```np.insert``` since it may produce couter-intuitive results 
        e.g. np.insert([1,2], [0,1], 46) --> array([46, 1, 46, 2])
        """
        
        # create the full x matrix
        x = np.zeros(self.n_nodes * self.dimBCX).reshape(self.n_nodes, self.dimBCX)

        # find the index of the nodes which are not fc or fd
        all_idx_streq_nodes = np.setdiff1d(np.arange(self.n_nodes), np.concatenate([self.fc_nodes, self.fd_nodes]))

        # assign the fitnesses of the structurally equivalent nodes
        x[all_idx_streq_nodes] = self.X

        # assign the fitnesses of the fc
        if self.fc_nodes.size > 0:
            # to avoid dividing by zero
            min_X = x[x>0].min()
            low_fcx = 38/min_X # 38 is the lowest value s.t. -np.expm1(-38) ~ 0 due to machine precisione
            x[self.fc_nodes] = [low_fcx]*self.dimBCX
        
        # they are mutually exclusive. Thus, the index (self.fd_nodes[0]) could be left untouch
        if self.fd_nodes.size > 0:
            x[self.fd_nodes] = [0]*self.dimBCX

        if self.get("only2fc_nodes"):
            x = np.insert(x, self.only2fc[0], [38/low_fcx]*self.dimBCX, axis = 0)

        return x

    def flat_to_matBC(self, flat_X):
        """ Returns the factos B,C from the flat_X """
        
        if flat_X.ndim == 1:
            # vectorialize flat_X by respecting the order of appearance, 
            # i.e. the first :self.n_nodes*self.dimB should be for the B parameters
            # and the rest for the C parameters
            flat_X = flat_X.reshape(-1, self.dimBCX) #, order = "F")
            
        return flat_X[:, :self.dimB], flat_X[:, self.dimB:]
    
    def mat_to_flat(self, mat_X):
        """ Returns the flat_X version of mat_X """
        if mat_X.ndim == 1:
            return mat_X
        else:
            return mat_X.ravel()

    def flat_to_mat(self, flat_X):
        """ Convert the flat_X to a matrix if it wasn't already """
        if flat_X.ndim == 1:
            return flat_X.reshape(-1, self.dimBCX)
        
        elif flat_X.ndim == 2:
            return flat_X
        else:
            print(f'-Multidimensional tensor, i.e. flat_X.shape: {flat_X.shape}',)

    def _map_red2orig2red_indexes(self, A = None, obs_deg = None):
        """
        Since fc, fd and only2fc may be considered as deterministic nodes. Remove from the statistics one uses, i.e. obs_deg or zero self-loops A. We will use the zero self-loops A since we will be looking at the interactions with the other nodes.
        Remember that the self-loops for if we observe the whole A will be fitted after the off-diagonal fitnesses.

        Moreover,
        -- for degree-structural equivalence, subtrackt from the deg the deterministic nodes, i.e. only the fully-connected. 
        Indeed, fit the "diminished" number of neighbors while recovering the deterministic connections, with fully-connected, afterwards;
        -- for neighbors-structural equivalence, fit the adjacency matrix without the deterministic nodes;

        As a by product, create a map from original indexes to the obs_deg w/o the fc- and fd- nodes
        Return no_fcfd_only2fc_A obs_deg or adjacency matrix A

         
        """
        
        
        if self.reduced_by == "deg":
            # deg = 1 does not imply that it is a only2fc_node. Therefore, if only deg is available, only2fc_node = []
            self.only2fc_nodes = np.array([])

            # remove from the self.obs_deg the fc and fd nodes from the deg as they are deterministic
            self.obs_deg = obs_deg[(obs_deg > 0) & (obs_deg < self.n_nodes - 1)] - self.n_fcfd_nodes
            set_red_obs_deg = self.obs_deg #set(self.obs_deg)
        
        else:     
            # concatenate fc and fd as initial deterministic nodes
            det_nodes = np.concatenate([self.fc_nodes, self.fd_nodes])

            # remove the fc and fc nodes
            mask = np.ones(A.shape[0], dtype = bool)
            mask[det_nodes] = False
            no_fcfd_A = A[mask][:, mask]
            
            # remove the fcfd even from mask since we will use it to filter the no_fcfd_A which has, indeed, less entries than mask
            mask = mask[mask]

            # find the only2fc_nodes as lowest fitness and remove them from no_fcfd_A
            self.only2fc_nodes = np.where(np.all(no_fcfd_A, axis=1))[0]
            mask[self.only2fc_nodes] = False
            no_fcfd_only2fc_A = no_fcfd_A[mask][:, mask]
        
        # as we are not optimizing over the fc and fd nodes, we need to keep track of the original index, to reassign the fitnesses afterwards
        # Therefore, expect in the dict not to have the fc and fd indexes but only the relative shifting into the idx of the remaining nodes
        remaining_nodes = np.setdiff1d(np.arange(self.n_nodes), np.concatenate([self.fc_nodes, self.fd_nodes, self.only2fc_nodes]))
        self.notdet_red2orig_idx = {i : remaining_nodes[i] for i in range(remaining_nodes.size)}
        self.notdet_orig2red_idx = {remaining_nodes[i] : i for i in range(remaining_nodes.size)}

        if self.reduced_by == "deg": return set_red_obs_deg
        else: return no_fcfd_only2fc_A

    def streq_nodes_neigh(self, A):             
        """
        Find the nodes that have the same neighbors, i.e. "structurally equivalence" wrt the "neighboring class".
        Use them to replicate the fitnesses for each node in the equivalent class.
        However, to solve the problem use the full-adjacency matrix and not the "reduced" one.

        Find also the only2fc_nodes as they will have the lowest fitnesses to connect only to fc_nodess
        """
        
        from functools import reduce

        # starting by removing deterministic nodes
        if self.n_fcfd_nodes > 0:
            A = self._map_red2orig2red_indexes(A = A)

        #calculate the nearest neighbors of each node
        near_neigh = {node : np.where(A[node])[0] for node in np.arange(A.shape[0])}

        def nn_of_rmv(key_node, int_n):
            """
            Nearest neighbors of key without int_n
            E.g., for symmetry of nodes 0,1, if exists the link (0,1) also (1,0) exists but near_neigh[0] = [1, NN] and near_neigh[1] = [0, NN]
            Therefore, discard each other
            """

            # select the near_neigh of key_node
            neigh_list = near_neigh[key_node]

            # remove the int_n from the list
            return neigh_list[neigh_list != int_n]

        self.streq_nodes = {}
        self.strineq_nodes = list()
        self.n_nodes_per_class = []
        
        # fill the dictionary of structural equivalent nodes
        self.only2fc = []
        ls_keys = list(near_neigh.keys())

        while len(ls_keys) > 0:
            key_node = ls_keys[0]

            # find the nodes that are structural equivalent to key_node
            # str_eqn_kn = [nn for nn in near_neigh if np.array_equal(rmv_near_neigh_(nn, key_node), rmv_near_neigh_(key_node, nn))]
            str_eqn_kn = [nn for nn in near_neigh if np.array_equal(nn_of_rmv(key_node, nn), nn_of_rmv(nn, key_node))]
            self.streq_nodes[key_node] = str_eqn_kn
            self.strineq_nodes.append(key_node)

            #append the number of nodes in that class
            self.n_nodes_per_class.append(len(str_eqn_kn))
            
            # remove already str_eqn_kn
            if ls_keys != []:
                ls_keys = list(reduce(lambda x,y : filter(lambda z: z!=y , x), str_eqn_kn, ls_keys))

        self.n_nodes_per_class = np.array(self.n_nodes_per_class)
        self.strineq_nodes = np.array(self.strineq_nodes)

        # print the number and ratio of structural equivalent nodes only for fitted model
        if not self.name.startswith(("sum", "fine")):
            print(f'-self.strineq_nodes.size / self.n_nodes: {self.strineq_nodes.size, self.n_nodes} ~ {np.round(self.strineq_nodes.size / self.n_nodes, 3) * 100} %',)
        
        self.zl_notdet_A = A