from dependencies import *

class variables():
    """
    This is class is meant only to store the variables that are used in the Graph class. Moreove, it should not have any methods, only properties.
    Roughly, a property is a method that is accessed as an attribute, i.e. without parentheses. 
    Therefore, we will call it as a normal variable, i.e. without paranthesis, but it will be computed on the fly. 
    More precisely, via the function ```load_or_create``` we will load the variable if has been previously stored in the Graph directory or create it if it does not.
    As a concrete example, if you have the pmatrix. Then, in principle, you may access ```class.deg, class.annd, class.cc``` and you will have them ready to be plotted.

    Another, PRO of having a property is that maybe that observed quantity is not needed in the current analysis, so we do we will never compute it.

    Important: each variable here will set to None the variables when __init__ from variables.py.
    Therefore, if we copy ``n_nodes`` in a Child class from obs_net and then, variables.__init__, the _n_nodes will be set to None.
    """
    def __init__(self): 
        self._wei_adj = None
        self._zl_wei_adj = None
        self._bin_adj = None
        self._zl_bin_adj = None
        self._deg = None
        self._stre = None
        self._annd = None
        self._cc = None
        self._n_edges = None
        self._n_nodes = None
        self._X = None
        self._w = None
        self._n_strineq_nodes = None  
        self._n_params_X = None
        self._n_params = None  

    @property
    def wei_adj(self):
        '''
        The weighted adjacency matrix. 
        Load or create just check for its existence and load it or it creates it and save it!
        '''
        if np.array_equal(self._wei_adj, None):
            self._wei_adj = self.load_or_create(var_name = "wei_adj")

        return self._wei_adj

    @wei_adj.setter
    def wei_adj(self, value):
        '''@setter method for `self.wei_adj = somthing` to be defined to access the _wei_adj variable'''
        self._wei_adj = value
    
    @property
    def zl_wei_adj(self):
        """The zero loop weighted property. No save"""
        if np.array_equal(self._zl_wei_adj, None):
            self._zl_wei_adj = self.frmv_diag(adj_mat = self.wei_adj)
        return self._zl_wei_adj

    @zl_wei_adj.setter
    def zl_wei_adj(self, value):
        self._zl_wei_adj = value

    @property
    def bin_adj(self):
        '''
        The binary adjacency matrix. 
        In the code "undirected_graph.bin_adj" will recall this method and filling JIT the variable not at the __init__ or via using a method with "()" on it.
        So, for this feature, we should check if the _bin_adj has been already filled. If not, `load_or_create` it!
        '''
        if np.array_equal(self._bin_adj, None):
            self._bin_adj = self.load_or_create(var_name = "bin_adj")

        return self._bin_adj

    @bin_adj.setter
    def bin_adj(self, value):
        '''@setter method for `self.bin_adj = somthing` to be defined to access the _bin_adj variable'''
        self._bin_adj = value

    @property
    def zl_bin_adj(self):
        """The zero loop binary property. No save"""
        if np.array_equal(self._zl_bin_adj, None):
            self._zl_bin_adj = self.frmv_diag(adj_mat = self.bin_adj)
        return self._zl_bin_adj

    @zl_bin_adj.setter
    def zl_bin_adj(self, value):
        self._zl_bin_adj = value

    @property
    def deg(self):
        """The degree sequence property."""
        if np.array_equiv(self._deg, None):
            self._deg = self.load_or_create("deg")

        return self._deg
    
    @deg.setter
    def deg(self, value):
        self._deg = value

    @property
    def stre(self):
        """The strength sequence property."""
        if np.array_equiv(self._stre, None):
            self._stre = self.load_or_create("stre")

        return self._stre
    
    @stre.setter
    def stre(self, value):
        self._stre = value

    @property
    def n_nodes(self):
        '''
        size of deg
        '''
        if np.array_equiv(self._n_nodes, None):
            self._n_nodes = self.deg.size

        return self._n_nodes

    @n_nodes.setter
    def n_nodes(self, value):
        '''@setter method for `self.bin_adj = somthing` to be defined to access the _bin_adj variable'''
        self._n_nodes = value

    @property
    def n_strineq_nodes(self):
        '''
        size of deg
        '''
        if np.array_equiv(self._n_strineq_nodes, None):
            self.n_strineq_nodes = self.n_nodes
            if self.get("reduced_by"):
                self.n_strineq_nodes = self.strineq_nodes.size

        return self._n_strineq_nodes

    @n_strineq_nodes.setter
    def n_strineq_nodes(self, value):
        '''@setter method for `self.bin_adj = somthing` to be defined to access the _bin_adj variable'''
        self._n_strineq_nodes = value
    
    @property
    def n_edges(self):
        '''
        The binary numb_of_edges
        Use np.sum(A) / 2 since np.sum(np.triu(A, k = 1)) ~ L with 1e-10 error
        '''
        if np.array_equiv(self._n_edges, None):
            self._n_edges = self.load_or_create("n_edges")

        return self._n_edges

    @n_edges.setter
    def n_edges(self, value):
        '''@setter method for `self.bin_adj = somthing` to be defined to access the _bin_adj variable'''
        self._n_edges = value

    @property
    def annd(self):
        """The average nearest neighbor degree property."""
        if np.array_equiv(self._annd, None):
            self._annd = self.load_or_create("annd")
        return self._annd

    @annd.setter
    def annd(self, value):
        self._annd = value

    @property
    def cc(self):
        """The binary clustering coefficient property."""
        if np.array_equiv(self._cc, None):
            self._cc = self.load_or_create("cc")
        return self._cc
    
    @cc.setter
    def cc(self, value):
        self._cc = value
    
    @property
    def X(self):
        '''
        The binary adjacency matrix. 
        In the code "undirected_graph.bin_adj" will recall this method and filling JIT the variable not at the __init__ or via using a method with "()" on it.
        So, for this feature, we should check if the _bin_adj has been already filled. If not, `load_or_create` it!
        '''

        if np.array_equal(self._X, None):
            dim_X = self.dimB + self.dimC if self.name.endswith("LPCA") else self.dimX
            self._X = self.load_or_create("X").reshape(-1, dim_X)
        
        return self._X

    @X.setter
    def X(self, value):
        '''@setter method for `self.X = something` to be defined to access the _X variable'''
        self._X = value

    @property
    def w(self):
        '''
        The binary adjacency matrix. 
        In the code "undirected_graph.bin_adj" will recall this method and filling JIT the variable not at the __init__ or via using a method with "()" on it.
        So, for this feature, we should check if the _bin_adj has been already filled. If not, `load_or_create` it!
        '''

        if np.array_equal(self._w, None):
            self._w = self.load_or_create(var_name = "w")
        
        return self._w

    @w.setter
    def w(self, value):
        '''@setter method for `self.w = something` to be defined to access the _w variable'''
        self._w = value

    @property
    def delta(self):
        '''
        The binary adjacency matrix. 
        In the code "undirected_graph.bin_adj" will recall this method and filling JIT the variable not at the __init__ or via using a method with "()" on it.
        So, for this feature, we should check if the _bin_adj has been already filled. If not, `load_or_create` it!
        '''

        if np.array_equal(self._delta, None):
            self._delta = self.load_or_create(var_name = "delta")
        
        return self._delta

    @delta.setter
    def delta(self, value):
        '''@setter method for `self.delta = something` to be defined to access the _delta variable'''
        self._delta = value

    @property
    def pmatrix(self):
        '''
        The binary adjacency matrix. 
        In the code "undirected_graph.bin_adj" will recall this method and filling JIT the variable not at the __init__ or via using a method with "()" on it.
        So, for this feature, we should check if the _bin_adj has been already filled. If not, `load_or_create` it!
        '''

        if np.array_equal(self._pmatrix, None):
            self._pmatrix = self.load_or_create(var_name = "pmatrix")
        
        return self._pmatrix

    @pmatrix.setter
    def pmatrix(self, value):
        '''@setter method for `self.pmatrix = somthing` to be defined to access the _pmatrix variable'''
        self._pmatrix = value
        
    @property
    def zl_pmatrix(self):

        if np.array_equal(self._zl_pmatrix, None):
            self._zl_pmatrix = self.frmv_diag(adj_mat = self.pmatrix)
        
        return self._zl_pmatrix

    @zl_pmatrix.setter
    def zl_pmatrix(self, value):
        '''@setter method for `self.zl_pmatrix = something` to be defined to access the _zl_pmatrix variable'''
        self._zl_pmatrix = value

    @property
    def n_params_X(self):
        '''calculate the number of parameters of the model'''

        # set the reduced number of nodes, i.e. n_strineq_nodes
        # if one enforces the reduced model, then self.n_strineq_nodes
        # Otherwise, it is the same as self.n_nodes

        if not self._n_params_X:
            # if self.name.endswith("LPCA"):
            #     return self.n_strineq_nodes * self.dimBCX
            # else:
            #     self._n_params_X = self.n_strineq_nodes * self.dimX
            #     # self._n_params_X = self.n_repr_nodes * self.dimX
            #     # self._n_params_X = (self.n_nodes - repr_exists(self.fc_nodes) - repr_exists(self.fd_nodes)) * self.dimX
            #     return self._n_params_X
            return self.n_strineq_nodes * self.dimBCX

        else:
            return self._n_params_X
    
    @n_params_X.setter
    def n_params_X(self, value):
        '''@setter method for `self.bin_adj = somthing` to be defined to access the _bin_adj variable'''
        self._n_params_X = value

    @property
    def n_params(self):
        '''calculate the number of parameters of the model'''

        # set the reduced number of nodes, i.e. n_strineq_nodes
        # if one enforces the reduced model, then self.n_strineq_nodes
        # Otherwise, it is the same as self.n_nodes

        if not self._n_params:
            if self.name.endswith(("maxlMSM", "degcMSM")):
                return self.n_params_X + self.n_strineq_nodes
            else:
                return self.n_params_X

        else:
            return self._n_params
    
    @n_params.setter
    def n_params(self, value):
        '''@setter method for `self.bin_adj = somthing` to be defined to access the _bin_adj variable'''
        self._n_params = value

    

    