from ..lib import *
from ..graphs.Graph import Graph
from ..graphs.variables import variables
from ..algorithms.param_rearranger import param_rearranger
from ..algorithms.ensemble_ops import ensemble_ops

class Undirected_Graph(Graph, variables, param_rearranger, ensemble_ops):
    def __init__(self, **kwargs):
        Graph.__init__(self, **kwargs)

        # execute _set_cg_pdtrans and get_graph_basics only if the name is the one of a obs network
        if self.name.endswith(("ING", "net", "Gleditsch")):
            self.pdtrans = self._set_cg_pdtrans(
                                                pdtrans = kwargs.get("pdtrans"), # pdtrans at the micro level to be coarse-grained
                                                distance_matrix = kwargs.get("distance_matrix"), \
                                                lvl_to_nclust = kwargs.get("lvl_to_nclust")
                                                )
            
            # for the WTW we will feed directly the bin_adj
            if np.atleast_1d(kwargs.get("bin_adj")).size > 1:
                self.save_var(var_name = "bin_adj", var = kwargs["bin_adj"])

            self.get_graph_basics(pdf = self.pdtrans)

    def sparse_ndarr_net_meas(self, meas = "deg"):
        # use csr_matrix ONLY for observed network
        # use np.array for the pmatrix
        if meas == "deg":
            if self.kind == "obs":
                return self.zl_bin_adj.sum(axis = 1).astype(int) #.A1 only if bin_adj is a csr_matrix
            else:
                deg = np.sum(self.zl_pmatrix, 1)
                return deg
        elif meas == "annd":
            if self.kind == "obs":
                self.annd = self.deg_filtered_annd()
                return self.annd
            else:
                num_annd = np.sum(self.zl_pmatrix @ self.zl_pmatrix, axis = 1) - np.sum(self.zl_pmatrix**2, 1)
                return 1 + num_annd / self.deg
        
        elif meas == "cc":
            if self.kind == "obs":
                if self.zl_bin_adj.shape[0] < 1e3:
                    self.cc = self.deg_filtered_cc()
                    return self.cc
                return print(f'-Matrix Too Big! Try to reset the 1e3 threshold',)
            else:
                if self.zl_pmatrix.shape[0] < 1e3:
                    self.cc = self.cc_einsum(m = self.zl_pmatrix)
                    return self.cc
                return print(f'-Matrix Too Big! Try to reset the 1e3 threshold',)

    def n_wedges(self, k):
        if self.kind == "obs":
            if k == 1:
                return 1
            else:
                return k*(k-1)

    def cc_einsum(self, m):
        # np.fill_diagonal(m, 0)
        denom = np.einsum('ij,ki->i', m, m)
        corr = np.einsum('ij,ji->i', m, m)
        d = (denom - corr)
        num = np.einsum('ij,jk,ki->i', m, m, m)
        # Use numpy.divide() to handle division by zero directly
        cc = np.divide(num, d, out=np.zeros_like(num), where = d != 0)

        return cc

    def deg_sseq(self, pdf, kind = "undirected"):
        """
        This function provides a way to discover the disconnected self-loops meanwhile calculating the deg and sseq
        Mask self-loops and find the deg. Then, select only the payer_grid_id of nodes without deg = 0
        """

        # mask self-loops 
        mask_self_loops = ~pdf[f"payer_{self.id_code}"].eq(pdf[f"beneficiary_{self.id_code}"])
        pdf = pdf.loc[mask_self_loops]

        # select only interesting columns
        pd_payben_id = pdf.loc[:, [f'payer_{self.id_code}', f'beneficiary_{self.id_code}', "amount"]]
        pd_payben_id.loc[:, "counts"] = 1

        def connected_pd_deg_strength(str_name, pdf):
            # remove the disconnected loops counts, i.e. self-loops only connected to them selves
            pd_deg_strength =  pdf\
                            .sort_values(str_name)\
                            .groupby([str_name])\
                            .sum().reset_index(drop = False).loc[:,[str_name, "amount", "counts"]]
            mask_disconnected_sl = ~(pd_deg_strength.counts == 0)
            return pd_deg_strength.loc[mask_disconnected_sl]

        payer_deg_strength = connected_pd_deg_strength(f'payer_{self.id_code}', pdf = pd_payben_id)
        benef_deg_strength = connected_pd_deg_strength(f'beneficiary_{self.id_code}', pdf = pd_payben_id)

        if kind == "undirected":
            # this would return payer_id_code, self.stre, self.deg
            _, self.stre, self.deg = payer_deg_strength.to_numpy().T
            self.save_var(var_name = "stre", var = self.stre)

        
        # elif kind == "directed":
        #     # this would return payer_id_code, self.stre, self.deg
        #     _, self.stre_out, self.deg_out = payer_deg_strength.to_numpy().T
        #     _, self.stre_in, self.deg_in = benef_deg_strength.to_numpy().T

    def rel_frob_err(self, exp_mat, obs_mat):
        return np.linalg.norm(exp_mat - obs_mat) / np.sqrt(np.sum(obs_mat))