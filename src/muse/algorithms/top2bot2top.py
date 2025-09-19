from ..utils.helpers import *

class top2bot2top():
    def __init__():
        pass

    def topX_2_botX(self, top_X, bot_l, top_l, verbose = False):
        '''
        Fine Grain the top_X @ top_level/picker_level to the nodes in bot_level by means of GDPs
        
        botX[i] = GDP[i] / GDP[I] * topX[I]

        l := "level", lab := "(integer) label of a node"
        
        Motivation:
        The cg_labels are only wrt to the 0th level. 
        Thus, we need to create the mapping ibot_2_Itop (a dictionary) from level A --> 0 (aka "self.cgl0_l(bot_l)") and 0 --> B (aka "self.cgl0_l(top_l)").
        '''
        # import cgl0_l to have the labels at level l
        self._import_funcs_labels_deg()
        #self.n_nodes_botl = self.cgl0_l(bot_l).max() + 1
        self.nodes_botl = np.unique(self.cgl0_l(bot_l))

        # import the (already fitted) fitnesses at the pick_l = top_l
        top_X = top_X.reshape(-1, self.dimX)
        
        gdp0 = pd.read_csv('outputs/datasets/gleditsch/sliced_vars/2000_gled_df.csv', usecols = ["gdp_o"])["gdp_o"].unique()
        
        # To fraction with GDPs, find the micronodes ending up in the "lab"/"Itop" block-node, i.e. the different labels in cgl0_l
        # In the theory, it is the Omega s.t. Omega(ibot) = Itop
        where_lab_l = lambda l, lab: np.where(self.cgl0_l(l) == lab)[0]

        # fixing bot_l, we end up with  where_array[i] = bot_lab[i] = Ilab
        bot_lab2idx = lambda bot_lab: where_lab_l(bot_l, bot_lab)
        
        # create the mapping bot_lab --> 1st micro-index: bot_lab2idx(bot_lab)[0] --> top_lab. 
        blab_2_tlab = lambda bot_lab: self.cgl0_l(top_l)[bot_lab2idx(bot_lab)[0]]

        # define the ratio_gdps and rep_top_X functions to go from input bot_node to either the ratio GDP[i] / GDP[Omega(i)] or the top_X[Omega(i)]
        
        # roll blab_2_tlab over all the ibot = bot_lab
        # this dict is mandatory since we need to replicate the top_X for all the micro-nodes in the same block-node
        ibot_2_Itop = {ibot : blab_2_tlab(ibot) for ibot in self.nodes_botl}

        # # Find the bot_gdp and top_gdp := sum(gdp0) that will be used in fractioning
        lab2gdp_l = lambda l: {node_lab : np.sum(gdp0[where_lab_l(l, node_lab)]) for node_lab in np.arange(self.cgl0_l(l).max() + 1)}
        bot_gdp = lab2gdp_l(bot_l)
        top_gdp = lab2gdp_l(top_l)
        self.top_gdp = top_gdp
        apply_over_ibot = lambda func: np.array(list(map(lambda ibot: func(ibot), self.nodes_botl)))

        # create GDP[i] / GDP[Omega(i)]
        self.bot_on_top_gdp = lambda bot_node: bot_gdp[bot_node] / top_gdp[ibot_2_Itop[bot_node]]
        self.ratio_gdps = apply_over_ibot(lambda bot_node: self.bot_on_top_gdp(bot_node)).reshape(-1, 1)

        # top_X[Omega(i)]
        self.rep_top_X = apply_over_ibot(lambda bot_node: top_X[ibot_2_Itop[bot_node]]).reshape(-1, self.dimX)
        
        # multiply every entry as in the theoretical formula
        # If multiple components of top_X the same factor is applied to all of them
        bot_X = self.ratio_gdps*self.rep_top_X

        return bot_X
            
    def cglhigh_l(self, top_level, verbose = False):
        '''
        Function which convert the self.level as the 0th level by converting the labels at the dendro-level which are the labells reffering to.
        E.g. the bot_level "0" is composed by the dendro-level nodes [12, 60, 94]. 
        Thus, the micro-nodes in "0" were [ 12  57  60  94 117] but at level 2 becomes [0, 57, 117]

        Then, replicate the top_lab for the indexes of the members of the same top_lab, e.g. at bot_lab([0, 57, 117]) = [0,0,0]
        '''
        if self.level == top_level:
            return np.arange(self.n_nodes)

    def cg_labels_l(self, l):
        '''
        The dendrogram labels refer to the 0-th level.
        Thus, to have bum-maxlMSM starting from level l, we need to convert the labels from the 0-th level to the l-th level.
        '''
        if self.level == 0:
            self._import_funcs_labels_deg() #cgl0_l = lambda l: np.genfromtxt(self.Gleditsch_dirl(l)+"/Gleditsch.cg_labels.csv", dtype = int)
            return self.cgl0_l(l)