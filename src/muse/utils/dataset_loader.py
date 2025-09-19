from ..utils.helpers import nodes_from
from ..lib import *

def default_4_undirected_paper(dataset_name):
    ''' Default values for the first 2 undirected papers'''
    
    if "Gleditsch" in dataset_name:
        id_code = "int"
        cg_method = "geo-dist"
        year = 2000
    
    return id_code, cg_method, year

def flip_payer_beneficiary_columns(df):
	"""
    Flips the values between 'payer_...' and 'beneficiary_...' columns

    Finds columns starting with 'payer_' and checks if a corresponding
    'beneficiary_' column with the same suffix exists. 
	If both exist, the values in these two columns are swapped for every row.
	If not, an error arise.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with the columns flipped where applicable.
                      Returns a copy of the original if no pairs are found.
    """

	# Extract all suffixes from payer and beneficiary columns
	# 
	suffixes = set()
	for col in df.columns:
		if col.startswith('payer_'):
			suffix = col.split('payer_', 1)[1]
			suffixes.add(suffix)
		elif col.startswith('beneficiary_'):
			suffix = col.split('beneficiary_', 1)[1]
			suffixes.add(suffix)
	
	# Swap values for each payer-beneficiary column pair
	for suffix in suffixes:
		payer_col = f'payer_{suffix}'
		beneficiary_col = f'beneficiary_{suffix}'
		if payer_col in df.columns and beneficiary_col in df.columns:
			df[payer_col], df[beneficiary_col] = df[beneficiary_col], df[payer_col]
	
	return df

def dataset_loader(name, id_code = None, cg_method = "geo-dist", year = 2000, direction = "Undirected", distance_matrix = None, lvl_to_nclust = None, max_n_entries = 0):

    # dataset name year and direction (nyd)
    name_year_direction = f"{name}-{year}-{direction.title()}"
    dataset_folder = f"./data/" + name_year_direction

    if "Gleditsch" in dataset_folder:
        pdtrans = pd.read_csv(dataset_folder + "/edges_int_pd.csv")
        print(f"\nReading from local source @ {dataset_folder + '/edges_int_pd.csv'}")

    # COARSE-GRAINING METHODS   
    if cg_method.startswith(("geo-dist", "rand-dist")):
        pdtrans.columns = ["payer_int", "beneficiary_int", 'amount']

        # define the diminishing rate of nodes from level to level + 1
        n_nodes_0 = nodes_from(pdtrans, id_code = 'int').size
        delta_n_clust = 30
        
        # assuming that a network with a number of nodes < lowest_n_nodes (e.g. 10) does not makes sense
        # find level l : N - l * r > c (N = n_nodes_0, r = delta_n_clust, c = lowest_n_nodes)
        lowest_n_nodes = 0
        n_clust_seq = [n_nodes_0 - l * delta_n_clust for l in range(1 + (n_nodes_0 - lowest_n_nodes) // delta_n_clust )]
        
        # num of levels is how many n_clust_seq before lowest_n_nodes
        total_levels = len(n_clust_seq)
        lvl_to_nclust = {k:v for k, v in enumerate(n_clust_seq)}

        if cg_method.startswith(("geo-dist")):
            # load the distance matrix, i.e. the geographical distance ones
            distance_matrix = pd.read_csv(dataset_folder + f"/geo_dist.csv", index_col = 0)
        

    elif cg_method.startswith("random"):
        total_levels = 1

    if max_n_entries > 0:
        pdtrans = pdtrans.iloc[:max_n_entries]

    kwargs = {
            "name" : name,
            "dataset_name" : name_year_direction,
            "direction" : direction,
            "level" : 0,
            "pdtrans" : pdtrans.copy(),
            "id_code" : id_code,
            "year" : year,
            "cg_method" : cg_method, # needed for defining the directory
            # "micro_id_codes" : micro_id_codes,
            # these 2 lines works only for random partitioning, i.e. "id_code" : "grid_id"
            "distance_matrix" : distance_matrix, "lvl_to_nclust" : lvl_to_nclust,
            "dataset_folder" : dataset_folder,
            }

    return pdtrans, kwargs, total_levels