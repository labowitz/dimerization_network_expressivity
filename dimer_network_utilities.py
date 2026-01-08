import eqtk
import itertools
import math
import networkx as nx
import numpy as np
import os
import pandas as pd
import scipy.stats

# Plotting
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib import rcParams

# For version tracking
# from https://stackoverflow.com/questions/40428931/package-for-listing-version-of-packages-used-in-a-jupyter-notebook
import types
def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, 
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]
            
        # Some packages are weird and have different
        # imported names vs. system/pip names. Unfortunately,
        # there is no systematic way to get pip names from
        # a package's imported name. You'll have to add
        # exceptions to this list manually!
        poorly_named_packages = {
            "PIL": "Pillow",
            "sklearn": "scikit-learn"
        }
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]
            
        yield name


# For plotting in general
def set_spine_linewidth(ax, linewidth):
    """
    Set linewidth of axes spines.
    """
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(linewidth)   

def set_simple_log_ticks(ax,which,log_lowerbound,log_upperbound,max_labels=4,fontname='Helvetica',fontsize=9):
    '''
    Set ticks for a simple log axis.

    Parameters
    ----------
    ax : matplotlib axis
    which: str
        'x', 'y', or 'both'
    log_lowerbound : int
        Lower bound of log axis, in log10 scale
    log_upperbound : int
        Upper bound of log axis, in log10 scale
    max_labels : int
        Maximum number of labels to use, for even number of ticks. Default is 4.
    fontname : str
        Font to use. Default is 'Helvetica'.
    fontsize : int
        Font size to use. Default is 9.
    '''
    if (log_upperbound-log_lowerbound)%2==0:
        xtick_label_ids = np.linspace(0,log_upperbound-log_lowerbound,max_labels-1)
    else:
        xtick_label_ids = np.linspace(0,log_upperbound-log_lowerbound,max_labels)
    minor_ticks = list(itertools.chain.from_iterable([[(10**float(x))*y for y in np.arange(2,10)] for \
                        x in range(log_lowerbound,log_upperbound)]))+[10**float(log_upperbound)]
    if which=='x' or which=='both':
        ax.set_xticks(np.logspace(log_lowerbound,log_upperbound,log_upperbound-log_lowerbound+1),labels=[f'$10^{{{log:.0f}}}$' if i in xtick_label_ids \
                                                            else '' for i,log in enumerate(np.linspace(log_lowerbound,log_upperbound,log_upperbound-log_lowerbound+1))],\
                                                                fontname=fontname,fontsize=fontsize)
        ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_ticks))
    if which=='y' or which=='both':
        ax.set_yticks(np.logspace(log_lowerbound,log_upperbound,log_upperbound-log_lowerbound+1),labels=[f'$10^{{{log:.0f}}}$' if i in xtick_label_ids \
                                                            else '' for i,log in enumerate(np.linspace(log_lowerbound,log_upperbound,log_upperbound-log_lowerbound+1))],\
                                                                fontname=fontname,fontsize=fontsize)
        ax.yaxis.set_minor_locator(mticker.FixedLocator(minor_ticks))

# For dimer networks now

def make_nXn_dimer_reactions(m):
    """
    Generate all pairwise reactions for m monomer species to be parsed by EQTK.

    Parameters
    ----------
    m : int
        Number of monomer species.

    Returns
    -------
    reactions : string
        Set of dimerization reactions for specified numbers of monomers, one reaction
        per line.
    """
    combinations = list(itertools.combinations_with_replacement(range(m), 2))
    reactions = [f'M_{i+1} + M_{j+1} <=> D_{i+1}_{j+1}\n' for i,j in combinations]
    return ''.join(reactions).strip('\n')

def make_nXn_species_names(m):
    """
    Enumerate names of monomers and dimers for ordering stoichiometry matrix of nXn rxn network

    Parameters
    ----------
    m : int
        Number of monomer species.

    Returns
    -------
    names : list, length (m(m+1)/2 + m)
        List where each element (string) represents a reacting species.
    """
    monomers = [f'M_{i+1}' for i in range(m)]
    combinations = itertools.combinations_with_replacement(range(m), 2)
    dimers = [f'D_{i+1}_{j+1}' for i,j in combinations]
    return monomers + dimers

def make_nXn_stoich_matrix(m):
    """
    For the indicated number of monomers, generate stochiometry matrix for dimerization reactions.
    Parameters
    ----------
    m : int
        Number of monomer species.

    Returns
    -------
    N : array_like, shape (m * (m-1)/2 + m, m(m+1)/2 + m)
        Array where each row corresponds to distinct dimerization reaction
        and each column corresponds to a reaction component (ordered monomers then dimers)
    """
    reactions = make_nXn_dimer_reactions(m)
    names = make_nXn_species_names(m)
    N = eqtk.parse_rxns(reactions)
    return N[names].to_numpy()

def number_of_dimers(m):
    """
    Calculate the number of distinct dimers from input number (m) of monomers.
    """
    return int(m*(m+1)/2)

def number_of_heterodimers(m):
    """
    Calculate the number of distinct heterodimers from input number (m) of monomers.
    """
    return int(m*(m-1)/2)

def number_of_species(m):
    """
    Calculate the number of monomers + dimers from input number (m) of monomers
    """
    return m + number_of_dimers(m)

def get_dynamic_range(data):
    # Get dynamic range of data
    return np.max(data)/np.min(data)

def make_C0_grid(m=2, M0_min=-3, M0_max=3, num_conc=10):
    """
    Construct grid of initial monomer and dimer concentrations.
    Initial dimer concentrations set to 0.

    Parameters
    ----------
    m : int
        Number of monomer species. Default is 2.
    M0_min : array_like, shape (m,) or (1,)
        Lower limit of the monomer concentration in log10 scale.
        Scalar values set the same limit for all ligands. Default is -3,
        corresponding to a lower limit of 10^-3.
    M0_max : array_like, shape (m,) or (1,)
        Upper limit of the monomer concentration in log10 scale.
        Scalar values set the same limit for all ligands. Default is 3,
        corresponding to a lower limit of 10^3.
    num_conc : array_like, shape (m,) or (1,)
        Number of concentrations for each monomer, sampled logarithmically.
        Scalar values set the same limit for all ligands. Default is 10.

    Returns
    -------
    C0 :  array_like, shape (numpy.product(n_conc), (m(m+1)/2 + m)) or (n_conc ** m, (m(m+1)/2 + m))
        Each row corresponds to distinct set of species concentrations.

    """
    num_dimers = number_of_dimers(m)
    if np.size(M0_min) == 1 and np.size(M0_max) == 1 and np.size(num_conc) == 1:
        titration = [np.logspace(M0_min, M0_max, num_conc)]*m
    elif np.size(M0_min) == m and np.size(M0_max) == m and  np.size(num_conc) == m:
        titration = [np.logspace(M0_min[i], M0_max[i], num_conc[i]) for i in range(m)]
    else:
        raise ValueError('Incorrect size of M0_min, M0_max, or num_conc.')
    titration = np.meshgrid(*titration, indexing='ij')
    M0 = np.stack(titration, -1).reshape(-1,m)
    return np.hstack((M0, np.zeros((M0.shape[0], num_dimers))))

def make_Kij_names(m, n_input = 2, rxn_ordered = True):
    """
    Create Kij names for ordering parameters
    """
    n_accesory = m - n_input
    if rxn_ordered:
        return [f'K_{i[0]}_{i[1]}' for i in itertools.combinations_with_replacement(range(1, m+1), 2)]
    else:
        names = []
        #add input homodimers
        names.extend([f'K_{i}_{i}' for i in range(1, n_input+1)])
        #add input heterodimers
        names.extend([f'K_{i[0]}_{i[1]}' for i in itertools.combinations(range(1, n_input+1), 2)])
        #add input-acc heterodimers
        names.extend([f'K_{i[0]}_{i[1]}' for i in itertools.product(range(1, n_input+1),
                                                                  range(n_input+1, n_input+n_accesory+1))])
        #add accessory homodimers
        names.extend([f'K_{i}_{i}' for i in range(n_input+1, n_input+n_accesory+1)])
        #add accessory heterodimers
        names.extend([f'K_{i[0]}_{i[1]}' for i in itertools.combinations(range(n_input+1, n_input+n_accesory+1), 2)])

        return names

def swap_monomer_labels(m,num_inputs,param_sets,map,defaultA=1):
    '''
    Function to take a parameter set and swap monomer labels.

    Parameters
    ----------
    m : int
        Number of monomers
    num_inputs : int
        Number of input monomers
    param_sets : Array-like, shape (num_sets, num_combos_with_replacement(m,2)+(m-1))
        Parameter sets, with both affinities (K) and accessory expression levels (A)
    map : Dict
        Dictionary mapping monomer labels to new labels. Unchanged labels can be excluded.
        0-indexed
    defaultA : float
        Default accessory expression level to use for monomers that used to be inputs. Default is 1.
    
    Returns
    -------
    param_sets_swapped : Array-like, shape (num_sets, num_combos_with_replacement(m,2)+(m-1))
    '''
    # Add in unchanged monomers, 0-indexed
    monomer_map = {old:(map[old] if old in map.keys() else old) for old in range(m)}
    monomer_map_inv = {v:k for k,v in monomer_map.items()}

    # Get K names
    K_names = make_Kij_names(n_input = num_inputs,m=m) # Monomers are 1-indexed
    K_map = {}
    for K_i in range(num_combos_with_replacement(m,2)):
        new_monomers = sorted([monomer_map[int(K_names[K_i].split('_')[1])-1],monomer_map[int(K_names[K_i].split('_')[2])-1]])
        K_map[K_i] = K_names.index('K_{}_{}'.format(new_monomers[0]+1,new_monomers[1]+1))

    new_param_sets = np.zeros_like(param_sets)
    for param_set_i, param_set in enumerate(param_sets):
        new_K = [param_set[K_map[K_i]] for K_i in range(num_combos_with_replacement(m,2))]
        new_A = []
        for new_monomer_i in range(num_inputs,m):
            old_monomer_i = monomer_map_inv[new_monomer_i]
            if old_monomer_i>=num_inputs:
                new_A.append(param_set[num_combos_with_replacement(m,2)+(old_monomer_i-num_inputs)])
            elif new_monomer_i>=num_inputs:
                new_A.append(defaultA)
                print(f"Warning: New monomer {new_monomer_i+1} is an input monomer in the original network. Setting accessory expression level to default value of {defaultA}.")
        new_param_sets[param_set_i] = np.array(new_K + new_A)
    
    return new_param_sets

def swap_monomer_labels_output_dimer(m,dimer_id=None,dimer_name=None,map=None):
    '''
    Function to take a dimer ID or name and return the correspond dimer ID or name given a swap of monomer labels.

    The map is 0-indexed
    '''
    # Add in unchanged monomers
    map = {old:(map[old] if old in map.keys() else old) for old in range(m)}
    if dimer_id is None and dimer_name is None:
        raise ValueError('Must provide either dimer_id or dimer_name')
    return_dimer_id = False
    return_dimer_name = False
    if dimer_id is not None and dimer_name is None:
        dimer_name = make_nXn_species_names(m)[m+dimer_id]
        return_dimer_id = True
    elif dimer_id is None and dimer_name is not None:
        dimer_id = make_nXn_species_names(m).index(dimer_name)-m
        return_dimer_name = True
    dimer_monomers = [int(x)-1 for x in dimer_name.split('_')[1:]]
    dimer_monomers_new = np.sort([monomer_rename_map[x] if x in monomer_rename_map.keys() else x for x in dimer_monomers])
    dimer_name_new = f'D_{dimer_monomers_new[0]+1}_{dimer_monomers_new[1]+1}'
    dimer_id_new = make_nXn_species_names(m).index(dimer_name_new)-m
    if return_dimer_id:
        return dimer_id_new
    elif return_dimer_name:
        return dimer_name_new

def get_network_subset(m_old, num_inputs_old, num_inputs_new,param_sets,monomer_subset,defaultA=1):
    '''
    Function to return a parameter set for a subset of a network (i.e., a subset of monomers with the same affinities)

    Parameters
    ----------
    m : int
        Original number of monomers
    num_inputs_old : int
        Original number of input monomers
    num_inputs_new : int
        New number of input monomers
    param_sets : Array-like, shape (num_sets, num_combos_with_replacement(m,2)+(m-num_inputs_old))
        Parameter sets, with both affinities (K) and accessory expression levels (A)
    monomer_subset : List or array-like
        List of monomers to include in the subset. 0-indexed.
    
    Returns
    -------
    param_sets_subset : Array-like, shape (num_sets, num_combos_with_replacement(m,2)+(m-num_inputs_new))
        Parameter sets for the subset of monomers.
    '''
    # Create a mapping from the original monomer indices to the subset indices
    monomer_map_newtoold = {i: monomer for i, monomer in enumerate(monomer_subset)}
    m_new = len(monomer_subset)

    # Get the subset of Kij names for the selected monomers
    new_Kij_names = make_Kij_names(m=m_new, n_input=num_inputs_new)

    # Initialize the subset parameter sets
    param_sets_subset = np.zeros((param_sets.shape[0], len(new_Kij_names) + m_new - num_inputs_new))

    # Map the Kij parameters
    for i, Kij in enumerate(new_Kij_names):
        new_monomer1, new_monomer2 = map(int, Kij.split('_')[1:]) # 1-indexed
        old_monomer1, old_monomer2 = monomer_map_newtoold[new_monomer1-1], monomer_map_newtoold[new_monomer2-1] # 0-indexed
        # print(f'Old: K_{old_monomer1 + 1}_{old_monomer2 + 1}')
        # print(f'New: K_{new_monomer1}_{new_monomer2}')
        original_index = make_Kij_names(m=m_old, n_input=num_inputs_old).index(
            f'K_{old_monomer1 + 1}_{old_monomer2 + 1}'
        )
        param_sets_subset[:, i] = param_sets[:, original_index]

    # Map the accessory monomer parameters
    for i, monomer_old in enumerate(monomer_subset[num_inputs_new:]):
        if monomer_old >= num_inputs_old:
            original_index = num_combos_with_replacement(m_old, 2) + (monomer_old - num_inputs_old)
            param_sets_subset[:, len(new_Kij_names) + i] = param_sets[:, original_index]
            # print(f"Old: A_{monomer_old + 1}")
            # print(f"New: A_{i + num_inputs_new + 1}")
        else:
            param_sets_subset[:, len(new_Kij_names) + i] = defaultA
            print(f"Warning: Monomer {monomer_old+1} is an input monomer in the original network. Setting accessory expression level to default value of {defaultA}.")

    return param_sets_subset

def param_set_to_networkx(m, param_set,num_inputs=1,edge_min_K=1e-5):
    """
    Convert a parameter set to a networkx graph object.

    Parameters
    ----------
    m : int
        Number of monomer species.
    param_set : array_like, shape (num_combos_with_replacement(m,2)+(m-1))
        Parameter set, with both affinities (K) and accessory expression levels (A).
    
    Returns
    -------
    G : networkx.Graph
        Graph object representing the parameter set.
    """
    G = nx.Graph()
    G.add_nodes_from(np.arange(m))
    for i in range(m):
        for j in range(i + 1, m):
            dimer_index = make_Kij_names(m=m, n_input=num_inputs).index(f'K_{i + 1}_{j + 1}')
            if param_set_original[0, dimer_index] >= edge_min_K:
                G.add_edge(i, j)
    return G

def run_eqtk(N, C0, params, acc_monomer_ind):
    """
    Run eqtk.solve given the input stoichiometry matrix (N), initial concentrations (C0), and parameters (params)
    Parameters includes includes Kij values and accessory monomer levels. 
    """
    num_rxns = N.shape[0]
    K = params[:num_rxns]
    C0[:,acc_monomer_ind] = params[num_rxns:]

    return eqtk.solve(c0=C0, N=N, K=K)

def num_total_K_permutations(m, n_K_options):
    '''
    Function to calculate the total number of parameter permutations (including symmetrically redundant parameter sets) for
    a network of m monomers and nK possible K values.
    '''
    return n_K_options**(m+((m*(m-1))/2))

def num_total_K_A_permutations(m, n_K_options, n_A_options):
    '''
    Function to calculate the total number of parameter permutations (including symmetrically redundant parameter sets) for
    a network of m monomers, nK possible K values, and nA possible A values.
    '''
    return (n_K_options**(m+((m*(m-1))/2)))*(n_A_options**(m-1))

def num_combos(n,r):
    # Number of combinations of n items with r items per group
    return int(math.factorial(n)/(math.factorial(r)*math.factorial(n-r)))

def num_combos_with_replacement(n,r):
    # Number of combinations with replacement of n items with r items per group
    return int(math.factorial(n+r-1)/(math.factorial(r)*math.factorial(n-1)))

def num_permutations(n,r):
    # Number of permutations of n items with r items per group
    return int(math.factorial(n)/(math.factorial(n-r)))

def suggest_K_range_from_A_range(A_log_range,fraction_bound):
    # Suggests a K range at which, if fraction_bound = 0.99,
    # Weakest K: 1% is bound at the highest concentration
    # Strongest K: 99% is bound at the lowest concentration
    Kmax = (fraction_bound*(10**A_log_range[0]))/(((1-fraction_bound)*(10**A_log_range[0]))**2)
    Kmin = ((1-fraction_bound)*(10**A_log_range[1]))/((fraction_bound*(10**A_log_range[1]))**2)
    return [np.log10(Kmin),np.log10(Kmax)]


def dimers_from_param_set(param_set,edges_to_count,min_affinity=1e-5):
    return np.array([param_set[edge]>=min_affinity for edge in edges_to_count])

def count_dimers_per_param_set(param_set,edges_to_count,min_affinity=1e-5):
    return sum([param_set[edge]>=min_affinity for edge in edges_to_count])

def count_edges_per_monomer(param_set,edges_to_count_by_monomer,min_affinity=1e-5):
    return np.array([np.where(param_set[edges_to_count]>=min_affinity)[0].shape[0] for edges_to_count in edges_to_count_by_monomer])

def edges_per_monomer_to_sum_edges_per_dimer(edges_per_monomer,m,dimers):
    return np.array([edges_per_monomer[combo[0]]+edges_per_monomer[combo[1]]-2 if combo[0]!=combo[1]\
                     else edges_per_monomer[combo[0]]+edges_per_monomer[combo[1]] for i,combo in enumerate(dimers)])

def edges_per_monomer_to_edges_per_dimer(edges_per_monomer,m,dimers):
    return np.array([[edges_per_monomer[combo[0]]-1,edges_per_monomer[combo[1]]-1] if combo[0]!=combo[1]\
                     else [edges_per_monomer[combo[0]],edges_per_monomer[combo[1]]] for i,combo in enumerate(dimers)])

################################################
# Plotting: Regular Network Diagram
################################################
def calculate_object_size(input_,lower_input_bound,upper_input_bound,lower_size_bound,upper_size_bound):
    '''
    Takes some input, and uses some input bounds to rescale that input to some output bounds (such as width of a line in points).
    '''
    if input_<lower_input_bound:
        return 0
    elif input_>upper_input_bound:
        return upper_size_bound
    else:
        return lower_size_bound+((input_-lower_input_bound)/(upper_input_bound-lower_input_bound))*(upper_size_bound-lower_size_bound)

def get_poly_vertices(n, r = 1, dec = 3, start = math.pi/2):
    """
    Get x and y coordinates of n-polygon with radius r.
    """
    #This could be broadcast with numpy
    #i.e. x = r * np.cos(2*np.pi * np.arange(1,n+1)/n)
    #but I think it's easier to follow as list comprehension
    x = np.array([round(r * math.cos((2*math.pi*i/n)+start), dec) for i in range(n)])
    y = np.array([round(r * math.sin((2*math.pi*i/n)+start), dec) for i in range(n)])
    return x,y

def make_network_nodes_polygon(m, r, n_input,start_angle):
    '''
    Make a dataframe of nodes with columns for species name, type, and x and y coordinates
    '''
    x, y = get_poly_vertices(m, r=r,start=start_angle)
    species = [f'M_{i}' for i in range(1,m+1)]
    species_type = ['input']*n_input + ['accessory']*(m - n_input)
    node_df = pd.DataFrame({'species': species, 'species_type':species_type, 
                            'x': x, 'y': y})
    return node_df

def make_self_edges(m, r_node, r_edge,start_angle):
    '''
    Make a dataframe of self-edges with columns for x and y coordinates and \
    Kij name
    '''
    edge_df_list = [0] * m
    x, y = get_poly_vertices(m, r=r_node+r_edge,start=start_angle)
    # weights_scaled = np.array(K)*edge_scale
    for i in range(m):
        # Set center of self-edge to be r_edge further from the origin
        # angle =  math.atan2(y[i],x[i])
        # y_new = (r_node+r_edge)*np.sin(angle)
        # x_new = (r_node+r_edge)*np.cos(angle)
        x_new = x[i]
        y_new = y[i]
        center = [[x_new, y_new]]
        tmp_df = pd.DataFrame(center, columns=['x', 'y'])
        tmp_df['Kij_names'] = [f'K_{i+1}_{i+1}']
        edge_df_list[i] = tmp_df
        
    return pd.concat(edge_df_list)

def make_heterodimer_edges(m, node_df):
    '''
    Make a dataframe of heterodimer edges with columns for Kij name and x and y coordinates
    '''
    pairs = itertools.combinations(range(m), 2)
    n_heterodimers = number_of_heterodimers(m)
    x = [0]*n_heterodimers
    x_end = [0]*n_heterodimers
    y = [0]*n_heterodimers
    y_end = [0]*n_heterodimers
    names = [0]*n_heterodimers
    for i, comb in enumerate(pairs):
        x[i] = node_df.loc[comb[0],'x']
        x_end[i] = node_df.loc[comb[1],'x']
        y[i] = node_df.loc[comb[0],'y']
        y_end[i] = node_df.loc[comb[1],'y']
        names[i] = f'K_{comb[0]+1}_{comb[1]+1}'

    edge_df = pd.DataFrame({'Kij_names': names,
                          'x': x, 'x_end': x_end,
                          'y': y, 'y_end': y_end})
    return edge_df


def get_circle_intersections(x0, y0, r0, x1, y1, r1):
    '''
    Get the intersection points of two circles
    FROM: https://stackoverflow.com/questions/55816902/finding-the-intersection-of-two-circles
    circle 1: (x0, y0), radius r0
    circle 2: (x1, y1), radius r1
    '''
    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)
    
    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
        
        return [[x3, y3], [x4, y4]]

def make_network_plots_polygon(m, n_input, univs_to_plot, param_sets, dimers_of_interest=None,input_node_values = np.array([0]), ncols = 1, r_node = 1, r_loop = 0.5,rotate_offset=0,\
                            custom_node_positions=None,
                            node_scales = [-3,3,5,50], K_edge_scales = [-5,7,4,10],K_edge_alpha_scales=[-5,7,0.2,1],saveto='',input_cmap='Pastel1',\
                            fontname = 'Helvetica',fontsize=16,non_output_dimer_color='gray',output_dimer_color='k',labels=True,dpi=72,\
                            upscale_arrowhead = 1.2,node_edge_width=0.5,padding=0.1):
    """
    Load a subset of networks from a parameter file and plot the affinity parameters between monomers.
    
    Parameters
    --------------
    m: Int
        Number of total monomers in the network.
    n_input: Int
        Number of input monomers
    univs_to_plot: array-like
        Array of which universes (parameter sets) to plot.
    param_sets: Array-like, shape (n_univ, n_parameters)
        Array of parameters to plot. Alternatively, can use:
    dimers_of_interest: array-like or None
        If specified, will draw the arrows for dimers of interest in black and all others in gray
    input_node_values: Array-like of len n_input
        Parameter value(s), in log10-scale, to use for the input monomer expression level(s).
    ncols: Int
        Number of columns to use for multiple subplots.
    r_node: Float
        Radius of the nodes in the network.
    r_loop: Float
        Radius of the self-loop edges in the network.
    rotate_offset: Float
        Counterclockwise offset to rotate the network by in degrees.
    custom_node_positions: Array-like of shape (m,2)
        Custom node positions to use. If None, will arrange nodes around a circle.
    node_scales: List
        Scaling information for the sizes of the nodes. Defines marker size, the square of the marker 
        diameter in points. Note that for input, the value is just one number (size to use).
        Values: [lower value bound (log10), upper value bound (log10), lower size bound, upper size bound]
    K_edge_scales: List
        Scaling information for the widths of the edges in points.
        Values: [lower value bound (log10), upper value bound (log10), min edgewidth, max edgewidth]
    K_edge_alpha_scales: List
        Scaling information for the alpha of the edges.
        Values: [lower value bound (log10), upper value bound (log10), min alpha, max alpha]
    saveto: str
        Directory to save the figure to, without extension. If '', will not save.
        Saves both pdf and png
    fontname: str. Name of font to use.
    fontsize: Int. Size of the labels.
    non_output_dimer_color: str. Edge color used for non-output dimers.
    output_dimer_color: str. Edge color used for output dimers.
    labels: Bool. Whether to label the nodes. Default True.
    dpi: Int. Resolution of the figure, default 72. This will significantly affect the proportions of the figure.
    upscale_arrowhead: Float. Multiplied factor to determine the length and width of the arrowheads from the linewidth.
    node_edge_width: Float. Width of node edges in points.
    padding: Float. Extra space to add around each edge of the figure in inches.

    Returns:
        fig, axs: Created plot
    """
    param_sets = param_sets[univs_to_plot,:]
    num_plots = len(univs_to_plot)
    species_names = np.array(make_nXn_species_names(m))
    dimer_names = species_names[m:]
    Kij_labels = make_Kij_names(n_input = n_input, m=m)
    num_rxns = len(Kij_labels)
    input_node_names = ["M_{}".format(i+1) for i in range(n_input)]

    # Make dataframe containing node positions. Color accessory monomer nodes and scale size by parameter value
    
    node_df_list = [0]*num_plots
    for i in range(num_plots):
        #scale acc monomer weights
        # acc_weights = np.log10(param_sets[i,num_rxns:])
        node_df_list[i] = make_network_nodes_polygon(m=m, r=r_node, n_input=n_input,\
                                                     start_angle=(math.radians(rotate_offset)+(math.pi/2)-((n_input-1)*(2*math.pi/(2*m)))))
        if custom_node_positions is not None:
            node_df_list[i][['x','y']] = custom_node_positions
    node_df_combined = pd.concat(node_df_list, keys=univs_to_plot).reset_index()
    node_df_combined.rename(columns={'level_0': 'univ'}, inplace=True)

    node_weights = param_sets[np.arange(num_plots), num_rxns:]
    node_weights = np.hstack((np.tile(10**(input_node_values.astype(np.float64)),reps=(num_plots,1)), node_weights))
    node_weights = np.log10(node_weights)

    node_df_combined['weight'] = node_weights.flatten()

    #Make dataframe for self-loops. Scale width by Kii value
    self_edge_df = make_self_edges(m, r_node, r_loop,\
                                                     start_angle=(math.radians(rotate_offset)+(math.pi/2)-((n_input-1)*(2*math.pi/(2*m)))))
    self_edge_df_combined = pd.concat([self_edge_df]*num_plots, keys=np.arange(num_plots)).reset_index() # Combine dfs
    self_edge_df_combined.rename(columns={'level_0': 'univ'}, inplace=True)
    self_edge_labels = [f'K_{i}_{i}' for i in range(1,m+1)] # Add labels
    self_edge_index = np.where(np.isin(Kij_labels, self_edge_labels))[0]

    self_edge_weights = param_sets[np.arange(num_plots)[:,np.newaxis], self_edge_index] # Get weights
    self_edge_weights = np.log10(self_edge_weights)

    self_edge_df_combined['weight'] = np.repeat(self_edge_weights.flatten(), self_edge_df_combined.level_1.max()+1)
    
    #Make dataframe for heterodimer edges. Scale width by Kij value
    hetero_edge_df = make_heterodimer_edges(m, node_df_combined)
    hetero_edge_df_combined = pd.concat([hetero_edge_df]*num_plots, keys=np.arange(num_plots)).reset_index()
    hetero_edge_df_combined.rename(columns={'level_0': 'univ'}, inplace=True)
    hetero_edge_index = np.where(~np.isin(Kij_labels, self_edge_labels))[0]
    hetero_edge_weights = param_sets[np.arange(num_plots)[:,np.newaxis], hetero_edge_index]
    hetero_edge_weights = np.log10(hetero_edge_weights)
    hetero_edge_df_combined['weight'] = hetero_edge_weights.flatten()
    
    if type(input_cmap)==str:
        cmap = plt.get_cmap(input_cmap)
    else:
        cmap = input_cmap.copy()

    if len(univs_to_plot)==1:
        ncols = 1
    
    nrows = math.ceil(len(univs_to_plot)//ncols)
    figsize = (2*(r_node+(2*r_loop)+padding)*ncols,2*(r_node+(2*r_loop)+padding)*nrows)
    # Create subplots but with no padding
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,figsize=figsize,dpi=dpi,squeeze=False,\
                        gridspec_kw={'hspace':0,'wspace':0,'left':0,'right':1,'top':1,'bottom':0})

    for univ in range(len(univs_to_plot)):
        row = univ//ncols
        col = univ%ncols
        ax = axs[row,col]
        ax.axis('off') # Hide axes
        ax.set_xlim(-(r_node+(2*r_loop)+padding),(r_node+(2*r_loop)+padding))
        ax.set_ylim(-(r_node+(2*r_loop)+padding),(r_node+(2*r_loop)+padding))
        # Calculate conversion between point size and data coordinates
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        xrange = ax.get_xlim()[1] - ax.get_xlim()[0]
        yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
        x_points_per_coord = (bbox.width*fig.get_dpi())/xrange
        y_points_per_coord = (bbox.height*fig.get_dpi())/yrange
        points_per_coord = np.mean([x_points_per_coord,y_points_per_coord])
        # FOR HOMODIMERS
        for i, edge in self_edge_df_combined.iterrows():
            if dimers_of_interest is not None:
                if Kij_labels.index(edge['Kij_names']) in dimers_of_interest[univ]:
                    arrow_color = output_dimer_color
                    zorder = 3
                    arrow_alpha = 1
                else:
                    arrow_color = non_output_dimer_color
                    zorder = 1
                    arrow_alpha = calculate_object_size(edge['weight'],K_edge_alpha_scales[0],K_edge_alpha_scales[1],K_edge_alpha_scales[2],K_edge_alpha_scales[3])
            else:
                arrow_color = 'k'
                zorder = 1
                arrow_alpha = calculate_object_size(edge['weight'],K_edge_alpha_scales[0],K_edge_alpha_scales[1],K_edge_alpha_scales[2],K_edge_alpha_scales[3])
            node_name = "M_{}".format(edge["Kij_names"].split('_')[1])
            # Coordinate of node center
            node_coord = np.array(node_df_combined.query('species == @node_name').reset_index(drop=True).loc[0,['x','y']])
            # Node radius in coordinate units
            node_radius = (calculate_object_size(node_df_combined.query('species == @node_name').reset_index(drop=True).loc[0,'weight'],\
               node_scales[0],node_scales[1],node_scales[2],node_scales[3])/2)/points_per_coord
            # Coordinate of center of the "self_edge" 
            self_edge_coord = np.array([edge['x'],edge['y']])
            # Write node name
            if labels:
                if node_name in input_node_names:
                    ax.text(edge['x'],edge['y'],node_name.replace('_',''),c='k',ha='center',va='center',fontsize=fontsize,fontweight='bold',fontname=fontname)
                else:
                    ax.text(edge['x'],edge['y'],node_name.replace('_',''),c=non_output_dimer_color,ha='center',va='center',fontsize=fontsize,fontweight='bold',fontname=fontname)
            # Draw arrow
            # Arrow size in pts (line width)
            weight = calculate_object_size(edge['weight'],K_edge_scales[0],K_edge_scales[1],K_edge_scales[2],K_edge_scales[3])
            if weight==0:
                continue
            # Arrow size in coordinate units
            arrow_size = weight*2/points_per_coord
            # intersection of the node circle and self-arrow
            circle_intersections = get_circle_intersections(node_coord[0],node_coord[1],node_radius,self_edge_coord[0],self_edge_coord[1],r_loop)
            if circle_intersections is not None:
                # Arc angle filled by arrowhead
                arrow_arc_angle = arrow_size/r_loop
                # Vector from the center of the "self-edge" circle to the node center
                self_edge_center_to_node = (node_coord-self_edge_coord)/np.linalg.norm(node_coord-self_edge_coord)
                # Angle from the center of the "self-edge" circle to the node center
                node_angle_from_self_edge_center = math.atan2(self_edge_center_to_node[1],self_edge_center_to_node[0])
                # Calculate angles formed between each intersection point and the self-edge center
                norm_relative_intersect_points = [(coord-self_edge_coord)/np.linalg.norm((coord-self_edge_coord),ord=2) for coord in circle_intersections]
                intersect_angles = [math.atan2(coord[1],coord[0]) for coord in norm_relative_intersect_points]
                # Convert to between 0 and 2 pi
                intersect_angles = np.mod(intersect_angles, 2 * np.pi)
                # Find the smallest intersect_angles-node_angle_from_self_edge_center that is still positive
                ## Calculate the difference between intersect_angles and node_angle_from_self_edge_center
                angle_differences = intersect_angles - node_angle_from_self_edge_center
                angle_differences[angle_differences<0] += 2*np.pi # Add 2pi to negative angles

                # For the counterclockwise arrow
                intersect_to_use_id = np.argmax(angle_differences)
                intersect_to_use = circle_intersections[intersect_to_use_id]
                # Calculate angle from center of "self-edge" circle to center of arrowhead
                arrow_center_angle = intersect_angles[intersect_to_use_id]-arrow_arc_angle
                # Get coordinate of center of arrowhead
                arrow_coord = self_edge_coord+r_loop*np.array([np.cos(arrow_center_angle),np.sin(arrow_center_angle)])
                # Get vector pointing in direction of arrow
                arrow_vector = (intersect_to_use-arrow_coord)/np.linalg.norm(intersect_to_use-arrow_coord,ord=2)
                # Get coordinate of arrow "base"
                arrow_base = arrow_coord-(arrow_vector*(arrow_size/2))
                # Make arrow
                arrow = plt.arrow(arrow_base[0], arrow_base[1], (intersect_to_use-arrow_base)[0], (intersect_to_use-arrow_base)[1], \
                      length_includes_head=True, head_width=arrow_size*upscale_arrowhead, head_length=arrow_size*upscale_arrowhead, \
                      overhang=0,color=arrow_color,linewidth=0,alpha=arrow_alpha,\
                      zorder=zorder)

                # For the clockwise arrow
                intersect_to_use_id = np.argmin(angle_differences)
                intersect_to_use = circle_intersections[intersect_to_use_id]
                # Calculate angle from center of "self-edge" circle to center of arrowhead
                arrow_center_angle = intersect_angles[intersect_to_use_id]+arrow_arc_angle
                # Get coordinate of center of arrowhead
                arrow_coord = self_edge_coord+r_loop*np.array([np.cos(arrow_center_angle),np.sin(arrow_center_angle)])
                # Get vector pointing in direction of arrow
                arrow_vector = (intersect_to_use-arrow_coord)/np.linalg.norm(intersect_to_use-arrow_coord,ord=2)
                # Get coordinate of arrow "base"
                arrow_base = arrow_coord-(arrow_vector*(arrow_size/2))
                # Make arrow
                plt.arrow(arrow_base[0], arrow_base[1], (intersect_to_use-arrow_base)[0], (intersect_to_use-arrow_base)[1], \
                      length_includes_head=True, head_width=arrow_size*upscale_arrowhead, head_length=arrow_size*upscale_arrowhead, overhang=0,\
                      color=arrow_color,linewidth=0,alpha=arrow_alpha,zorder=zorder,width=0)
                
                # Add arc for arrow
                # Need to choose theta1 and theta2 to make the arc draw on the outside
                theta1 = np.degrees(intersect_angles[np.argmin(angle_differences)]+arrow_arc_angle)
                theta2 = np.degrees(intersect_angles[np.argmax(angle_differences)]-arrow_arc_angle)
                arc = mpatches.Arc(xy=np.array([edge['x'],edge['y']]), width=2*r_loop, height=2*r_loop,
                         angle=0, linewidth=weight, fill=False, zorder=zorder,color=arrow_color,\
                        theta1=theta1,\
                        theta2=theta2,\
                        alpha=arrow_alpha)
                ax.add_patch(arc)

        # FOR HETERODIMERS
        for i, edge in hetero_edge_df_combined.iterrows():
            if dimers_of_interest is not None:
                if Kij_labels.index(edge['Kij_names']) in dimers_of_interest[univ]:
                    arrow_color = output_dimer_color
                    zorder = 3
                    arrow_alpha = 1
                else:
                    arrow_color = non_output_dimer_color
                    zorder = 1
                    arrow_alpha = calculate_object_size(edge['weight'],K_edge_alpha_scales[0],K_edge_alpha_scales[1],K_edge_alpha_scales[2],K_edge_alpha_scales[3])
            else:
                arrow_color = 'k'
                zorder = 1
                arrow_alpha = calculate_object_size(edge['weight'],K_edge_alpha_scales[0],K_edge_alpha_scales[1],K_edge_alpha_scales[2],K_edge_alpha_scales[3])
            arrow_size = calculate_object_size(edge['weight'],K_edge_scales[0],K_edge_scales[1],K_edge_scales[2],K_edge_scales[3])*2/points_per_coord
            if arrow_size==0:
                continue
            start_node_coord = np.array([edge['x'],edge['y']])
            end_node_coord = np.array([edge['x_end'],edge['y_end']])
            if np.all(start_node_coord==end_node_coord):
                continue
            start_node_name = "M_{}".format(edge["Kij_names"].split('_')[1])
            end_node_name = "M_{}".format(edge["Kij_names"].split('_')[2])
            start_node_size = calculate_object_size(node_df_combined.query('species == @start_node_name')['weight']\
                                                  .reset_index(drop=True)[0],\
                                                  node_scales[0],node_scales[1],node_scales[2],node_scales[3])
            start_node_radius_dataunits = (start_node_size/2)/points_per_coord
            end_node_size = calculate_object_size(node_df_combined.query('species == @end_node_name')['weight']\
                                                  .reset_index(drop=True)[0],\
                                                  node_scales[0],node_scales[1],node_scales[2],node_scales[3])
            end_node_radius_dataunits = (end_node_size/2)/points_per_coord
            start_end_vector = end_node_coord-start_node_coord
            start_end_vector = start_end_vector/np.linalg.norm(start_end_vector,ord=2)
            start_arrow_tip = start_node_coord+start_end_vector*start_node_radius_dataunits
            start_arrow_base = start_node_coord+start_end_vector*(start_node_radius_dataunits+arrow_size)
            end_arrow_tip = end_node_coord-(start_end_vector*end_node_radius_dataunits)
            end_arrow_base = end_node_coord-(start_end_vector*(end_node_radius_dataunits+arrow_size))
            # Make arrows 
            plt.arrow(start_arrow_base[0], start_arrow_base[1], (start_arrow_tip-start_arrow_base)[0], (start_arrow_tip-start_arrow_base)[1], \
                      length_includes_head=True, head_width=arrow_size*upscale_arrowhead, head_length=arrow_size*upscale_arrowhead, overhang=0,\
                      color=arrow_color,linewidth=0,alpha=arrow_alpha,zorder=zorder,width=0)
            plt.arrow(end_arrow_base[0], end_arrow_base[1], (end_arrow_tip-end_arrow_base)[0], (end_arrow_tip-end_arrow_base)[1], \
                      length_includes_head=True, head_width=arrow_size*upscale_arrowhead, head_length=arrow_size*upscale_arrowhead, overhang=0,\
                      color=arrow_color,linewidth=0,alpha=arrow_alpha,zorder=zorder,width=0)
            ax.plot([start_arrow_base[0], end_arrow_base[0]],[start_arrow_base[1], end_arrow_base[1]], color = arrow_color,\
                   lw=calculate_object_size(edge['weight'],K_edge_scales[0],K_edge_scales[1],K_edge_scales[2],K_edge_scales[3]),\
                   alpha=arrow_alpha,marker=None,zorder=zorder+1)

        # Make nodes
        # Note that the edges are normally applied half-in, half-out. We reduce the diameter by one edge width because we lose half an edge width on both sides.
        ax.scatter(node_df_combined['x'],node_df_combined['y'],\
           s=[(calculate_object_size(x,\
           node_scales[0],node_scales[1],node_scales[2],node_scales[3])-node_edge_width)**2 for x in node_df_combined['weight']],\
           color=[cmap(i) for i in range(m)]*num_plots,linewidths=node_edge_width,edgecolors='k',zorder=4)
        
    if saveto!='':
        plt.savefig(saveto+'.pdf',pad_inches=0,bbox_inches='tight',transparent=True)
        fig.patch.set_facecolor('white')
        plt.savefig(saveto+'.png',dpi=900,bbox_inches='tight')
    
    return fig, axs




