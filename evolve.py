import numpy as np
import h5py
import os
from scipy import sparse
from copy import deepcopy
from datetime import datetime  # For evaluating the compute time.
from tqdm import tqdm  # progress bar


def save(half_edges, heads_idx, synapses, filename, t, extra_info=None, add_to_prev=False, savemode='a'):
    """Write data to the output file.

    The data of every time step is stored in a new group.
    Group labels are the numbers of the steps, e.g. the initial condition is saved in group '0' and so on.
    For the initial condition the full vectors of half-edges and synapses are saved.
    For all subsequent time steps only the changes with respect to the previous step are saved as sparse arrays.
    Additionally, the current time and optionally additional time dependent quantities are saved in a dataset with key
    'time' and additional datasets within the group of the current time step, respectively.

    half_edges: Sparse array in COOrdinate format giving the initial array of in-going half-edges or changes to it with
           respect to the previous step.
    heads_idx: Sparse array in COOrdinate format giving the initial array of the half-edges' indices or changes to it with
               respect to the previous step.
    synapses: Sparse array in COOrdinate format giving the initial binary array indicating the (non-)existence of
              a synapse between the respective tail and head or changes to it with respect to the previous step.
    filename: String giving the path to and name of the output file.
    t: Float giving the current time.
    extra_info: Dictionary of additional time-dependent quantities to write to the output. For every key a dataset with
                that name will be created in the group of the respective time step which stores the corresponding value.
                Optional.
                For other additional quantities which are not time-dependent and therefore should not be associated with
                a certain time step, use the function save_extra() below!
    add_to_prev: Bool indicating whether the changes should be added to the previous step. Needed if the time did not
                 change with respect to the previous step. This is the case i.e. if the initial condition had been reset
                 due to an instantaneous perturbation.
    savemode: String, mode for opening the file. Should only be changed to something other than 'a' if wanted the first
              time the save() function is called. Else the file might get overwritten every time this function is called
              for saving data throughout the simulation!
    """

    with h5py.File(filename + '.hdf5', savemode) as file:
        # Create dataset for time if it does not exist yet.
        if 'time' not in file.keys():
            file.create_dataset('time', data=np.array([t]), maxshape=(None,), dtype='f')
        # Save the current time.
        else:
            if not add_to_prev:
                file['time'].resize(file['time'].shape[0] + 1, axis=0)
            file['time'][-1] = t

        # Check how many groups are already in the file.
        n_group = len(file.keys()) - 1  # One dataset is the time.
        # Subtract the dataset of additional quantities that are not belonging to a time step if existing.
        if 'extra' in file.keys():
            n_group -= 1

        if add_to_prev:
            # Go to savings of last step.
            n_group -= 1
            # Load previous savings.
            data_prev = file[str(n_group)]
            half_edges_prev = sparse.coo_matrix((data_prev['half_edges']['data'][:],
                                            (np.zeros_like(data_prev['half_edges']['col'][:]), data_prev['half_edges']['col'][:])),
                                           shape=data_prev['half_edges'].attrs['shape'])
            heads_idx_prev = sparse.coo_matrix((data_prev['heads_idx']['data'][:],
                                                (np.zeros_like(data_prev['heads_idx']['col'][:]), data_prev['heads_idx']['col'][:])),
                                               shape=data_prev['heads_idx'].attrs['shape'])
            synapses_prev = sparse.coo_matrix((data_prev['synapses']['data'][:],
                                               (data_prev['synapses']['col'][:], data_prev['synapses']['row'][:])),
                                              shape=data_prev['synapses'].attrs['shape'])

            # Add current changes.
            half_edges = half_edges_prev.tocsc() + half_edges.tocsc()
            heads_idx = heads_idx_prev.tocsc() + heads_idx.tocsc()
            synapses = synapses_prev.tocsc() + synapses.tocsc()

            # Convert back to coo format.
            half_edges = half_edges.tocoo()
            heads_idx = heads_idx.tocoo()
            synapses = synapses.tocoo()

            # Delete the last group such that the previous savings will be overwritten with the updated changes.
            del file[str(n_group)]

        # Create the next group labeled by the step no.
        group = file.create_group(str(n_group))

        
        # Save the (changes to) the half edges vector.
        half_edges_subgrp = group.create_group('half_edges')
        half_edges_subgrp.create_dataset('data', data=half_edges.data)
        half_edges_subgrp.create_dataset('col', data=half_edges.col)
        half_edges_subgrp.attrs['shape'] = half_edges.shape
        # Save the (changes) to the heads' indices vector.
        heads_idx_subgrp = group.create_group('heads_idx')
        heads_idx_subgrp.create_dataset('data', data=heads_idx.data)
        heads_idx_subgrp.create_dataset('col', data=heads_idx.col)
        heads_idx_subgrp.attrs['shape'] = heads_idx.shape
        # Save the (changes to) the synapses vector.
        synapses_subgrp = group.create_group('synapses')
        synapses_subgrp.create_dataset('data', data=synapses.data)
        synapses_subgrp.create_dataset('col', data=synapses.col)
        synapses_subgrp.create_dataset('row', data=synapses.row)
        synapses_subgrp.attrs['shape'] = synapses.shape

        # Save additional time-dependent quantities if given any.
        if extra_info is not None:
            for key, val in extra_info.items():
                group.create_dataset(key, shape=val.shape, data=val)


def save_extra(quantity, key, filename):
    """Write additional data to the output file.

    quantity: Array-like, quantity to be saved.
    key: String giving the name for the extra dataset.
    filename: String giving the path to and name of the output file.

    All extra quantities will be stored as datasets in the group 'extra'.
    If a dataset with the given key already exists, data will be appended to that dataset.
    """
    with h5py.File(filename + '.hdf5', 'a') as file:
        if 'extra' not in file.keys():
            extra_grp = file.create_group('extra')
        else:
            extra_grp = file['extra']

        if key not in extra_grp.keys():
            if len(quantity.shape) > 1:
                max_shape = (None,) + quantity.shape[1:]
            else:
                max_shape = (None,)
            extra_grp.create_dataset(key, data=quantity, maxshape=max_shape)
        else:
            extra_grp[key].resize(extra_grp[key].shape[0] + quantity.shape[0], axis=0)
            extra_grp[key][-quantity.shape[0]:] = quantity[:]


def rate_to_prob(rate, stepwidth):
    """Convert a rate of an event to the probability that this event takes place within a certain time period.

    rate: Float giving the rate of that event per unit of time.
    stepwidth: Float giving the time interval over which the probability of the event to take place is to be computed.
    """
    return 1 - np.exp(-rate * stepwidth)


def monte_carlo_step(network, steps_survived, alpha, beta_noise, stepwidth, t, filename):
    """Make a random step in the Markov chain.

    network: Instance of the Network class.
    steps_survived: Array indicating for how many steps existing synapses have already been persistent or -1 if the
                    corresponding half-edges currently do not form a synapse (unbound).
                    Same order as network.synapses.
                    Needed for evaluating the synaptic lifetime distribution.
    alpha: Rate of random edge creation.
    beta_noise: Uniform noise level of random edge deletion.
    stepwidth: Float giving the time difference between two steps in the Markov chain.
    t: Float giving the current time.
    filename: String with the path to and name of the output file.
    """
    # Update the number of steps that existing synapses have survived.
    #steps_survived[steps_survived >= 0] += 1

    # Initialize arrays to store the new micro-configuration.
    half_edges_new = deepcopy(network.half_edges)
    heads_idx_new = deepcopy(network.heads_idx)
    synapses_new = deepcopy(network.synapses)

    
    # --- RANDOM EDGE DELETION (NOISE) ---
    # All edges are randomly broken with a probability corresponding to the rate beta_noise no matter if there is an
    # explicit perturbation or not.
    # Entries in the network's synapse array corresponding to broken edges are switched to '-1' like above.
    if beta_noise != 0:
        synapses_new += np.where(synapses_new == 1,
                                 -np.random.binomial(1, rate_to_prob(beta_noise, stepwidth), size=network.z) * 2,
                                 0)

    # Save the lifetimes of synapses that have been eliminated.
    #lifetimes = steps_survived[synapses_new == -1] * stepwidth
    # Set their corresponding entries in steps_survived back to -1.
    #steps_survived[synapses_new == -1] = -1

    # --- RANDOM WIRING ---
    # Heads (tails) that are unconnected are randomly connected to other free tails (heads) at a rate alpha.
    # Those are all half-edges belonging to entries of '0' in the network's synapse array.
    if alpha != 0:
        #where can new synapses emerge?
        zero_indices = np.setdiff1d(heads_idx_new, np.where(synapses_new == 1)[0]).astype(int)
        pot = np.random.permutation(zero_indices).reshape(int(len(zero_indices)/2),2)

        for coordy in pot:
            new = np.random.binomial(1, rate_to_prob(alpha, stepwidth))*2
            synapses_new[tuple(coordy)] = new
            synapses_new[tuple(np.flip(coordy,axis=0))] = new
        
        # Set counter of steps survived to 0 for new synapses.
        #steps_survived[synapses_new == 2] = 0
        # Switch labels of new synapses from '2' to '1'.
        synapses_new = np.where(synapses_new == 2, 1, synapses_new)

    # Switch labels of synapses broken due to perturbation above from '-1' to '0' for that the freed half-edges are
    # being considered for random wiring in the next iteration.
    if beta_noise != 0:
        synapses_new = np.where(synapses_new == -1, 0, synapses_new)

    # Compute the changes to the micro-configuration of the previous step.
    changes_half_edges = sparse.coo_matrix(half_edges_new - network.half_edges)
    changes_heads_idx = sparse.coo_matrix(heads_idx_new - network.heads_idx)
    changes_synapses = sparse.coo_matrix(synapses_new - network.synapses)

    # Save the new micro-configuration in the output file.
    save(changes_half_edges, changes_heads_idx, changes_synapses, filename, t)
    # Write lifetimes to the output file if any synapses died.
    '''if len(lifetimes) > 0:
        save_extra(lifetimes, "lifetimes", filename)'''

    # Update the network's attributes according to the new micro-configuration.
    network.half_edges = half_edges_new
    network.heads_idx = heads_idx_new
    network.synapses = synapses_new

    return steps_survived



def evolve(network, t_range, h, alpha, beta_noise, filename, savemode='w-',
           edge_perts=None, edge_pert_times=np.array([None]),
           inst_perts=None, inst_pert_times=np.array([None]),
           verbose=True):
    """Time evolve a network object using the MCMC (Markov chain Monte Carlo) method.

    network: Instance of the Network class in 'network.py'.
    t_range: Array indicating the start and end time of the simulation: [start, end].
    h: Float indicating the stepwidth, the time interval dt between two steps in the Markov chain.
    alpha: Rate of random edge creation.
    beta_noise: Uniform noise level of random edge deletion.
    filename: String indicating the path to and name of the output file.
    savemode: String indicating the mode for opening the output file.
              Use the default 'w-' if the file does not exist yet. If it exists already this fails, use e.g. 'a' then.
              Use 'w' if the file already exists but it should be overwritten.
              Valid modes are: 'r', 'r+', 'w', 'w-' or 'x', 'a'.
    edge_perts: List of dictionaries for every perturbation-/no-perturbation-phase lasting for a specific
                   time interval. Optional.
                   For every phase: {'epsilon': Array of Int indicating how many edges are broken at the corresponding
                                                location, specified in 'pattern',
                                     'pattern': Array of two node indices for specifying where to break edges,
                                                empty array if no perturbation (beta=0).}
    edge_pert_times: Array of time points for switching to the next edge perturbation.
                        Make sure that the time interval for each phase is a multiple of the stepwidth h!
    inst_perts: List of dictionaries for every instantaneous, absolute change (perturbation) of the network's
                configuration. Optional.
                For every change: {'epsilon': Integer indicating the number of edges to be deleted. Has to be even!,
                                   'pattern': Array of node indices that are part of the pattern for perturbation.}
    inst_pert_times: Array of time points for applying the instantaneous changes of the configuration.
                     Should have the same length as inst_perts.
                     Make sure that the intervals between two changes are multiples of the stepwidth h!
    verbose: Bool, turn on (True) or off (False) the verbosity.
    """
    start = datetime.now()

    #no error
    lasting_pert_times = edge_pert_times
    lasting_perts = edge_perts

    # Create/open output file and write initial configuration to it.
    if verbose:
        print("Write simulation data to %s.hdf5" % filename)
    save(sparse.coo_matrix(network.half_edges), sparse.coo_matrix(network.heads_idx), sparse.coo_matrix(network.synapses),
         filename, t_range[0], savemode=savemode)

    # Check if dictionaries passed for instantaneous edge perturbations and corresponding time points are compatible.
    if np.all(edge_pert_times == None) and edge_perts is None:
        edge_pert_times = np.array([])
        edge_perts = [{'epsilon': np.array([0]), 'pattern': np.empty(0)}]
    elif edge_perts is not None:
        if np.all(edge_pert_times == None):
            raise Exception("No specified times for edge perturbation!")
        else:
            if len(edge_pert_times) != len(edge_perts):
                raise Exception("The amount of edge perturbation time points do not match the amount of patterns for edge perturbation!")
            
    if not np.all(edge_pert_times == None) and edge_perts is None:
        raise Exception("Time points for applying edge perturbations were passed but no perturbation patterns.")

    # Check if dictionaries and time points passed for instantaneous node perturbations are compatible.
    if inst_perts is None and not np.all(inst_pert_times == None):
        raise Exception("Time points for instantaneous perturbations were passed but no perturbation patterns.")
    elif inst_perts is not None:
        if np.all(inst_pert_times == None):
            raise Exception("Patterns for instantaneous perturbations were passed but no time points when to apply them.")
        elif len(inst_perts) != len(inst_pert_times):
            raise Exception("The numbers of patterns for instantaneous perturbations and corresponding time points "
                            + "should be the same.")

    # Create a list of times at which parameters need to be changed.
    
    times = np.concatenate((edge_pert_times, inst_pert_times))
    times = np.append(times, t_range[0]) # Include start point of the simulation.
    times = times[times != np.array(None)]  # Remove None that may come from the times for instantaneous perturbations.
    times = np.unique(times)  # Remove duplicates.
    times = np.sort(times)
    times = np.append(times, t_range[1])  # Include end point of the simulation.
    
    # Counters for perturbations.
    edge_pert_idx = -1
    inst_pert_idx = -1

    # Array for keeping track of synaptic lifetimes.
    # If >= 0 this is the number of iterations a synapse at the same position in network.synapses has already existed.
    # If -1 the respective synapse does not exist (the half-edges are unbound).
    steps_survived = np.full_like(network.synapses, -1)
    # Set counter to 0 for initially existing synapses.
    steps_survived[network.synapses == 1] = 0

    #for i in tqdm(range(len(times)-1)):
    for i in range(len(times)-1):
        if verbose:
            print("New time step from t_init=", times[i], "to t_final =", times[i+1], ".")

        # Check if the time interval is a multiple of h.
        # Due to precision errors, the remainder can be either close to zero or close to the value of the stepwidth.
        if not np.isclose(np.mod(times[i+1] - times[i], h), 0) and not np.isclose(np.mod(times[i+1] - times[i], h), h):
            raise Exception("The time interval is not a multiple of the stepwidth. Cannot divide the interval into "
                            "equally spaced steps with a stepwidth of h.")
        # Breaking edges if the current time is in edge_pert_times.
        if times[i] in edge_pert_times:
            
            changes_synapses = deepcopy(network.synapses)
            
            edge_pert_idx += 1
            edge_pert_i = edge_perts[edge_pert_idx]
            if verbose:
                print(r"Instantaneous perturbation of the configuration! Deleting $\epsilon$ =", edge_pert_i['epsilon'],
                      "random edges coming between the following nodes: ", edge_pert_i['pattern'], ".")

            delete_half_edges = np.array([[0,0]])
            for j in range(edge_pert_i['epsilon'].size):
                patt = edge_pert_i['pattern'][j]
                #identifying which synapses can be broken
                pot_micro_connections_idx = np.argwhere(np.all(np.dstack((network.half_edges[sparse.coo_matrix(network.synapses).col],
                                                                          network.half_edges[sparse.coo_matrix(network.synapses).row]))[0]==patt,axis=1)).flatten()
                if pot_micro_connections_idx.size==0:
                    print("FAILING: Not enough synapses can be broken for synapses between ", edge_pert_i['pattern'], ".")
                    break
                #choosing which synapses to break
                ind = np.random.choice(pot_micro_connections_idx,  # Only existing synapses can be deleted.
                                            int(edge_pert_i['epsilon'][j]), replace=False)
                pot_micro_connections = np.dstack([sparse.coo_matrix(network.synapses).col[ind],
                                                  sparse.coo_matrix(network.synapses).row[ind]])[0]
                #saving the synapses to break
                delete_half_edges = np.append(delete_half_edges,pot_micro_connections,axis=0)
                delete_half_edges = np.append(delete_half_edges,np.flip(pot_micro_connections,axis=1),axis=0)
            delete_half_edges = np.delete(delete_half_edges, 0, axis = 0)

            # Get the changes to the synapses vector.
            network.synapses[delete_half_edges] = 0
            network.compute_z_hat()
            #lifetimes = steps_survived[delete_half_edges] * h
            
            changes_synapses = network.synapses - changes_synapses        

            # Save changed configuration. Overwrite last time step in  the output file.
            save(sparse.coo_matrix(np.zeros(network.half_edges.shape)), sparse.coo_matrix(np.zeros(network.heads_idx.shape)),
                 sparse.coo_matrix(changes_synapses), filename, times[i], add_to_prev=True)
            # Write lifetimes of deleted synapses to the output file.
            #save_extra(lifetimes, "lifetimes", filename)

        # Apply an instantaneous, absolute perturbation if the current time is in inst_pert_times.
        if times[i] in inst_pert_times:
            inst_pert_idx += 1
            inst_pert_i = inst_perts[inst_pert_idx]

            changes_synapses = deepcopy(network.synapses)

            if verbose:
                print(r"Instantaneous perturbation of the configuration! Deleting $\epsilon$ =", inst_pert_i['epsilon'],
                      "random edges coming from and/or going to a node in the pattern", inst_pert_i['pattern'], ".")

            
            # Delete epsilon random synapses out of all whose synapse belongs to a pattern-node.

            #zufällig aussuchen, welche Synapsen zerstört werden, die eine Koordinate in 'pattern' haben.
            pattern_half = np.array([])
            for patt in inst_pert_i['pattern']:
                pattern_half = np.append(pattern_half,np.where(network.half_edges == patt))
            coords = np.dstack((sparse.coo_matrix(np.triu(network.synapses)).row,sparse.coo_matrix(np.triu(network.synapses)).col))
            #randomly choosing the synapses to break if there are enough
            possibilities = np.where(np.any(np.dstack(np.isin(coords, pattern_half)[0]),axis=1)[0])[0]
            if possibilities.shape[0]<int(inst_pert_i['epsilon']):
                print("FAILING: Not enough synapses can be broken for synapses between ", inst_pert_i['pattern'], ".")
            else: 
                ind = np.random.choice(possibilities,  # Only existing synapses can be deleted.
                                                int(inst_pert_i['epsilon']), replace=False)
                #saving the synapses to break
                delete_half_edges = coords[0][ind]
                delete_half_edges = np.append(delete_half_edges,np.flip(coords[0][ind],axis=1),axis=0)

                # Get the changes to the synapses vector.
                network.synapses[delete_half_edges] = 0
                network.compute_z_hat()
            
                # Save lifetimes of deleted synapses.
                #lifetimes = steps_survived[delete_half_edges] * h
                # Switch the number of steps survived back to -1 for deleted synapses.
                #steps_survived[delete_half_edges] = -1

                changes_synapses = network.synapses - changes_synapses
                
                # Save changed configuration. Overwrite last time step in  the output file.
                save(sparse.coo_matrix(np.zeros(network.half_edges.shape)), sparse.coo_matrix(np.zeros(network.heads_idx.shape)),
                    sparse.coo_matrix(changes_synapses), filename, times[i], add_to_prev=True)
                # Write lifetimes of deleted synapses to the output file.
                #save_extra(lifetimes, "lifetimes", filename)

        # Divide the time interval into n_steps equally spaced steps of stepwidth h.
        n_steps = int((times[i+1] - times[i]) / h)

        # Take n_steps random steps in the Markov chain.
        for j in range(n_steps):
            t = times[i] + (j+1) * h  # current time
            steps_survived = monte_carlo_step(network, steps_survived, alpha, beta_noise, h, t,
                                              filename)
            # Update the perturbation vector since synapses might have been permuted during the wiring phase.
            

    # Save min lifetimes of all synapses that survived until the end of the simulation.
    #min_lifetimes = steps_survived[steps_survived >= 0] * h
    #save_extra(min_lifetimes, "min_lifetimes", filename)

    end = datetime.now()
    print("DONE. Computation time:", end - start)
