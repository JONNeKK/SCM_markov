import numpy as np
from scipy import sparse
import h5py
from copy import deepcopy


class Network:
    """An instance of Network is a network as defined within the Directed Configuration Model (DCM) with fixed vectors
    X and Y of potential out- and in-degrees of all nodes, respectively, and a specific initial configuration.
    In general, different configurations can give rise to the same adjacency matrix W (degeneracy).

    DCM: see Gasparovic, Gallinaro, Rotter (2022) 'Associative remodeling and repair in self-organizing neuronal
             networks.' (in preparation)
    """

    def __init__(self, half_edges, synapses, heads_idx=None) -> object:
        """tails: Numpy array of potential out-degrees where the number of repeats of a node's index/label corresponds
                  to its number of maximally available out-going half-edges. dtype=int.
                  E.g. [0, 0, 1, 2, 2, 2] if node '0' has 2 tails, node '1' has 1 tail, and node '2' has 3 tails.
                  Node indices/labels should always be in the range [0, n-1] where n is the number of nodes in total.
            heads: Numpy array of potential in-degrees for every node. dtype=int.
                   Defined analogously to tails.
            synapses: Sparse boolean array of existing synapses in the initial configuration.
                      1 if the head and tail at the same positions in the heads and tails arrays form a synapse,
                      0 if not.
            heads_idx: Numpy array of the heads' indices. dtype=int. Needed for distinguishing degenerate micro-
                       configurations.
                       If not passed, the indices will be integers from 0 to n-1 in ascending order.
        """
        #self.tails = tails
        self.half_edges = half_edges
        self.synapses = synapses

        # Number of nodes in total. Nodes without any heads and tails are neglected because they will never be connected
        # to any other node of the network.
        self.n = np.max(self.half_edges) + 1

        self.x = np.bincount(half_edges)  # vector of potential out-degrees
        #self.y = np.bincount(heads)  # vector of potential in-degrees
        self.z = np.sum(self.x)/2  # maximum number of edges

        self.z_hat = self.z - self.synapses.sum()/2  # number of unbound edges

        if heads_idx is None:
            heads_idx = np.arange(0, self.z*2, dtype='int')
        self.heads_idx = heads_idx

        self.w = None
        self.microconf = None

    def compute_w(self):
        """Compute the edge count matrix W corresponding to the network's configuration."""
        w = np.zeros([self.n, self.n])
        #print(np.where(self.synapses==1))
        np.add.at(w, (self.half_edges[np.where(self.synapses==1)[0]],self.half_edges[np.where(self.synapses==1)[1]]), 1)
        self.w = w

    
    # Not really needed anymore. 
    # z_hat can directly be computed via the synapses vector and is stored as an attribute now in __init__() already.
    # The Solution class has its own method for computing the time evolution of z_hat anyways.
    # Just here for record...
    def compute_z_hat(self, w=None, axis=None):
        """Compute the number of free edges.

        w: Adjacency matrix W. Use default if for a single point in time. Else pass a 3d array of W over time.
        axis: Choose None for single point in time, (1, -1) for time evolution."""
        evo = True

        if w is None:
            evo = False
            # Compute the adjacency matrix if not done yet.
            if self.w is None:
                self.compute_w()
            w = self.w

        z_hat = self.z - (np.sum(w, axis=axis)/2)
        if not evo:
            self.z_hat = z_hat
        else:
            return z_hat
    

    def entropy(self):
        """Compute the entropy of W."""
        if self.w is None:
            self.compute_w()
        w = deepcopy(self.w)
        w = np.longdouble(w)  # for higher precision

        

        w_over_bound = w / (self.z - self.z_hat)

        # Set summands for which w_ij==0 to zero.
        # TODO: also set to zero for vals close to zero?
        entropy_matrix = np.piecewise(w_over_bound, condlist=[w_over_bound == 0],
                                      funclist=[0, lambda w: - w * np.log2(w)])

        # Sort all summands.
        entropy_matrix = np.sort(entropy_matrix.flatten())

        return np.sum(entropy_matrix)

    def tr_w(self):
        """Compute the trace of the adjacency matrix."""
        # Compute the adjacency matrix if not done yet.
        if self.w is None:
            self.compute_w()

        return np.trace(self.w*np.diag(np.full(self.n,0.5)))

    def compute_microconf(self):
        """Compute the matrix of the network's micro-configuration.

        Returns a (z, z)-array with entries ij = 1 if the out-going half-edge j and in-going half-edge i form a
        synapse, 0 else.
        Labeling of the half-edges starts from node 0 and then follows ascending node labels. Labels for in- and out-
        going half-edges go from 0 to z-1, respectively.
        """
        mc = np.zeros([self.z, self.z])
        np.add.at(mc, (self.heads_idx[self.synapses == 1], np.where(self.synapses == 1)[0]), 1)
        self.microconf = mc


class Solution:
    """An instance of Solution allows to analyze the outcome of a MCMC simulation of a network object."""

    def __init__(self, filename, update=False) -> object:
        """ filename: String, name of the hdf5.output file of the simulation (without the ending '.hdf5')."""
        self.filename = filename

        with h5py.File(filename + '.hdf5', 'r') as file:
            self.time = file['time'][:]

            # Load extra datasets.
            if "extra" in file.keys():
                self.extra = {}
                for key, val in file["extra"].items():
                    self.extra[key] = val[()]

            # Load extra time-dependent datasets.
            # Assumes that all additional time-dependent quantities have been saved at every time step, also for the
            # initial condition!
            self.extra_t = {}
            for key in file['0'].keys():
                if key not in ['half_edges', 'synapses']:
                    self.extra_t[key] = np.array([file[str(i)][key] for i in range(len(self.time))])


            # Initial micro-configuration.
            init_data = file['0']
            self.init_half_edges = sparse.coo_matrix((init_data['half_edges']['data'][:],
                                                 (np.zeros_like(init_data['half_edges']['col'][:]),
                                                  init_data['half_edges']['col'][:])),
                                                shape=init_data['half_edges'].attrs['shape'])
            self.init_heads_idx = sparse.coo_matrix((init_data['heads_idx']['data'][:],
                                                     (np.zeros_like(init_data['heads_idx']['col'][:]),
                                                      init_data['heads_idx']['col'][:])),
                                                    shape=init_data['heads_idx'].attrs['shape'])
            self.init_synapses = sparse.coo_matrix((init_data['synapses']['data'][:],
                                                    (init_data['synapses']['col'][:],
                                                     init_data['synapses']['row'][:])),
                                                   shape=init_data['synapses'].attrs['shape'])
        
        #print('synapses: ',self.init_synapses)
        #print('all three [0]',self.init_half_edges.toarray()[0].astype(int),
        #                       self.init_synapses.toarray().astype(int),
        #                       self.init_heads_idx.toarray()[0].astype(int))
        # Create a network object with the initial object of the simulation.
        self.network = Network(self.init_half_edges.toarray()[0].astype(int),
                               self.init_synapses.toarray().astype(int),
                               heads_idx=self.init_heads_idx.toarray()[0].astype(int))

        self.network.compute_w()
        self.init_w = self.network.w
        #self.network.compute_z_hat()
        self.init_z_hat = self.network.z_hat

    def __load_time_interval__(self, t_fin=None, end=False, h=True, h_idx=False, s=True):
        """Load the changes to the initial condition up to a time t_fin.

        (Private method only for calls from other methods inside the Solution class. Do not call from outside!)

        t_fin: Float indicating the time up to which the data should be loaded. Should be within the time range of the
               corresponding simulation. Optional: Does not need to be passed if end == True (see below).
        end: Bool indicating if the data up to the end of the simulation (all data stored in the output) should be
             loaded (True) or not (False). Should be False if t_fin < final time point of the simulation (default).
        h, h_idx, s: Bools indicating whether the changes to the heads, the heads' indices, and/or synapses vectors
                     should be loaded. Enables to only load changes to quantities of interest.

        Returns lists of sparse matrices of changes to the initial heads and synapses vectors over time up to t_fin.
        Every element in the list is a sparse coo matrix containing the changes of the respective vector relative to the
        previous time step.
        """
        if not end and t_fin == None:
            raise Exception("No upper limit for the time interval was given. Either pass a value or set end=True if it "
                            "is supposed to be the end of the simulation.")

        def find_nearest(t, t_arr):
            """Find the index of the element in an array that is closest to a given value.

            t: Float giving the time one is interested in.
            t_arr: Array of time points available in the output.

            Returns the index of the element in t_arr that is closest to the value of t.
            """
            idx = (np.abs(t_arr - t)).argmin()
            print("Nearest time point found:", t_arr[idx])
            return idx

        with h5py.File(self.filename + '.hdf5', 'r') as file:
            # Find the label of the group in the output file corresponding to the end of the time interval of interest.
            if end:
                max_idx = len(self.time) - 1
            else:
                max_idx = find_nearest(t_fin, self.time)

            # Load changes up to that time.
            half_edges_changes = []
            heads_idx_changes = []
            synapses_changes = []

            # TODO: Is there a more efficient way to do this?
            for idx in range(1, max_idx + 1):
                data = file[str(idx)]
                if h:
                    half_edges_changes.append(sparse.coo_matrix((data['half_edges']['data'][:],
                                                            (np.zeros_like(data['half_edges']['col'][:]),
                                                             data['half_edges']['col'][:])),
                                                           shape=data['half_edges'].attrs['shape']))
                if h_idx:
                    heads_idx_changes.append(sparse.coo_matrix((data['heads_idx']['data'][:],
                                                                (np.zeros_like(data['heads_idx']['col'][:]),
                                                                 data['heads_idx']['col'][:])),
                                                               shape=data['heads_idx'].attrs['shape']))
                if s:
                    synapses_changes.append(sparse.coo_matrix((data['synapses']['data'][:],
                                                               (data['synapses']['col'][:],
                                                                data['synapses']['row'][:])),
                                                              shape=data['synapses'].attrs['shape']))
        for i in range(len(synapses_changes)):
            pass
            #print(synapses_changes[i].toarray())
        return half_edges_changes, heads_idx_changes, synapses_changes

    def __add_changes__(self, half_edges_changes=None, heads_idx_changes=None, synapses_changes=None):
        """Add up the changes to the initial condition to obtain the micro-configuration at a certain time.

        (Private method only for calls from other methods inside the Solution class. Do not call from outside!)

        heads_changes: List of sparse matrices in coo format giving the changes to the heads vector at every step in the
                       Markov simulation.
        heads_idx_changes: Analogous to heads_changes but for the vector of the heads' indices.
        synapses_changes: Analogous to heads_changes but for the synapses vector.
        If not passed, no changes will be added and an empty list is returned for that quantity.

        Returns: curr_heads: Heads vector at the time of interest resulting from the summation of all changes up to that
                             time. Sparse matrix in csr format.
                 curr_heads_idx: Analogous to curr_heads but for the vector of the heads' indices.
                 curr_synapses: Analogous to curr_heads but for the synapses vector.
        """
        curr_half_edges, curr_heads_idx, curr_synapses = [], [], []

        if half_edges_changes is not None:
            curr_half_edges = self.init_half_edges.tocsr()
            for h_changes in half_edges_changes:
                curr_half_edges += h_changes.tocsr()

        if heads_idx_changes is not None:
            curr_heads_idx = self.init_heads_idx.tocsr()
            for h_idx_changes in heads_idx_changes:
                curr_heads_idx += h_idx_changes.tocsr()

        if synapses_changes is not None:
            curr_synapses = self.init_synapses.tocsr()
            for s_changes in synapses_changes:
                curr_synapses += s_changes.tocsr()

        return curr_half_edges, curr_heads_idx, curr_synapses

    def w(self, t=None, end=False):
        """Compute the adjacency matrix W at time t or the end of the simulation if end==True."""
        half_edges_changes, _, synapses_changes = self.__load_time_interval__(t_fin=t, end=end)
        # Heads and synapses vectors at time t.
        half_edges_t, _, synapses_t = self.__add_changes__(half_edges_changes=half_edges_changes, synapses_changes=synapses_changes)

        # Update the network object.
        self.network.half_edges = half_edges_t.toarray()[0].astype(int)
        self.network.synapses = synapses_t.toarray().astype(int)

        # Delete arrays that are not needed anymore to free memory space.
        del half_edges_changes, synapses_changes
        del half_edges_t, synapses_t

        self.network.compute_w()
        return self.network.w

    def z_hat_t(self, t_fin=None, end=False):
        """Compute the number of free edges over time up to a time t_fin or the end of the simulation.

        Returns a 1d numpy array of the number of free edges at every time step from the beginning of the simulation
        until t_fin/the end of the simulation.
        """
        z_hat_t = [self.init_z_hat]

        _, _, synapses_changes = self.__load_time_interval__(t_fin=t_fin, end=end, h=False, h_idx=False, s=True)

        for s_changes in synapses_changes:
            z_hat_t.append(z_hat_t[-1] - (s_changes.sum()/2))

        del synapses_changes

        return np.array(z_hat_t)

    def entropy(self, t=None, end=False):
        """Compute the entropy of the adjacency matrix W at time t."""
        _ = self.w(t=t, end=end)
        return self.network.entropy()

    def tr_w(self, t=None, end=False):
        """Compute the trace of the adjacency matrix W at time t."""
        w = self.w(t=t, end=end)
        return np.trace(w)

    def microconf(self, t=None, end=False):
        """Compute the matrix of the micro-configuration at time t or at the end of the simulation if end==True."""
        _, heads_idx_changes, synapses_changes = self.__load_time_interval__(t_fin=t, end=end,
                                                                             h=False, h_idx=True, s=True)
        # Heads and synapses vectors at time t.
        _, heads_idx_t, synapses_t = self.__add_changes__(heads_idx_changes=heads_idx_changes,
                                                          synapses_changes=synapses_changes)

        # Update the network object.
        self.network.heads_idx = heads_idx_t.toarray()[0].astype(int)
        self.network.synapses = synapses_t.toarray()[0].astype(int)

        # Delete arrays that are not needed anymore to free memory space.
        del heads_idx_changes, synapses_changes
        del heads_idx_t, synapses_t

        self.network.compute_microconf()
        return self.network.microconf

    def survival_func(self, lifetimes=np.empty(0), min_lifetimes=np.empty(0)):
        """Compute the synaptic survival function based on the sampled lifetimes.

        lifetimes: Array of lifetime samples based on which the survival function should be computed.
                   If not passed, self.extra["lifetimes"] will be used which are the lifetimes of all synapses that were
                   formed and died again throughout the simulation.
                   Input lifetimes e.g. if one is only interested in the lifetimes of a certain subpopulation of
                   synapses.
        min_lifetimes: Array of lower bounds of the lifetime of synapses which survived until the end of the simulation.
                       One can only now that they survived for at least (!) this amount of time. Therefore they are not
                       included in the pdf of the lifetime distribution. Nevertheless, they should be accounted for in
                       the survival function!
                       If not passed, self.extra["min_lifetimes"] will be used which are the min lifetimes of all
                       synapses that were still present at the end of the simulation.

        Returns an array of times (sorted) and an array of the corresponding fractions of synapses which existed for at
        least that amount of time.
        """
        if len(lifetimes) == 0:
            try:
                lifetimes = self.extra["lifetimes"]
            except KeyError:
                raise Exception("Network has no attribute 'lifetimes'.")

        if len(min_lifetimes) == 0:
            try:
                min_lifetimes = self.extra["min_lifetimes"]
            except KeyError:
                print("No min lifetimes found.")
                pass

        lifetimes = np.append(lifetimes, min_lifetimes)

        frac_survived = (len(lifetimes) + 1 - np.arange(0, len(lifetimes) + 1)) / (len(lifetimes) + 1)

        # Remove duplicates. If several synapses had the same lifetimes, use the highest fraction for that time.
        times = np.append(0, lifetimes[np.argsort(lifetimes)])
        times_rev, unique_idx_last = np.unique(times[::-1], return_index=True)
        unique_idx_last = len(times) - unique_idx_last - 1

        return np.unique(times), frac_survived[unique_idx_last]

    def lifetime_dist(self, lifetimes=np.empty(0), min_lifetimes=np.empty(0)):
        """Compute the synaptic lifetime distribution.

        The lifetime distribution F(t) is the cumulative distribution of sampled lifetimes during a simulation and
        complementary to the survival function S(t) such that
        F(t) + S(t) = 1 for all t.

        lifetimes: Array of lifetime samples based on which the lifetime distribution should be computed.
                   If not passed, self.extra["lifetimes"] will be used which are the lifetimes of all synapses that were
                   formed and died again throughout the simulation.
                   Input lifetimes e.g. if one is only interested in the lifetimes of a certain subpopulation of
                   synapses.
        min_lifetimes: Array of lower bounds of the lifetime of synapses which survived until the end of the simulation.
                       One can only now that they survived for at least (!) this amount of time.
                       If not passed, self.extra["min_lifetimes"] will be used which are the min lifetimes of all
                       synapses that were still present at the end of the simulation.

        Returns an array of times (sorted) and an array fractions of synapses that have been existing for less than that
        amount of time.
        """
        return self.survival_func(lifetimes=lifetimes, min_lifetimes=min_lifetimes)[0],\
               1 - self.survival_func(lifetimes=lifetimes, min_lifetimes=min_lifetimes)[1]
