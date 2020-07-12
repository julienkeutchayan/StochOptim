# -*- coding: utf-8 -*-

from typing import Iterator, Tuple, Optional, List

import math
import numpy as np
import scipy.special

def partitioning(collection: List[int], 
                 cardinality: Optional[int] = None, 
                 n_sample: Optional[int] = None, 
                 lowerbound: int = 1, 
                 uniform_sampling: bool = False) -> Iterator[Tuple[Tuple[int]]]:
    """Generates exhaustive or random partitions.
    
    Arguments:
    ----------
    collection: list of int
        Set to be partitioned. 
        
    cardinality: int >= 1 or None (default: None)
        Cardinality of the partition.
        If None, partitions of any cardinality are generated (implemented only for exhaustive partitions)
        
    n_sample: int >= 1 or None (default: None)
        Number of partitions randomly drawn.
        If None, partitions are exhaustively generated.
    
    lowerbound: int >= 1 (default: 1)
        Minimum size of each subset of the partition (implemented only for random partitions).
        
    uniform_sampling: bool (default: False)
        (Only for random partitions)
        If True, the random partitions are sampled from the uniform distribution over all partitions. 
        If False, the distribution is not uniform but it does give a positive probability to each partition.
        
    Returns:
    --------
    Iterator over partitions, where a partition is a tuple of tuples of int.
    """
    # random partitions
    if n_sample is not None:
        if cardinality is not None: # fixed cardinality
            if uniform_sampling:
                yield from sampling_uniform_partition(collection, cardinality, n_sample, lowerbound)
            else:
                yield from sampling_nonuniform_partition(collection, cardinality, n_sample)
        else: # any cardinality
            raise NotImplementedError("Random partitions of any cardinality is not implemented")
    # exhaustive partitions    
    elif n_sample is None: 
        if cardinality is not None: # fixed cardinality  
            if cardinality == 1:
                yield tuple(tuple(collection))
            else:
                for partition in algorithm_u(collection, cardinality):
                    yield tuple(tuple(subset) for subset in partition)
        else: # any cardinality
            for partition in exhaustive_partition(collection):
                yield tuple(tuple(subset) for subset in partition)        
            
                
def sampling_nonuniform_partition(collection: List[int], cardinality: int, n_sample: int):
    """Generates random partitions of a given cardinality.
    
    The partition are not drawn from the uniform distribution over the set of all partitions, but each partition has a
    positive probability.
    
    Arguments:
    ----------
    collection: list of int
        Set to be partitioned. 
        
    cardinality: int >= 1 
        Cardinality of the partition.
        
    n_sample: int >= 1
        Number of samples randomly drawn.

    Returns:
    --------
    Iterator over partitions, where a partition is a tuple of tuples of int.
    """
    all_indices = list(range(len(collection)))
    possible_split_indices = all_indices[1:-1] # extreme indices cannot be used as break points (otherwise some clusters will be empty)
    for _ in range(n_sample):
        np.random.shuffle(collection)
        split_indices = np.sort(np.random.choice(possible_split_indices, replace=False, size=cardinality-1))
        random_partition = tuple([tuple(part) for part in np.split(collection, split_indices)])
        yield random_partition
    
def sampling_uniform_partition(collection: List[int], cardinality: int, n_sample: int, lowerbound: int = 1):
    """Generates random partitions of a given cardinality drawn from the uniform distribution.
    
    Arguments:
    ----------
    collection: list of int
        Set to be partitioned. 
        
    cardinality: int >= 1
        Cardinality of the partition.
        
    n_sample: int >= 1
        Number of samples randomly drawn.
    
    lowerbound: int >= 1 (default: 1)
        Minimum size of each subset of the partition.
        
    Returns:
    --------
    Iterator over partitions, where a partition is a tuple of tuples of int.
    """
    # (step 1) determine the number of possible partitions    
    nber_partitions_dict = {} # {(size of the clusters): number of partitions having this size of clusters}
    # loop over all integer partitions (an integer partition is tuple giving the cluster sizes)
    for clusters_size in integer_partition(len(collection), cardinality, lowerbound): 
        # compute how many set partitions exist for these cluster sizes
        nber_partitions_dict[clusters_size] = multinomial_coefficients(*clusters_size) 
         # compute how many clusters of the same size are in each partition
        multiple_dict = {c1: len([1 for c2 in clusters_size if c1 == c2]) for c1 in clusters_size}
        divisor = np.prod([math.factorial(value) for value in multiple_dict.values()])
        nber_partitions_dict[clusters_size] = int(nber_partitions_dict[clusters_size] / divisor)

    # (step 2) sample the size of each cluster in the partition
    number_of_partitions = sum(nber_partitions_dict.values()) # number of possible partitions
    # {cluster_sizes: probability}
    proba_partitions_dict = {key: value / number_of_partitions for key, value in nber_partitions_dict.items()} 
    # {index: cluster_sizes}
    index_dict = {index: key for index, key in enumerate(proba_partitions_dict.keys())} 
     # [index_1, index_2, ..., index_{n_sample}]
    random_index = np.random.choice(list(index_dict.keys()), size = n_sample, p = list(proba_partitions_dict.values()))
    # [cluster_size_1, cluster_size_2, ..., cluster_size_{n_sample}]
    random_clusters_size = [index_dict[k] for k in random_index] 

    # (step 3) sample the elements in each cluster of the partition
    for clusters_size in random_clusters_size:
        np.random.shuffle(collection)
        random_permutation = collection
        random_partition = []
        for size in clusters_size:
            random_cluster = list(random_permutation[:size])
            random_permutation = random_permutation[size:]
            random_partition += [random_cluster]
        yield tuple(tuple(subset) for subset in random_partition)

        
def exhaustive_partition(collection: List[int]): 
    """Generates all partitions of any cardinality.
    (implementation from: https://stackoverflow.com/questions/19368375/set-partitions-in-python)
    
    Arguments:
    ----------
    collection: list of int
        Set to be partitioned. 
        
    Returns:
    --------
    Iterator over partitions, where a partition is a list of lists of int.
    """
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in exhaustive_partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [[first]] + smaller
            
                
def algorithm_u(collection: List[int], cardinality: int):
    """Generates all partitions of a given cardinality.
    (implementation from: https://codereview.stackexchange.com/questions/1526/finding-all-k-subset-partitions)
    
    Arguments:
    ----------
    collection: list of int
        Set to be partitioned. 
        
    cardinality: int >= 2
        Cardinality of the partition.
        
    Returns:
    --------
    Iterator over partitions, where a partition is a list of lists of int.
    """
    def visit(n, a):
        ps = [[] for i in range(cardinality)]
        for j in range(n):
            ps[a[j + 1]].append(collection[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(collection)
    a = [0] * (n + 1)
    for j in range(1, cardinality + 1):
        a[n - cardinality + j] = j - 1
    return f(cardinality, n, 0, n, a)

    
def integer_partition(number: int, cardinality: int, lowerbound: int = 1):
    """Exhaustive enumeration of an integer partition. 
    
    Arguments:
    ----------
    number: int >= 1
        The integer being partitionned.
    
    cardinality: int <= `number`
        Number of elements partitionning the integer

    lowerbound: int >= 1 (default: 1)
        Lowerbound for the elements partitionning the integer.
        
    Returns:
    --------
        Tuple[int]: (n_1, ..., n_{cardinality}) such that: 
            (i) number = n_1 + ... + n_{cardinality}, 
            (ii) n_i is an integer and n_i >= lowerbound for all i=1, ..., cardinality, 
            (iii) n_1 <= n_2 <= ..., n_{cardinality}.
    """
    if cardinality == 1:
        if number >= lowerbound:
            yield (number,)
            return

    for i in range(lowerbound, number - (cardinality - 1) * lowerbound + 1):
        for t in integer_partition(number - i, cardinality - 1, i):
            yield (i,) + t
            
            
def multinomial_coefficients(*args: int):
    """Returns the multinomial coefficient computed recursively (to avoid dealing with huge numbers)"""       
    multinomial_coeff = 1
    for i, n_i in enumerate(args, 1):
        multinomial_coeff *= int(scipy.special.binom(sum(args[:i]), n_i))
    return multinomial_coeff