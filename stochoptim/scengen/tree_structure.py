# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List, Callable, Any, Dict, Tuple, Sequence, Union
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


class Node:

    def __init__(self, *children, **data):
        self._parent = None
        self._children = []
        self._data = data
        self.add(*children)

    @property
    def parent(self):	
        """Parent of the node"""
        return self._parent

    @property
    def children(self):
        """List of children of the node"""
        return self._children
    
    @children.setter
    def children(self, value):
        self._children = value

    @property
    def data(self):
        """Data associated to the node in the form of a dictionary"""
        return self._data

    @data.setter	
    def data(self, value):
        self._data = value

    @data.deleter
    def data(self):
        del self._data
    
    @property
    def is_leaf(self): 
        """True if the node is a tree leaf"""
        return len(self._children) == 0

    @property 
    def is_parent_of_leaf(self):
        if self.is_leaf:
            return False
        else: 
            return self.children[0].is_leaf
        
    @property
    def is_root(self):
         """True if the node is the tree root"""
         return self.parent is None
    
    def make_it_root(self):
        """Turn the node into a root"""
        self._parent = None
        
    @property
    def depth(self):	
        """Number of nodes from the node to the furthest leaf."""
        if self._children:
            return 1 + max(c.depth for c in self._children)
        else:
            return 1

    @property
    def level(self): 
        """Level of the node (root has level 0)"""
        if self._parent:
            return self._parent.level + 1
        else:
            return 0

    @property
    def root(self):
        """Root of the tree (for which self might be a subtree)"""
        return self.branch[0]
    
    @property
    def size(self):	
        """Nomber of nodes in the tree."""
        return 1 + sum(c.size for c in self._children)

    @property
    def address(self):
        if self._parent:
            return self._parent.address + tuple([self._parent._children.index(self)])
        else:
            return tuple([])
    
    @property
    def has_siblings(self):
        return self.parent and len(self.parent.children) >= 2

    @staticmethod
    def get_data_path(node, key, default=None):
        """Returns a dict mapping a level to the data values associated to 'key' on the path leading to a node"""
        return {n.level: n.data.get(key, default) for n in node.branch}

    @property
    def width(self) -> List[int]:
        """Computes the number of nodes on each level (excluding the root)."""
        width = [0] * (self.depth-1)
        for node in self.nodes:
            if not node.is_root:
                width[node.level-1] += 1
        return width
    
    @property
    def bushiness(self) -> List[float]:
        """Returns the list of branching factors at each level 
        (possibly non-integers if structure is non-symmetrical)"""
        return bushiness_from_width(self.width)
    
    @staticmethod
    def bushiness_from_width(width: List[int]) -> List[float]:
        return list(np.array(width) / np.concatenate(([1], width[:-1])))
    
    def has_key(self, 
                key: str, 
                including_levels: List[int] = None, 
                excluding_levels: List[int] = None):
        """ Returns True if the tree rooted at self has a specific data key. 
        
        including_levels: list[int] or None
            Levels to look into. If None, then all levels are looked into except those in excluding_levels.
            
        excluding_levels: list[int] or None
            Levels to be ignored. This argument is used only if `including_levels` is None.
        """
        if excluding_levels is None:
            excluding_levels = []
            
        if including_levels is not None:
            levels_to_look = including_levels
        else:
            levels_to_look = list(range(self.depth))
            for level in excluding_levels:
                levels_to_look.remove(level)
            
        for node in self.nodes:
            if node.level in levels_to_look:
                if key not in node.data.keys():
                    return False
        return True
        
    def node_at_address(self, address):
        """Returns the node located at an address (returns None if no node is at this address)."""
        if address == ():
            return self
        if len(self.children) <= address[0]:
            return None
        else:
            return self.children[address[0]].node_at_address(address[1:])
        
    # --- Iterables ---
    def nodes_at_level(self, t):
        """Returns an iterator on the nodes at level t in the subtree rooted at self"""
        return iter(n for n in self.nodes if n.level == t)

    def ancestor_branches(self, other): #find_split_point??
        """Returns the branches rooted at the closest common ancestor."""
        if self.depth < other.depth:
            return self.find_split_point(other.parent)
        if self.depth > other.depth:
            return self.parent.find_split_point(other)
        if self.parent == other.parent:
            return self, other
        else:
            return self.parent.find_split_branches(self.parent, other.parent)
        
    @property
    def leaves(self):
        """Returns an iterator on the tree leaves."""
        return iter(n for n in self.nodes if n.is_leaf)

    @property
    def branch(self):
        if self._parent:
            return self._parent.branch + [self]
        else:
            return [self]
        
    @property
    def nodes(self):
        """Returns an iterator that walks the tree exhaustively by depth first exploration."""
        yield self
        for c in self._children:
            for w in c.nodes:
                yield w

    @property
    def backward_nodes(self):
        """Returns an iterator that walks the tree exhaustively by backward exploration.
        (i.e., all nodes at level t+1 are explored before any node level at level t)."""
        for t in range(self.depth-1, -1, -1):
            yield from self.nodes_at_level(t)  
        
    @property
    def forward_nodes(self):
        """Returns an iterator that walks the tree exhaustively by breadth first exploration.
        (i.e., all nodes at level t are explored before any node level at level t+1)."""
        for t in range(self.depth):
            yield from self.nodes_at_level(t) 
                
    @property
    def trajectories(self):
        return ((n for n in l.branch) for l in self.leaves)
    
    # --- Structure update ---
    def copy(self, deep_copy=False):
        """Copy the structure of the tree. This copies the references to the nodes and their 
        data dictionary, but the data inside the dictionary still points to the same object 
        (i.e., this is not a deepcopy of the data attribute)."""
        if deep_copy:
            tree = Node(**deepcopy(self.data))
        else:
            tree = Node(**self.data)
        for c in self.children:
            tree.add(c.copy(deep_copy))
        return tree

    def add(self, *children):
        """Adds children to the node."""
        for c in children:
            self._children.append(c)
            c._parent = self
    
    def remove(self):
        """Delete the subtree rooted at self (including self itself)"""
        if self.parent:
            self.parent.children.remove(self)
            self._parent = None
        else:
            raise Exception("cannot remove root")

    def remove_branch(self):
        """Same as remove() but additionally keep deleting nodes backwards along the path 
        leading to self if those nodes don't have children. This deletion ensures that the 
        tree leaves are all at the same level."""
        p = self.parent
        self.remove()
        if p.is_leaf:
            p.remove_branch()
            
    # --- Data update ---        
    def delete_data(self, keys: Union[str, List[str]]):
        """Delete all data under certain keys"""
        if isinstance(keys, str):
            keys = list(keys)
        for node in self.nodes:
            for key in keys:
                if node.data.get(key) is not None:
                    del(node.data[key])
                
    def append_data_dict(self, data_dict):
        """Append data to a tree in place.
        Argument:
        ---------
        data_dict: dict mapping {tuple: dict} 
            Dictionary mapping a node address to a node data (node data is a dictionary)
            Each node in the tree will be appended with the data found at its address in the argument dictionary.
        """    
        for node in self.nodes:
            node.data.update(data_dict.get(node.address, dict()))
          #  for key, value in data_dict.get(node_address, {}).items():
            #    node.data[key] = value
    
    def sort(self, key):
        """Sort (in place) the tree according to some criteria function.
        Argument:
        ---------
        key: function mapping a node to a number.
        """
        self._children.sort(key=key)
        for c in self.children:
            c.sort(key=key)
            
    # --- Encoding ---
    @property            
    def topology(self): 
        """Returns the underlying topology of the tree structure. 
        A topology is the most concise way to encode arbitrary structures."""
        if self.is_leaf:
            pass
        elif self.is_parent_of_leaf: 
            return len(self._children)
        else:
            return tuple([child.topology for child in self._children])

    def get_data_dict(self, with_keys=None, without_keys=None):
        """Turn a tree into a dictionary that maps each node address to its node data.
        
        Argument:
        ---------
        with_keys: list of strings or None
            If not None, it is the data keys to be kept. 
            If None, then all keys in the list will be kept except those in without_keys.
            
        without_keys: list of strings or None
            This argument is used only if 'with_keys' is None. In that case it contains the data keys to be ignored.
        """
        assert with_keys is None or without_keys is None, "One of `with_keys` or `without_keys` must be None"
        if without_keys is None:
            without_keys = []
        if with_keys is not None:
            keys_to_keep = lambda node: set(with_keys)
        else:
            keys_to_keep = lambda node: set(node.data.keys()) - set(without_keys)  
        return {node.address: {key: value for key, value in node.data.items()
                                       if key in keys_to_keep(node)} for node in self.nodes}

    # --- Save and load --- 
    def save_topology(self, path, extension):
        """Save the underlying tree topology into a file.
        This is the best way to save large trees with empty data. 
        For large trees with non-empty data, use .to_file().
        Arguments:
        ----------
        path: string
        
        extension: {'txt', 'pickle'} 
        """
        path = path + f'.{extension}'
        if extension == 'txt':
            with open(path, "w") as f:
                f.write(repr(self.topology))
        elif extension == 'pickle':
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self.topology, f)
        else:
            TypeError(f"Extension should be in {'pickle', 'txt'}, not {extension}.")
            
    def to_file(self, path, extension, with_keys=None, without_keys=None):
        """Save the tree and its data in a file. This is the way to save
        large trees with non-empty data.
        
        Argument:
        ---------
        path: str
        
        extension: {'txt', 'pickle'}
        
        with_keys: list of str or None (default: None)
            List of data keys that will be saved. 
            If None, then all keys in the tree will be saved except those in `without_keys`.
            
        without_keys: list of str or None (default: None)
            List of data keys that will not be saved. 
            This argument is used only if `with_keys` is None.
        """
        if with_keys is not None:
            assert isinstance(with_keys, list), \
            f"Argument `with_keys` should be of type list, not {type(with_keys)}."
        elif without_keys is not None:
            assert isinstance(without_keys, list), \
            f"Argument `without_keys` should be of type list, not {type(without_keys)}."
        data_dict = self.get_data_dict(with_keys, without_keys)
        path = path + f'.{extension}'
        if extension == 'txt':
            np.set_printoptions(threshold=np.inf) # no limit on the number of elements printed in an array
            with open(path, "w") as f:
                f.write(repr(data_dict).replace("},", "},\n"))
        elif extension == 'pickle':
            import pickle
            with open(path, "wb") as f:
                pickle.dump(data_dict, f)
        else:
            TypeError(f"Extension should be 'pickle' or 'txt', not {extension}.")
            
    @classmethod
    def from_file(cls, path, extension):
        """Constructor of a tree and its data directly from a file. 
        This is the way to load large trees with non-empty data. 
        (Reverse of the .to_file() method.)
        Arguments:
        ---------
        path: string
        
        extension: {'txt', 'pickle'}
        """
        path = path + f'.{extension}'
        if extension == 'txt':
            with open(path, "r") as f:
                file_str = f.read()
                file_str = file_str.replace('array', 'np.array')
                file_str = file_str.replace('nan', 'np.nan')
                for type_ in ['int8', 'int16', 'int64', 'float16', 'float32']:
                    file_str = file_str.replace(f'dtype={type_}', f'dtype="{type_}"')
                data_dict = eval(file_str)
        elif extension == 'pickle':
            import pickle
            with open(path, "rb") as f:
                data_dict = pickle.load(f)
        return cls.from_data_dict(data_dict)
    
    # --- Alternative constructors ---
    @classmethod
    def from_data_dict(cls, data_dict: Dict[Tuple[int], dict]):
        """Constructor of a tree from a dictionary that maps each node address to its node data.
        It is the reverse of the .get_data_dict() method, i.e., tree = from_data_dict(tree.get_data_dict())"""
        tree = cls._from_addresses(list(data_dict.keys()))    
        for node in tree.nodes:
            node.data = data_dict.get(node.address, {})
        return tree
    
    @classmethod
    def _from_addresses(cls, addresses):
        """
        Constructor of a tree structure from a list of node addresses.

        Arguments:
        ----------
        addresses: list of tuples
            Each tuple corresponds to the address of a node in the structure (as would be given 
            by node.address()).
        
        Returns:
        --------
        Node: the root node of the specified tree structure
        """
        root = cls()
        for address in addresses:
            if root.node_at_address(address) is not None:
                continue
            else:
                root.node_at_address(address[:-1]).add(cls())  
        return root

    @classmethod
    def from_topology(cls, topology: Sequence[Sequence[int]], _tree=None):
        """
        Constructor of a tree structure from a topology. 
        A topology is the most general way to build any possible type of structure.
        
        Arguments:
        ----------
        topology: tuple/list of int nested into (possibly several) layers of tuple/list.
            The integers are the number of sibling leaves. Integers (and sub-tuples) are inside 
            the same tuple if they have the same parent. 
        
        _tree: unused argument (internally used for the recurrent calls to the method)
            
        Returns:
        --------
        Node: the root node of the specified tree structure
        """
        if _tree is None:
            _tree = cls()
        if isinstance(topology, int):
            _tree.add(*[cls() for i in range(topology)])
            return _tree
        else: 
            if _tree.is_leaf:
                _tree.add(*[cls() for i in range(len(topology))]) 
            for item, leaf in zip(topology, list(_tree.leaves)):
                cls.from_topology(item, leaf)
            return _tree
        
    @classmethod
    def from_recurrence(cls, 
                        max_level: int, 
                        init: int, 
                        recurrence: Dict[int, Sequence[int]], 
                        _tree=None):
        """
        Constructor of a tree structure where the number of child nodes satisfies a recurrence relation.
        
        Arguments:
        ----------
        max_level: integer
            Level of the tree leaves (root has level 0)
            
        init: integer
            Number of nodes at level 1 (to initialize the recurrence).
            
        recurrence: dictionary that maps an integer k to a k-tuple of integers 
            All integers must be positive and correspond to the number of sibling nodes. 
            (Each integers found in the dict values must also be found in the keys.)
            
        _tree: unused argument (internally used for the recurrent calls to the method)

        Example:     
            recurrence = {1: (2,), 2: (1,3), 3: (1,2,3)}) 
            will produce a tree where:
                - a single node (without sibling) will give birth to 2 child nodes,
                - 2 sibling node will give birth to 1 and 3 children, 
                - 3 sibling nodes will give birth to 1, 2, and 3 children.

        Returns:
        --------
        Node: the root node of the specified tree structure
        """
        if _tree is None:
            _tree = cls(*[cls() for _ in range(init)])
        else:
            _tree.add(*[cls() for _ in range(init)])
        if max_level == 1:
             return _tree
        else:
            for i, node in enumerate(_tree.children):
                if node.is_leaf:
                    cls.from_recurrence(max_level-1, recurrence[len(_tree.children)][i], recurrence, node)
            return _tree
    
    @classmethod
    def from_bushiness(cls, bushiness: Sequence[int]):
        """
        Constructor of a tree structure with a time-dependent branching factor.
        
        Arguments:
        ----------
        bushiness: tuple of integers
            Integer at location i in the tuple is the branching factor at the i-th level
            
        Returns:
        --------
        Node: the root node of the specified tree structure
        """
        integer_bushiness = np.array(bushiness).astype('int')
        assert (np.array(bushiness) == integer_bushiness).all(), \
            f"The bushiness must only contain integer values, not {bushiness}"
        tree = cls()
        for t in range(len(integer_bushiness)):
            for leaf in list(tree.leaves):
                leaf.add(*[cls() for k in range(integer_bushiness[t])])
        return tree
    
    # --- Plots ---
    def plot(self, 
             print_on_nodes: Callable[['Node'], str] = None,
             print_on_edges: Callable[['Node'], str] = None,
             max_stage=None,
             centered_around_leaves=True, 
             sorting_fct=None,
             color_nodes='blue',
             size_nodes=20,
             color_edges='blue',
             width_edges=0.5,
             size_txt_nodes=12,
             color_txt_nodes='black',
             size_txt_edges=10,
             color_txt_edges='black',
             figsize=(5,5),
             xlim=None,
             ylim=None,
             show_xaxis=False,
             show_yaxis=False,
             show_frame=False,
             to_file=None,
             extension=".pdf",
             ax=None):
    
        if sorting_fct is not None:
            tree = self.copy()
            tree.sort(sorting_fct)
        else: 
            tree = self
            
        if max_stage is not None:
            tree = tree.copy()
            for node in list(tree.nodes_at_level(max_stage+2)):
                node.remove()
        else:
            max_stage = tree.depth
            
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # define the color and size functions for each node and edge
        color_nodes_fct = lambda node: color_nodes if not callable(color_nodes) else color_nodes
        size_nodes_fct = lambda node: size_nodes if not callable(size_nodes) else size_nodes
        color_edges_fct = lambda node: color_edges if not callable(color_edges) else color_edges 
        width_edges_fct = lambda node: width_edges if not callable(width_edges) else width_edges 
        size_txt_nodes_fct = lambda node: size_txt_nodes if not callable(size_txt_nodes) else size_txt_nodes
        color_txt_nodes_fct = lambda node: color_txt_nodes if not callable(color_txt_nodes) else color_txt_nodes
        size_txt_edges_fct = lambda node: size_txt_edges if not callable(size_txt_edges) else size_txt_edges
        color_txt_edges_fct = lambda node: color_txt_edges if not callable(color_txt_edges) else color_txt_edges
        
        if print_on_nodes is None:
            print_on_nodes = lambda node: None
        if print_on_edges is None:
            print_on_edges = lambda node: None
            
        # get the vertical position for the leaves
        pos_y = {leaf.address: i for i, leaf in enumerate(tree.leaves)}
    
        # get the vertical position for all the nodes but the leaves
        for node in tree.backward_nodes:
            if not node.is_leaf:
                if centered_around_leaves:
                    pos_y[node.address] = np.mean([pos_y[n.address] for n in node.leaves])
                else:
                    pos_y[node.address] = np.mean([pos_y[n.address] for n in node.children])
                    
        # plot the nodes
        for node in tree.nodes:
            if node.level == max_stage+1:
                continue
            x, y = node.level, pos_y[node.address]
            ax.scatter(x, y, color=color_nodes_fct(node), s=size_nodes_fct(node))
            
            # write text on node if required
            txt = print_on_nodes(node)
            if txt is not None:
                fontsize = size_txt_nodes_fct(node)
                color = color_txt_nodes_fct(node)
                if not node.parent:
                    ax.text(x-0.1, y, txt, fontsize=fontsize, color=color, 
                            verticalalignment='center', horizontalalignment='right')
                elif node.is_leaf:
                    ax.text(x+0.1, y, txt, fontsize=fontsize, color=color,  
                            verticalalignment='center', horizontalalignment='left')
                else:
                    ax.text(x-0.05, y, txt, fontsize=fontsize, color=color, 
                            verticalalignment='center', horizontalalignment='right')
    
        # plot the edges
        for node in tree.nodes:
            if node.level == max_stage+1:
                continue
                
            if node.parent:
                x = [node.level, node.parent.level]
                y = [pos_y[node.address], pos_y[node.parent.address]]
                ax.plot(x, y, color=color_edges_fct(node), linewidth=width_edges_fct(node))
                
                # write text on edge if required
                txt = print_on_edges(node)
                if txt is not None:
                    ax.text(np.mean(x), np.mean(y), txt, 
                                fontsize=size_txt_edges_fct(node), color=color_txt_edges_fct(node), 
                                verticalalignment='center', horizontalalignment='center')
            
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.get_xaxis().set_visible(show_xaxis)
        ax.get_yaxis().set_visible(show_yaxis)
        if not show_frame and not show_xaxis and not show_yaxis:
            ax.axis('off')
            
        if to_file is not None:
            plt.savefig(to_file + extension)
        return ax

    @staticmethod
    def plot_multiple_trees(*trees, along_x=True, figsize=(5,5), to_file=None, extension=".pdf", **kwargs):
        """
        Plots several trees side by side.
        
        Arguments:
        ----------
        *trees: instances of Node
            All trees to be plotted.
            
        along_x: boolean
            If True, trees are displayed from left to right; if False, from up to down.
            
        figsize: 2-tuple of numbers
            Size of the whole figure (not of each figure individually).
            
        to_file: string
            Path where the figure will be saved (not saved if None).
           
        extension: string
            Extension in which the figure will be saved.
            
        **kwargs: 
            All keyword args of .plot() except 'ax', 'figsize' and 'to_file'.
        """
        assert len(trees) > 1, "There must be 2 trees at least"
        kwargs.pop('ax', None), kwargs.pop('figsize', None), kwargs.pop('to_file', None)
        if along_x:
            fig, axes = plt.subplots(1, len(trees), figsize=figsize)
        else:
            fig, axes = plt.subplots(len(trees), 1, figsize=figsize)
            
        for index, tree in enumerate(trees):
            axes[index] = tree.plot(ax=axes[index], **kwargs)
            
        if to_file is not None:
            plt.savefig(to_file + extension)
        plt.show()
        
    # --- Representations ---
    def __repr__(self):
        return self.format()

    def format(self, level=0, margin=20, label=lambda n: "Node", data=None):
       # from stochoptim.util import Formatter
        if data is None:
            ndata = self.data.keys()
        else:
            ndata = data
        #fmt = Formatter(precision=3)
        s1 = '\n' + ('  ' * level) + str(label(self))
        s1 += ' ' * (margin - len(s1))
        #s1 += '\t'.join('{}={}'.format(k, fmt(self.data[k])) for k in ndata if k in self.data)
        s1 += '\t'.join('{}={}'.format(k, self.data[k]) for k in ndata if k in self.data)
        s1 += ''.join(c.format(level + 1, margin, label, data) for c in self.children)
        if level == 0:
            s1 = s1[1:]
        return s1

get_data_path = Node.get_data_path
bushiness_from_width = Node.bushiness_from_width
from_file = Node.from_file
from_data_dict = Node.from_data_dict
from_topology = Node.from_topology
from_recurrence = Node.from_recurrence
from_bushiness = Node.from_bushiness