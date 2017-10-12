"""This module encapsulate the basic elements of the UCCA annotation.

A UCCA annotation is practically a directed acyclic graph (DAG), which
represents a :class:Passage of text and its annotation. The annotation itself
is divided into :class:Layer objects, where in each layer :class:Node objects
are connected between themselves and to Nodes in other layers using
:class:Edge objects.

"""

import functools


# Max number of digits allowed for a unique ID
UNIQUE_ID_MAX_DIGITS = 5

# Attribute to ignore when comparing entities
IRRELEVANT_ATTRIBUTES = {"uncertain"}


# Used as the default ordering key function for ordered objects, namely
# :class:Layer and :class:Node .
def id_orderkey(node):
    """Key function which sorts by layer (string), then by unique ID (int).

    Args:
        node: :class:Node which we will to sort according to its ID

    Returns:
        a string with the layer and unique ID in such a way that sort will
        first order lexicography the layer ID then numerically the unique ID.

    """
    layer, unique = node.ID.split(Node.ID_SEPARATOR)
    return "{} {:>{}}".format(layer, unique, UNIQUE_ID_MAX_DIGITS)


def edge_id_orderkey(edge):
    """Key function which sorts Edges by its IDs (using :func:id_orderkey).

    Args:
        edge: :class:Edge which we wish to sort according to the ID of its
        parent and children after using :func:id_orderkey.

    Returns:
        a string with the layer and unique ID in such a way that sort will
        first order lexicography the layer ID then numerically the unique ID.

    """
    return Edge.ID_FORMAT.format(id_orderkey(edge.parent),
                                 id_orderkey(edge.child))


class UCCAError(Exception):
    """Base class for all UCCA package exceptions."""
    pass


class FrozenPassageError(UCCAError):
    """Exception raised when trying to modify a frozen :class:Passage."""
    pass


class DuplicateIdError(UCCAError):
    """Exception raised when trying to add an element with an existing ID.

    For each element, a unique ID must be assigned. If the ID of the new element
    is already present in the :class:Passage in some way, this exception is
    raised.

    """
    pass


class MissingNodeError(UCCAError):
    """Exception raised when trying to access a non-existent :class:Node."""
    pass


class UnimplementedMethodError(UCCAError):
    """Exception raised when trying to call a not-yet-implemented method."""
    pass


class ModifyPassage:
    """Decorator for changing a :class:Passage or any member of it.

    This decorator is mandatory for anything which causes the elements in
    a :class:Passage to change by adding or removing an element, or changing
    an attribute.

    It validates that the Passage is not frozen before allowing the change.

    The decorator can't be used for __init__ calls, as at the stage of the
    check there are no instance attributes to check. So in such cases,
    a function that binds the object created with the Passage should be
    decorated instead (and should be called after the instance attributes
    are set).

    Attributes:
        fn: the function object to decorate

    """

    def __init__(self, fn):
        self.fn = fn

    def __get__(self, obj, cls):
        """Used to bind the function to the instance (add 'self')."""
        return functools.partial(self.__call__, obj)

    def __call__(self, *args, **kwargs):
        """Decorating functions which modify :class:Passage elements.

        :param args: list of all arguments, assuming the first is the object
                which modifies :class:Passage, and it has an attribute root
                which points to the Passage it is part of.
        :param kwargs: list of all keyword arguments

        :return The decorated function result.

        :raise FrozenPassageError: if the :class:Passage is frozen and can't be
                modified.

        """
        @functools.wraps(self.fn)
        def decorated(*args, **kwargs):
            if args[0].root.frozen:
                raise FrozenPassageError()
            return self.fn(*args, **kwargs)
        return decorated(*args, **kwargs)


class _AttributeDict:
    """Dictionary which stores attributes for any UCCA element.

    This dictionary is used to store attributes which are part of any
    element in the UCCA annotation scheme. It's advantage over regular
    dictionary is adhering to :class:Passage frozen status and modification
    decorators.

    Attributes:
        root: the Passage this object is linked with

    """

    def __init__(self, root, mapping=None):
        self._root = root
        self._dict = mapping.copy() if mapping is not None else dict()

    def __getitem__(self, key):
        return self._dict[key]

    def get(self, key, default=None):
        return self._dict.get(key, default)

    def equals(self, other):
        """True iff the two objects are equal (only dicts, w.o.r.t Passage).

        :param other: AttributeDict to compare to

        :return True iff the dictionaries are equal.

        """
        def omit_irrelevant(d):
            return {k: v for k, v in d.items() if k not in IRRELEVANT_ATTRIBUTES}
        return omit_irrelevant(self._dict) == omit_irrelevant(other._dict)

    @property
    def root(self):
        return self._root

    def copy(self):
        return self._dict.copy()

    @ModifyPassage
    def __setitem__(self, key, value):
        self._dict[key] = value

    @ModifyPassage
    def __delitem__(self, key):
        del self._dict[key]

    def __len__(self):
        return len(self._dict)

    def items(self):
        return self._dict.items()


class Edge:
    """Labeled edge between two :class:Node objects in UCCA annotation graph.

    An edge between Nodes in a :class:Passage is a simple object; it is a
    directed edge whose ID is derived by the parent and child of the edge,
    it is mostly immutable except for its attributes, and it is labeled with
    the connection type between the Nodes.

    Attributes:
        ID: ID of the Edge, constructed from the IDs of the two Nodes
        root: the Passage this object is linked with
        attrib: attribute dictionary of the Edge
        extra: temporary storage space for undocumented attributes and data
        tag: the string label of the Edge
        parent: the originating Node of the Edge
        child: the target Node of the Edge
        ID_FORMAT: format string which creates the ID of the Edge from
            the IDs of the parent (first argument to the formattinf string)
            and the child (second argument).

    """

    ID_FORMAT = "{}->{}"

    def __init__(self, root, tag, parent, child, attrib=None):
        """Creates a new :class:Edge object.

        :param see :class:Edge documentation.

        :raise FrozenPassageError: if the :class:Passage object we are part of
                is frozen and can't be modified.

        """
        if root.frozen:
            raise FrozenPassageError()
        self._tag = tag
        self._root = root
        self._parent = parent
        self._child = child
        self._attrib = _AttributeDict(root, attrib)
        self.extra = {}

    @property
    def tag(self):
        return self._tag

    @tag.setter
    @ModifyPassage
    def tag(self, new_tag):
        old_tag = self._tag
        self._tag = new_tag
        self._root._change_edge_tag(self, old_tag)

    @property
    def root(self):
        return self._root

    @property
    def parent(self):
        return self._parent

    @property
    def child(self):
        return self._child

    @property
    def attrib(self):
        return self._attrib

    @property
    def ID(self):
        return Edge.ID_FORMAT.format(self._parent.ID, self._child.ID)

    def equals(self, other, *, recursive=True, ordered=False,
               ignore_node=None, ignore_edge=None):
        """Returns whether self and other are Edge-equals.

        Edge-equality is determined by having the same tag and attributes.
        Recursive Edge-equality means that the Edges are equal, and their
        children are recursively Node-equal.

        :param other: an Edge object to compare to
        :param recursive: whether to compare recursively, defaults to True
        :param ordered: if recursive, whether the children are Node-equivalent
            w.r.t order (see Node.equals())
        :param ignore_node: function that returns whether to ignore a given node
        :param ignore_edge: function that returns whether to ignore a given edge


        :return True iff the Edges are equal.

        """
        return self.tag == other.tag and \
               self._attrib.equals(other._attrib) and \
               (not recursive or
                self.child.equals(other.child,
                                  ordered=ordered,
                                  ignore_node=ignore_node, ignore_edge=ignore_edge))

    def __repr__(self):
        return self.ID


class Node:
    """Labeled Node in UCCA annotation graph.

    A Node in :class:Passage UCCA annotation is an vertex in the annotation
    graph, which may be an internal vertex or a leaf, and is labeled with a
    tag that specifies both the :class:Layer it belongs to and it's ID in this
    Layer. It can have multiple children Nodes through :class:Edge objects,
    and these children are ordered according to an internal order function.

    Attributes:
        ID: ID of the Node, constructed from the ID of the Layer it belongs to,
            a separator, and a unique alphanumeric ID in the layer.
        root: the Passage this object is linked with
        attrib: attribute dictionary of the Node
        extra: temporary storage space for undocumented attributes and data
        tag: the string label of the Node
        layer: the Layer this Node belongs to
        incoming: a copy of the incoming Edges to this object
        outgoing: a copy of the outgoing Edges from this object
        parents: the Nodes which have incoming Edges to this object
        children: the Nodes which have outgoing Edges from this object
        orderkey: the key function for ordering the outgoing Edges
        ID_SEPARATOR: separator function between the Layer ID and the unique
            Node ID in the complete ID of the Node. Mustn't be alphanumeric.

    """

    ID_SEPARATOR = '.'

    def __init__(self, ID, root, tag, attrib=None, *,
                 orderkey=edge_id_orderkey):
        """Creates a new :class:Node object.

        :param see :class:Node documentation.

        :raise FrozenPassageError: if the :class:Passage object we are part of
                is frozen and can't be modified.

        """
        if root.frozen:
            raise FrozenPassageError()
        self._tag = tag
        self._root = root
        self._ID = ID
        self._attrib = _AttributeDict(root, attrib)
        self.extra = {}
        self._outgoing = []
        self._incoming = []
        self._orderkey = orderkey

        # After properly initializing self, add it to the Passage/Layer
        root._add_node(self)
        root.layer(self.layer.ID)._add_node(self)

    @property
    def tag(self):
        return self._tag

    @tag.setter
    @ModifyPassage
    def tag(self, new_tag):
        old_tag = self._tag
        self._tag = new_tag
        self._root._change_node_tag(self, old_tag)

    @property
    def root(self):
        return self._root

    @property
    def ID(self):
        return self._ID

    @property
    def attrib(self):
        return self._attrib

    @property
    def layer(self):
        return self._root.layer(self._ID.split(Node.ID_SEPARATOR)[0])

    @property
    def incoming(self):
        return tuple(self._incoming)

    @property
    def outgoing(self):
        return tuple(self._outgoing)

    @property
    def parents(self):
        return [edge.parent for edge in self._incoming]

    @property
    def children(self):
        return [edge.child for edge in self._outgoing]

    def __bool__(self):
        return True

    def __len__(self):
        return len(self._outgoing)

    def __getitem__(self, index):
        return self._outgoing[index]

    def __repr__(self):
        return Node.__name__ + "(" + self.ID + ")"

    @ModifyPassage
    def add(self, edge_tag, node, *, edge_attrib=None):
        """Adds another :class:Node object as a child of self.

        :param edge_tag: the label of the :class:Edge connecting between the
                Nodes
            node: the Node object which we want to have an Edge to
            edge_attrib: Keyword only, dictionary of attributes to be passed
                to the Edge initializer.

        :return the newly created Edge object

        :raise FrozenPassageError: if the :class:Passage object we are part of
                is frozen and can't be modified.

        """
        edge = Edge(root=self._root, tag=edge_tag, parent=self,
                    child=node, attrib=edge_attrib)
        self._outgoing.append(edge)
        self._outgoing.sort(key=self._orderkey)
        node._incoming.append(edge)
        node._incoming.sort(key=node._orderkey)
        self.root._add_edge(edge)
        return edge

    @ModifyPassage
    def remove(self, edge_or_node):
        """Removes the :class:Edge between self and a child :class:Node.

        This methods removes the Edge given, or the Edge connecting self and
        the Node given, from the annotation of :class:Passage. It does not
        remove the target or originating Node from the graph but just unlinks
        them.

        :param edge_or_node: either an Edge or Node object to remove/unlink

        :raise MissingNodeError: if the Node or Edge is not connected with self.

        """
        if edge_or_node not in self._outgoing:  # a Node, or an error
            try:
                edge = [edge for edge in self._outgoing
                        if edge.child == edge_or_node][0]
            except IndexError:
                raise MissingNodeError()
        else:  # an Edge object
            edge = edge_or_node

        try:
            self._outgoing.remove(edge)
            edge.child._incoming.remove(edge)
            self.root._remove_edge(edge)
        except ValueError:
            raise MissingNodeError()

    @property
    def orderkey(self):
        return self._orderkey

    @orderkey.setter
    def orderkey(self, value):
        self._orderkey = value
        self._outgoing.sort(key=value)

    @ModifyPassage
    def destroy(self):
        """Removes the :class:Node from the :class:Passage annotation graph.

        This method unlinks self from all other :class:Node objects and removes
        self from the :class:Layer and Passage objects.

        """
        # using outgoing and incoming so I won't change the list I'm working on
        for edge in self.outgoing:
            self.remove(edge)
        for edge in self.incoming:
            edge.parent.remove(edge)
        self.layer._remove_node(self)
        self._root._remove_node(self)

    def equals(self, other, *, recursive=True, ordered=False,
               ignore_node=None, ignore_edge=None):
        """Returns whether the self Node-equals other.

        Node-equality is basically determined by self and other having the same
        tag and attributes. Recursive equality is achieved when all outgoing
        Edges are Edge-equal, and their children are recursively Node-equal
        as well. Ordered equality means that the outgoing Edges should be
        equivalent to each other w.r.t order (the first to the first etc.),
        while unordered equality means that each Edges are equivalent after
        being ordered with some determined order.

        :param other: the Node object to compare to
        :param recursive: whether comparison is recursive, defaults to True.
        :param ordered: whether comparison should include strict ordering
        :param ignore_node: function that returns whether to ignore a given node
        :param ignore_edge: function that returns whether to ignore a given edge

        :return True iff the Nodes are equal in the terms given.

        """
        if self.tag != other.tag or not self._attrib.equals(other._attrib):
            return False
        if not recursive:
            return True
        edges, other_edges = [[edge for edge in node
                               if (ignore_node is None or
                                   not ignore_node(edge.child)) and (
                                   ignore_edge is None or
                                   not ignore_edge(edge))]
                              for node in (self, other)]
        if len(edges) != len(other_edges):
            return False  # not necessary, but gives better performance
        if ordered:
            return all(e1.equals(e2, ordered=True,
                                 ignore_node=ignore_node, ignore_edge=ignore_edge)
                       for e1, e2 in zip(edges, other_edges))
        # For unordered equality, I try to find & remove an equivalent
        # Edge + Node couple from other's Edges until exhausted.
        # Because both Edge-equality and Node-equality are equivalence
        # classes, I can just take the first I found and remove it w/o
        # trying to iterate through possible orders.
        try:
            for e1 in edges:
                other_edges.remove(next(e2 for e2 in other_edges
                                        if e1.equals(e2, ignore_node=ignore_node,
                                                     ignore_edge=ignore_edge)))
        except StopIteration:
            return False
        return not other_edges

    def missing_edges(self, other, ignore_node=None):
        """Returns edges present in this node but missing in the other.

        :param other: the Node object to compare to
        :param ignore_node: function that returns whether to ignore a given node

        :return List of edges present in this node but missing in the other.

        """
        edges, other_edges = [[edge for edge in node
                               if ignore_node is None or
                               not ignore_node(edge.child)]
                              for node in (self, other)]
        return sorted([e1 for e1 in edges if not any(e1.equals(e2) for e2 in other_edges)],
                      key=edge_id_orderkey)

    def iter(self, obj="nodes", method="dfs", duplicates=False, key=None):
        """Iterates the :class:Node objects in the subtree of self.

        :param obj: yield Node objects (use value "nodes", default) or Edge
                objects (use values "edges")
            method: do breadth-first iteration (use value "bfs") or depth-first
                iteration (value "dfs", default).
            duplicates: If True, may return the same object twice if it is
                encountered twice, because of the DAG structure which isn't
                necessarily a tree. If it is False, all objects will be yielded
                only the first time they are encountered. Defaults to False.
            key: boolean function that filters the iterable items. key function
                takes one argument (the item) and returns True if it should be
                returned to the user. If an item isn't returned, its subtree
                is still iterated.  Defaults to None (returns all items).

        Yields:
            a :class:Node or :class:Edge object according to the iteration
            parameters.

        """
        if method not in ("dfs", "bfs"):
            raise ValueError("method can be either 'dfs' or 'bfs'")
        if obj not in ("nodes", "edges"):
            raise ValueError("obj can be either 'nodes' or 'edges'")
        processed = set()
        if obj == 'nodes':
            waiting = [self]
        else:
            waiting = self._outgoing[:]
        while len(waiting):
            curr = waiting.pop(0)
            if key is None or key(curr):
                yield curr
            processed.add(curr)
            to_add = curr.children if obj == 'nodes' else list(curr.child)
            to_add = [x for x in to_add if duplicates or x not in processed]
            if method == "bfs":
                waiting.extend(to_add)
            else:
                waiting = to_add + waiting


class Layer:
    """Group of similar :class:Node objects in UCCA annotation graph.

    A Layer in UCCA annotation graph is a subgraph of the whole :class:Passage
    annotation graph which consists of similar Nodes and :class:Edge objects
    between them. The Nodes and the Layer itself has some formal definition for
    being grouped together.

    Attributes:
        ID: ID of the Layer, must be alphanumeric.
        root: the Passage this object is linked with
        attrib: attribute dictionary of the Layer
        extra: temporary storage space for undocumented attributes and data
        orderkey: the key function for ordering the Nodes in the layer.
            Note that it must rely only on the Nodes and/or Edges in the Layer.
            If it, for example, rely on Edges added between Nodes in the Layer
            and Nodes outside the Layer (hence, the Edges are not in the Layer)
            the order will not be updated (because the Layer object won't know
            that something has changed).
        all: a list of all the Nodes which are part of this Layer
        heads: a list of all Nodes which have no incoming Edges in the subgraph
            of the Layer (can have Edges from Nodes in other Layers).

    """

    def __init__(self, ID, root, attrib=None, *, orderkey=id_orderkey):
        """Creates a new :class:Layer object.

        :param see :class:Layer documentation.

        :raise FrozenPassageError: if the :class:Passage object we are part of
                is frozen and can't be modified.

        """
        if root.frozen:
            raise FrozenPassageError()
        self._ID = ID
        self._root = root
        self._attrib = _AttributeDict(root, attrib)
        self.extra = {}
        self._all = []
        self._heads = []
        self._orderkey = orderkey
        root._add_layer(self)

    @property
    def ID(self):
        return self._ID

    @property
    def root(self):
        return self._root

    @property
    def attrib(self):
        return self._attrib

    @property
    def all(self):
        return self._all[:]

    @property
    def heads(self):
        return self._heads[:]

    @property
    def orderkey(self):
        return self._orderkey

    @orderkey.setter
    def orderkey(self, value):
        self._orderkey = value
        self._all.sort(key=value)
        self._heads.sort(key=value)

    def equals(self, other, *, ordered=False, ignore_node=None, ignore_edge=None):
        """Returns whether two Layer objects are equal.

        Layers are considered Layer-equal if their attribute dictionaries are
        equal and all their heads are recursively Node-equal.
        Ordered Layer-equality implies that the heads should be
        ordered the same for the Layers to be considered equal, and the
        Node-equality is ordered too.

        :param other: the Layer object to compare to
        :param ordered: whether strict-order equality is used, defaults to False
        :param ignore_node: function that returns whether to ignore a given node
        :param ignore_edge: function that returns whether to ignore a given edge

        :return True iff self and other are Layer-equal.

        """
        if not self._attrib.equals(other._attrib):
            return False
        heads, other_heads = [[head for head in layer.heads
                               if ignore_node is None or
                               not ignore_node(head)]
                              for layer in (self, other)]
        if len(heads) != len(other_heads):
            return False  # can be removed, here for performance gain
        if ordered:
            return all(x1.equals(x2, ordered=True,
                                 ignore_node=ignore_node, ignore_edge=ignore_edge)
                       for x1, x2 in zip(heads, other_heads))
        # I can just find the first equal head in unordered search, as
        # Node-equality is an equivalence class (see their for details).
        try:
            for h1 in heads:
                other_heads.remove(next(h2 for h2 in other_heads
                                        if h1.equals(h2, ignore_node=ignore_node,
                                                     ignore_edge=ignore_edge)))
        except StopIteration:
            return False
        return not other_heads

    def _add_edge(self, edge):
        """Alters self.heads if an :class:Edge has been added to the subgraph.

        Should be called when both :class:Node objects of the edge are part
        of this Layer (and hence part of the subgraph of it).

        :param edge: the Edge added to the Layer subgraph

        """
        if edge.child in self._heads:
            self._heads.remove(edge.child)
        # Order may depend on edges, so re-order
        self._all.sort(key=self._orderkey)
        self._heads.sort(key=self._orderkey)

    def _remove_edge(self, edge):
        """Alters self.heads if an :class:Edge has been removed.

        Should be called when the child :class:Node object of the edge is part
        of this Layer (and hence part of the subgraph of it).

        :param edge: the Edge removed from the Layer subgraph

        """
        if edge.child.layer == self and all(p.layer != self for p in edge.child.parents):
            self._heads.append(edge.child)
            self._heads.sort(key=self._orderkey)
        # Order may depend on edges, so re-order
        self._all.sort(key=self._orderkey)
        self._heads.sort(key=self._orderkey)

    def _add_node(self, node):
        """Adds a :class:node to the :class:Layer.

        Assumes node has no incoming or outgoing :class:Edge objects.

        """
        self._all.append(node)
        self._all.sort(key=self._orderkey)
        self._heads.append(node)
        self._heads.sort(key=self._orderkey)

    def _remove_node(self, node):
        """Removes a :class:node from the :class:Layer.

        Assumes node has no incoming or outgoing :class:Edge objects.

        """
        self._all.remove(node)
        self._heads.remove(node)

    def _change_edge_tag(self, edge, old_tag):
        """Updates the :class:Layer objects with the change.

        :param edge: the updated :class:Edge object
            old_tag: the Edge's tag before the change

        """
        pass  # meant to be overriden by subclasses

    def _change_node_tag(self, node, old_tag):
        """Updates the :class:Layer objects with the change.

        :param node: the updated :class:Node object
            old_tag: the Node's tag before the change

        """
        pass  # meant to be overriden by subclasses


class Passage:
    """An annotated text with UCCA annotation graph.

    A Passage is an object representing a text annotated with UCCA annotation.
    UCCA annotation is a directed acyclic graph of :class:Node and :class:Edge
    objects grouped into :class:Layer objects.

    Attributes:
        ID: ID of the Passage
        root: simply self, for API similarity with other UCCA objects
        attrib: attribute dictionary of the Passage
        extra: temporary storage space for undocumented attributes and data
        layers: all Layers of the Passage, no order guaranteed
        nodes: dictionary of ID-node pairs for all the nodes in the Passage
        frozen: indicates whether the Passage can be modified or not, boolean.

    """

    def __init__(self, ID, attrib=None):
        """Creates a new :class:Passage object.

        :param see :class:Passage documentation.

        """
        self._ID = ID
        self._attrib = _AttributeDict(self, attrib)
        self.extra = {}
        self._layers = {}
        self._nodes = {}
        self.frozen = False

    @property
    def ID(self):
        return self._ID

    @property
    def root(self):
        return self

    @property
    def attrib(self):
        return self._attrib

    @property
    def layers(self):
        return self._layers.values()

    @property
    def nodes(self):
        return self._nodes.copy()

    def layer(self, ID):
        """Returns the :class:Layer object whose ID is given.

        :param ID: ID of the Layer requested.

        :raise KeyError: if no Layer with this ID is present

        """
        return self._layers[ID]

    def equals(self, other, *, ordered=False, ignore_node=None, ignore_edge=None):
        """Returns whether two passages are equivalent.

        Passage-equivalence is determined by having the same attributes and
        all layers (according to ID) are Layer-equivalent.

        :param other: the Passage object to compare to
        :param ordered: is Layer-equivalency should be ordered (see there)
        :param ignore_node: function that returns whether to ignore a given node
        :param ignore_edge: function that returns whether to ignore a given edge

        :return True iff self is Passage-equivalent to other.

        """
        if not self._attrib.equals(other._attrib):
            return False
        # noinspection PyTypeChecker
        if len(self.layers) != len(other.layers):
            return False  # can be removed, here for performance gain
        try:
            for lid, l1 in self._layers.items():
                l2 = other.layer(lid)
                if not l1.equals(l2, ordered=ordered,
                                 ignore_node=ignore_node, ignore_edge=ignore_edge):
                    return False
        except KeyError:  # no layer with same ID found
            return False
        return True

    def missing_nodes(self, other, ignore_node=None, ignore_edge=None):
        """Returns nodes present in this passage but missing in the other.

        :param other: the Passage object to compare to
        :param ignore_node: function that returns whether to ignore a given node
        :param ignore_edge: function that returns whether to ignore a given edge

        :return List of nodes present in this passage but missing in the other.

        """
        nodes, other_nodes = [[node for node in passage.nodes.values()
                               if ignore_node is None or
                               not ignore_node(node)]
                              for passage in (self, other)]
        return sorted([n1 for n1 in nodes
                       if not any(n1.equals(n2, ignore_node=ignore_node,
                                            ignore_edge=ignore_edge)
                                  for n2 in other_nodes)],
                      key=id_orderkey)

    def copy(self, layers):
        """Copies the Passage and specified layers to a new object.

        The main "building block" of copying is the Layer, so copying is
        truly copying the Passage attributes (attrib, extra, ID, frozen)
        and creating the equivalent layers (each layer for itself).

        :param layers: sequence of layer IDs to copy to the new object.

        :return A new Passage object.

        :raise KeyError if a given layer ID doesn't exist.
            UnimplementedMethodError if copying for a layer is unimplemented.

        """
        other = Passage(ID=self.ID, attrib=self.attrib.copy())
        other.extra = self.extra.copy()
        for lid in layers:
            try:
                self.layer(lid).copy(other)
            except AttributeError:
                raise UnimplementedMethodError()
        other.frozen = self.frozen
        return other

    def by_id(self, ID):
        """Returns a Node whose ID is given.

        :param ID: ID string

        :return The node.Node object whose ID matches

        :raise KeyError if no Node with this ID is found

        """
        return self._nodes[ID]

    @ModifyPassage
    def _add_layer(self, layer):
        """Adds a :class:Layer object to the :class:Passage.

        :param layer: the Layer object to add

        :raise DuplicateIdError: if layer.ID is identical to a Layer already
                present in the Passage.
            FrozenPassageError: if the :class:Passage object we are part of
                is frozen and can't be modified.

        """
        if layer.ID in self._layers:
            raise DuplicateIdError()
        self._layers[layer.ID] = layer

    @ModifyPassage
    def _add_node(self, node):
        """Adds a :class:Node object to the :class:Passage.

        :param node: the Node object to add

        :raise DuplicateIdError: if node.ID is identical to a Node already
                present in the Passage.
            FrozenPassageError: if the :class:Passage object we are part of
                is frozen and can't be modified.

        """
        if node.ID in self._nodes:
            raise DuplicateIdError()
        self._nodes[node.ID] = node

    def _remove_node(self, node):
        """Removes a :class:Node object from the :class:Passage.

        :param node: the Node object to remove, must be unlinked with any other
                Node objects and removed from its :class:Layer.

        :raise KeyError: if no Node with this ID is present

        """
        del self._nodes[node.ID]

    @ModifyPassage
    def _add_edge(self, edge):
        """Adds a :class:Edge object to :class:Passage.

        Handles altering the Passage and :class:Layer objects accordingly.

        :param edge: the Edge object to add

        """
        # Currently no work is done in the Passage level
        edge.parent.layer._add_edge(edge)

    def _remove_edge(self, edge):
        """Removes a :class:Edge object from :class:Passage.

        Handles altering the Passage and :class:Layer objects accordingly.

        :param edge: the Edge object to remove

        """
        # Currently no work is done in the Passage level
        edge.parent.layer._remove_edge(edge)

    def _change_edge_tag(self, edge, old_tag):
        """Updates the :class:Passage and :class:Layer objects with the change.

        :param edge: the updated :class:Edge object
            old_tag: the Edge's tag before the change

        """
        # Currently no work is done in the Passage level
        edge.parent.layer._change_edge_tag(edge, old_tag)

    def _change_node_tag(self, node, old_tag):
        """Updates the :class:Passage and :class:Layer objects with the change.

        :param node: the updated :class:Node object
            old_tag: the Node's tag before the change

        """
        # Currently no work is done in the Passage level
        node.layer._change_node_tag(node, old_tag)
