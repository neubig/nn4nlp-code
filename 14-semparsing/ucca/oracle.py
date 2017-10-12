
from ucca import layer1

from actions import Actions

LABEL_ATTRIB = "label"
COMPOUND = "compound"


class InvalidActionError(AssertionError):
    def __init__(self, *args, is_type=False):
        super().__init__(*args)
        self.is_type = is_type



#modified from: https://github.com/danielhers/tupa/blob/master/tupa/parse.py
# Constants for readability, used by Oracle.action
RIGHT = PARENT = NODE = 0
LEFT = CHILD = EDGE = 1
ACTIONS = (  # index by [NODE/EDGE][PARENT/CHILD or RIGHT/LEFT][True/False (remote)]
    (  # node actions
        (Actions.Node, Actions.RemoteNode),  # creating a parent
        (Actions.Implicit, None)  # creating a child (remote implicit is not allowed)
    ),
    (  # edge actions
        (Actions.RightEdge, Actions.RightRemote),  # creating a right edge
        (Actions.LeftEdge, Actions.LeftRemote)  # creating a left edge
    )
)


class Oracle(object):
    """
    Oracle to produce gold transition parses given UCCA passages
    To be used for creating training data for a transition-based UCCA parser
    :param passage gold passage to get the correct edges from
    """
    def __init__(self, passage):
        linkage = False
        implicit = False
        remote = True
        l1 = passage.layer(layer1.LAYER_ID)
        self.nodes_remaining = {node.ID for node in l1.all
                                if node is not l1.heads[0] and
                                (linkage or node.tag != layer1.NodeTags.Linkage) and
                                (implicit or not node.attrib.get("implicit"))}
        self.edges_remaining = {edge for node in passage.nodes.values() for edge in node
                                if (linkage or edge.tag not in (
                                    layer1.EdgeTags.LinkRelation, layer1.EdgeTags.LinkArgument)) and
                                (implicit or not edge.child.attrib.get("implicit")) and
                                (remote or not edge.attrib.get("remote"))}
        self.passage = passage
        self.found = False
        self.log = None

    def get_actions(self, state, all_actions, create=True):
        """
        Determine all zero-cost action according to current state
        Asserts that the returned action is valid before returning
        :param state: current State of the parser
        :param all_actions: Actions object used to map actions to IDs
        :param create: whether to create new actions if they do not exist yet
        :return: dict of action ID to Action
        """
        actions = {}
        invalid = []
        for action in self.generate_actions(state):
            all_actions.generate_id(action, create=create)
            if action.id is not None:
                try:
                    state.check_valid_action(action, message=True)
                    actions[action.id] = action
                except InvalidActionError as e:
                    invalid.append((action, e))
        assert actions, self.generate_log(invalid, state)
        return actions

    def generate_log(self, invalid, state):
        self.log = "\n".join(["Oracle found no valid action",
                              state.str("\n"), self.str("\n"),
                              "Actions returned by the oracle:"] +
                             ["  %s: %s" % (action, e) for (action, e) in invalid] or ["None"])
        return self.log

    def generate_actions(self, state):
        """
        Determine all zero-cost action according to current state
        :param state: current State of the parser
        :return: generator of Action items to perform
        """
        if not ((state.buffer or state.stack) and (self.edges_remaining or
                                                   any(map(self.need_label, state.stack + list(state.buffer))))):
            yield Actions.Finish
            if state.stack and not self.need_label(state.stack[-1]):
                yield Actions.Reduce
            return

        self.found = False
        if state.stack:
            s0 = state.stack[-1]
            incoming = self.edges_remaining.intersection(s0.orig_node.incoming)
            outgoing = self.edges_remaining.intersection(s0.orig_node.outgoing)
            if not incoming and not outgoing and not self.need_label(s0):
                yield Actions.Reduce
                return
            else:
                # Check for node label action: if all terminals have already been connected
                if self.need_label(s0) and not any(e.tag == layer1.EdgeTags.Terminal for e in outgoing):
                    self.found = True
                    yield Actions.Label(0, orig_node=s0.orig_node, oracle=self)

                # Check for actions to create new nodes
                for edge in incoming:
                    if edge.parent.ID in self.nodes_remaining and not edge.parent.attrib.get("implicit") and (
                                not edge.attrib.get("remote") or
                                # Allow remote parent if all its children are remote/implicit
                                all(e.attrib.get("remote") or e.child.attrib.get("implicit") for e in edge.parent)):
                        yield self.action(edge, NODE, PARENT)  # Node or RemoteNode

                for edge in outgoing:
                    if edge.child.ID in self.nodes_remaining and edge.child.attrib.get("implicit") and (
                            not edge.attrib.get("remote")):  # Allow implicit child if it is not remote
                        yield self.action(edge, NODE, CHILD)  # Implicit

                if len(state.stack) > 1:
                    s1 = state.stack[-2]
                    # Check for node label action: if all terminals have already been connected
                    if self.need_label(s1) and not any(e.tag == layer1.EdgeTags.Terminal for e in
                                                       self.edges_remaining.intersection(s1.orig_node.outgoing)):
                        self.found = True
                        yield Actions.Label(1, orig_node=s1.orig_node, oracle=self)

                    # Check for actions to create binary edges
                    for edge in incoming:
                        if edge.parent.ID == s1.node_id:
                            yield self.action(edge, EDGE, RIGHT)  # RightEdge or RightRemote

                    for edge in outgoing:
                        if edge.child.ID == s1.node_id:
                            yield self.action(edge, EDGE, LEFT)  # LeftEdge or LeftRemote
                        elif state.buffer and edge.child.ID == state.buffer[0].node_id and \
                                len(state.buffer[0].orig_node.incoming) == 1:
                            yield Actions.Shift  # Special case to allow getting rid of simple children quickly

                    if not self.found:
                        # Check if a swap is necessary, and how far (if compound swap is enabled)
                        related = dict([(edge.child.ID,  edge) for edge in outgoing] +
                                       [(edge.parent.ID, edge) for edge in incoming])
                        distance = None  # Swap distance (how many nodes in the stack to swap)
                        for i, s in enumerate(state.stack[-3::-1], start=1):  # Skip top two: checked above, not related
                            edge = related.pop(s.node_id, None)
                            if edge is not None:
                                swap = 'regular'
                                max_swap = 3
                                if not swap == COMPOUND:  # We have no chance to reach it, so stop trying
                                    self.remove(edge)
                                    continue
                                if distance is None and swap == COMPOUND:  # Save the first one
                                    distance = min(i, max_swap)  # Do not swap more than allowed
                                if not related:  # All related nodes are in the stack
                                    yield Actions.Swap(distance)
                                    return

        if state.buffer and not self.found:
            yield Actions.Shift

    def action(self, edge, kind, direction):
        self.found = True
        remote = edge.attrib.get("remote", False)
        node = (edge.parent, edge.child)[direction] if kind == NODE else None
        return ACTIONS[kind][direction][remote](tag=edge.tag, orig_edge=edge, orig_node=node, oracle=self)

    def remove(self, edge, node=None):
        self.edges_remaining.discard(edge)
        if node is not None:
            self.nodes_remaining.discard(node.ID)

    def need_label(self, node):
        node_labels = False
        return node_labels and not node.labeled and node.orig_node.attrib.get(LABEL_ATTRIB)

    def str(self, sep):
        return "nodes left: [%s]%sedges left: [%s]" % (" ".join(self.nodes_remaining), sep,
                                                       " ".join(map(str, self.edges_remaining)))

    def __str__(self):
        return str(" ")
