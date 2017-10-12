COMPOUND = "compound"

class Labels(object):
    def __init__(self, size):
        self.size = size  # Maximum number of labels, NOT enforced here but by the user

    @property
    def all(self):
        raise NotImplementedError()

    @all.setter
    def all(self, labels):
        raise NotImplementedError()

    def save(self, skip=False):
        return (None if skip else self.all), self.size

    def load(self, all_size):
        self.all, self.size = all_size


class Action(dict):
    type_to_id = {}

    def __init__(self, action_type, tag=None, orig_edge=None, orig_node=None, oracle=None, id_=None):
        self.type = action_type  # String
        self.tag = tag  # Usually the tag of the created edge; but if COMPOUND_SWAP, the distance
        self.orig_node = orig_node  # Node created by this action, if any (during training)
        self.orig_edge = orig_edge  # Edge created by this action, if any (during training)
        self.node = None  # Will be set by State when the node created by this action is known
        self.edge = None  # Will be set by State when the edge created by this action is known
        self.oracle = oracle  # Reference to oracle, to inform it of actually created nodes/edges
        self.index = None  # Index of this action in history

        self.type_id = Action.type_to_id.get(self.type)  # Allocate ID for fast comparison
        if self.type_id is None:
            self.type_id = len(Action.type_to_id)
            Action.type_to_id[self.type] = self.type_id
        self.id = id_
        super().__init__(action_type=self.type, tag=self.tag)

    def is_type(self, *others):
        return self.type_id in (o.type_id for o in others)

    def apply(self):
        if self.oracle is not None:
            self.oracle.remove(self.orig_edge, self.orig_node)

    def __repr__(self):
        return Action.__name__ + "(" + ", ".join(map(str, filter(None, (self.type, self.tag)))) + ")"

    def __str__(self):
        s = self.type
        if self.tag:
            s += "-%s" % self.tag
        return s

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __call__(self, *args, **kwargs):
        return Action(self.type, *args, **kwargs)

    @property
    def remote(self):
        return self.is_type(Actions.RemoteNode, Actions.LeftRemote, Actions.RightRemote)

    @property
    def is_swap(self):
        return self.is_type(Actions.Swap)


class Actions(Labels):
    Shift = Action("SHIFT")
    Node = Action("NODE")
    RemoteNode = Action("REMOTE-NODE")
    Implicit = Action("IMPLICIT")
    Label = Action("LABEL")
    Reduce = Action("REDUCE")
    LeftEdge = Action("LEFT-EDGE")
    RightEdge = Action("RIGHT-EDGE")
    LeftRemote = Action("LEFT-REMOTE")
    RightRemote = Action("RIGHT-REMOTE")
    Swap = Action("SWAP")
    Finish = Action("FINISH")

    def __init__(self, actions=None, size=None):
        super().__init__(size=size)
        self._all = None
        self._ids = None
        if actions is not None:
            self.all = actions

    def init(self):
        # edge and node action will be created as they are returned by the oracle
        swap = 'regular'
        self.all = [Actions.Reduce, Actions.Shift, Actions.Finish] + \
            (list(map(Actions.Swap, range(1, 3))) if swap == COMPOUND else
             [Actions.Swap] if swap else []) + \
            ([Actions.Label] if False else [])

    @property
    def all(self):
        if self._all is None:
            self.init()
        return self._all

    @all.setter
    def all(self, actions):
        self._all = [Action(**a) if isinstance(a, dict) else a for a in actions]
        self._ids = {(action.type_id, action.tag): i for i, action in enumerate(self._all)}
        for action in self._all:
            self.generate_id(action)

    @property
    def ids(self):
        if self._all is None:
            self.init()
        return self._ids

    def generate_id(self, action, create=True):
        if action.id is None:
            key = (action.type_id, action.tag)
            action.id = self.ids.get(key)
            if create and action.id is None:  # New action, add to list
                # noinspection PyTypeChecker
                action.id = len(self.all)
                self.all.append(action(tag=action.tag, id_=action.id))
                self.ids[key] = action.id
