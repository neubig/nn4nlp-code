import operator
import warnings
from collections import defaultdict

import matplotlib.cbook
import networkx as nx

from ucca import layer0, layer1
from ucca.layer1 import Linkage

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=UserWarning)


def draw(passage):
    G = nx.DiGraph()
    terminals = sorted(passage.layer(layer0.LAYER_ID).all, key=operator.attrgetter("position"))
    G.add_nodes_from([(n.ID, {"label": n.text, "node_color": "white"}) for n in terminals])
    G.add_nodes_from([(n.ID, {"label": "IMPLICIT" if n.attrib.get("implicit") else "",
                              "node_color": "gray" if isinstance(n, Linkage) else (
                                  "white" if n.attrib.get("implicit") else "black")})
                      for n in passage.layer(layer1.LAYER_ID).all])
    G.add_edges_from([(n.ID, e.child.ID, {"label": e.tag, "style": "dashed" if e.attrib.get("remote") else "solid"})
                      for layer in passage.layers for n in layer.all for e in n])
    pos = topological_layout(passage)
    nx.draw(G, pos, arrows=False, font_size=10,
            node_color=[d["node_color"] for _, d in G.nodes(data=True)],
            labels={n: d["label"] for n, d in G.nodes(data=True) if d["label"]},
            style=[d["style"] for _, _, d in G.edges(data=True)])
    nx.draw_networkx_edge_labels(G, pos, font_size=8,
                                 edge_labels={(u, v): d["label"] for u, v, d in G.edges(data=True)})


def topological_layout(passage):
    visited = defaultdict(set)
    pos = {}
    implicit_offset = 1 + max((n.position for n in passage.layer(layer0.LAYER_ID).all), default=-1)
    remaining = [n for layer in passage.layers for n in layer.all if not n.parents]
    while remaining:
        node = remaining.pop()
        if node.ID in pos:  # done already
            continue
        if node.children:
            children = [c for c in node.children if c.ID not in pos and c not in visited[node.ID]]
            if children:
                visited[node.ID].update(children)  # to avoid cycles
                remaining += [node] + children
                continue
            xs, ys = zip(*(pos[c.ID] for c in node.children))
            pos[node.ID] = (sum(xs) / len(xs), 1 + max(ys))  # done with children
        elif node.layer.ID == layer0.LAYER_ID:  # terminal
            pos[node.ID] = (int(node.position), 0)
        else:  # implicit
            pos[node.ID] = (implicit_offset, 0)
            implicit_offset += 1
    return pos
