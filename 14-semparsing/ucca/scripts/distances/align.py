import inspect
import re
from collections import Counter

import distance
import numpy as np
import zss
from munkres import Munkres

import evaluation
from textutil import break2sentences, extract_terminals
from ucca import layer0, layer1


def compare(n1, n2):
    return bool(labels(n1) & labels(n2))


def labels(n):
    return evaluation.expand_equivalents(set(e.tag for e in n.incoming))


def label(n):
    return n.ftag


def is_terminal(n):
    """returns true if the node contains only one terminal or less"""
    return len(n.get_terminals()) <= 1


def is_foundational(node):
    return node.tag == layer1.NodeTags.Foundational


def is_passage(n):
    return not bool(n.fparent)


def is_comparable(n):
    """checks if the node is a node that should be compared between passages"""
    return is_foundational(n) and (not is_terminal(n)) and (not is_passage(n))


def top_from_passage(p):
    """returns the top elements from a passage"""
    l = p.layer(layer1.LAYER_ID)
    return l.top_scenes + l.top_linkages


def preprocess_word(word):
    """standardize word form for the alignment task"""
    return word.strip().lower()


def align(sen1, sen2, string=True):
    """finds the best mapping of words from one sentence to the other
    string = a boolean represents if sentences are given as strings or as list of ucca terminal nodes
    returns list of word tuples and the corresponding list of indexes tuples"""
    if string:
        sen1 = list(map(preprocess_word, sen1.split()))
        sen2 = list(map(preprocess_word, sen2.split()))
    else:
        sen1 = [preprocess_word(terminal.text) for terminal in sen1]
        sen2 = [preprocess_word(terminal.text) for terminal in sen2]

    # find lengths
    length_dif = len(sen1) - len(sen2)
    if length_dif > 0:
        shorter = sen2
        longer = sen1
        switched = False
    else:
        shorter = sen1
        longer = sen2
        switched = True
        length_dif = abs(length_dif)
    shorter += ["emptyWord"] * length_dif

    # create matrix
    matrix = np.zeros((len(longer), len(longer)))
    for i in range(len(longer)):
        for j in range(len(longer) - length_dif):
            matrix[i, j] = distance.levenshtein(longer[i], shorter[j]) + float(abs(i - j)) / len(longer)

    # compare with munkres
    m = Munkres()
    indexes = m.compute(matrix)

    # remove indexing for emptywords and create string mapping
    refactored_indexes = []
    mapping = []
    start = 0 if string else 1
    for i, j in indexes:
        if j >= len(longer) - length_dif:
            j = -1 - start
        if switched:
            refactored_indexes.append((j + start, i + start))
            mapping.append((shorter[j], longer[i]))
        else:
            refactored_indexes.append((i + start, j + start))
            mapping.append((longer[i], shorter[j]))
    return mapping, refactored_indexes


def regularize_word(word):
    """changes structure of the word to the same form (e.g. lowercase)"""

    # remove non-alphanumeric
    pattern = re.compile("[\W_]+")
    word = pattern.sub("", word)
    return word


def _to_text(passage, position):
    """ returns the text og the position in the passage"""
    return passage.layer(layer0.LAYER_ID).by_position(position).text


def _choose_ending_position(passage, position):
    """chooses the correct ending index,
    position - an estimated sentence ending (e.g. by textutil.break2sentences)"""
    reg = regularize_word(_to_text(passage, position))

    # check current token is not deleted by regularization
    if not reg and position:
        position -= 1
        reg = regularize_word(_to_text(passage, position))
    return position, reg


def _count_mapping(positions1, positions2, word2word, from_key):
    """counts the number of satisfied mappings from positions[from_key] to the other positions"""
    from_key -= 1
    to_key = int(not from_key)
    positions_list = [positions1, positions2]

    terminal_positions1 = sorted(positions_list[from_key])
    terminal_positions2 = sorted(positions_list[to_key])
    sorted_word2word = sorted(word2word.copy(), key=lambda x: x[from_key])

    # count in one pass in parallel, they are sorted
    index = 0
    count = 0
    for mapping in sorted_word2word:
        frm = mapping[from_key]
        to = mapping[to_key]

        if index == len(terminal_positions1):
            break
        if terminal_positions1[index] == frm:
            index += 1
            if to in terminal_positions2:
                count += 1
    return count


def two_sided_f(count1, count2, sum1, sum2):
    """computes an F score like measure"""
    # check input
    if not (sum1 and sum2):
        print("got empty sums for F scores")
        return 0
    if sum1 < count1 or sum2 < count2:
        print("got empty sums for F scores")
        return 0

    # calculate
    precision = count2 / sum2
    recall = count1 / sum1
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def node_word2word_similarity(node1, node2, word2word, graceful=True):
    """ compute an F score for two nodes based on the word2word mapping"""
    if not (is_foundational(node1) and is_foundational(node2)):
        if not graceful:
            print("one of the requested nodes is not foundational")
        return 0
    terminal_positions1 = [term.para_pos for term in node1.get_terminals()]
    terminal_positions2 = [term.para_pos for term in node2.get_terminals()]

    # edge cases
    if not (len(terminal_positions1) and len(terminal_positions2)):
        if not graceful:
            print("error: no terminals in node")
            if not len(terminal_positions2):
                print(node2.ID)
            if not len(terminal_positions1):
                print(node1.ID)
        return 0

    count1 = _count_mapping(terminal_positions1, terminal_positions2, word2word, 1)
    count2 = _count_mapping(terminal_positions1, terminal_positions2, word2word, 2)

    return two_sided_f(count1, count2, len(terminal_positions1), len(terminal_positions2))


def get_lowest_fn(p):
    """ finds the FN that has terminals as children"""
    s = set()
    for term in extract_terminals(p):
        s.update([edge.parent for edge in term.incoming if is_foundational(edge.parent)])
    return s


def fully_align(p1, p2, word2word=None):
    """ aligns nodes from p1 to those of p2 by finding the best matches from all pairs of nodes"""
    if not word2word:
        word2word = align_yields(p1, p2)
    nodes1 = set(node for node in p1.layer(layer1.LAYER_ID).all if is_comparable(node))
    nodes2 = set(node for node in p2.layer(layer1.LAYER_ID).all if is_comparable(node))
    return align_nodes(nodes1, nodes2, word2word)


def top_down_align(p1, p2, word2word=None):
    """aligns nodes from p1 to those of p2 top down"""
    if not word2word:
        word2word = align_yields(p1, p2)
    new = align_nodes(top_from_passage(p1), top_from_passage(p2), word2word)
    remaining = dict(new)
    mapping = dict(new)

    while remaining:
        n1, n2 = remaining.popitem()
        new = align_nodes(n1.children, n2.children, word2word)
        remaining.update(new)
        mapping.update(new)
    return mapping


def buttom_up_by_levels_align(p1, p2, word2word=None):
    """ aligns all the nodes in two paragraphs going up from the terminals level by level"""
    if not word2word:
        word2word = align_yields(p1, p2)
    mapping = {}
    nodes1 = set(get_lowest_fn(p1))
    nodes2 = set(get_lowest_fn(p2))
    while nodes1 and nodes2:
        mapping.update((align_nodes(nodes1, nodes2, word2word)))
        nodes1 = set(node.fparent for node in nodes1 if node.fparent is not None)
        nodes2 = set(node.fparent for node in nodes2 if node.fparent is not None)
    return mapping


def buttom_up_paragraph_align(p1, p2, word2word=None):
    """ aligns all the nodes in two paragraphs going up from the terminals level by level"""
    if not word2word:
        word2word = align_yields(p1, p2)
    pairs1 = dict(p1.layer(layer0.LAYER_ID).pairs)
    pairs2 = dict(p2.layer(layer0.LAYER_ID).pairs)
    mapping = dict((pairs1[i], pairs2[j]) for i, j in word2word if i != -1 and j != -1)

    next_checking1 = next_checking2 = None
    checking1 = set(n for n in get_lowest_fn(p1) if is_comparable(n))
    checking2 = set(n for n in get_lowest_fn(p2) if is_comparable(n))
    save_next = True
    waiting = set()  # nodes that have unmapped children
    while checking1 and checking2:
        # save nodes from one level up for next look
        if save_next:
            next_checking1 = set(node.fparent for node in checking1 if node.fparent is not None)
            next_checking2 = set(node.fparent for node in checking2 if node.fparent is not None)
            save_next = False
        # look for best match for one of the checked node
        n1 = checking1.pop()
        best = 0
        is_waiting = False
        n2 = None
        for n in checking2:
            current = 0
            for child in n1.children:
                if is_comparable(child) or child.tag == layer0.NodeTags.Word:
                    if child not in mapping:
                        waiting.add(n1)
                        best = -1
                        is_waiting = True
                        # print(n1)
                        break
                    if mapping[child] in n.children:
                        current += 1
            if is_waiting:
                break
            if current > best or (current == best and compare(n1, n)):
                n2 = n
                best = current
        if n2 and not is_waiting:
            mapping[n1] = n2
            checking2.remove(n2)  # can only map one node to each node in p2
        # if the queue is empty, fill it up if able
        if not checking1 or not checking2:
            checking1 |= next_checking1 | waiting
            checking2 |= next_checking2
            checking1 = checking1.difference(mapping.keys())
            save_next = True
    return mapping


def align_nodes(nodes1, nodes2, word2word):
    """finds best matching from the set of nodes nodes1 to the set nodes2
        Note: this function is not symmetrical
        """
    best = {}
    mapping = {}
    gen1 = (node for node in nodes1 if is_foundational(node))
    gen2 = [node for node in nodes2 if is_foundational(node)]
    for node1 in gen1:
        for node2 in gen2:
            sim = node_word2word_similarity(node1, node2, word2word)
            if sim and (node1 not in best or sim > best[node1] or (sim == best[node1] and compare(node1, node2))):
                best[node1] = sim
                mapping[node1] = node2
                # if best match got, stop looking for it
                if best[node1] == 1 and compare(node1, node2):
                    break
    return mapping


def break2common_sentences(p1, p2):
    """finds the positions of the common sentence ending

    Breaking is done according to the text and to the ucca annotation of both passages
    returns two lists each containing positions of sentence endings
    guarentees same number of positions is acquired and the last position is the passage end"""
    # break to sentences
    broken1 = break2sentences(p1)
    broken2 = break2sentences(p2)

    # find common endings
    positions1 = []
    positions2 = []
    i = 0
    j = 0
    while j < len(broken2) and i < len(broken1):
        position1, reg1 = _choose_ending_position(p1, broken1[i])
        position2, reg2 = _choose_ending_position(p2, broken2[j])
        if i + 1 < len(broken1):
            pos_after1, one_after1 = _choose_ending_position(p1, broken1[i + 1])
        else:
            pos_after1, one_after1 = position1, reg1
        if j + 1 < len(broken2):
            pos_after2, one_after2 = _choose_ending_position(p2, broken2[j + 1])
        else:
            pos_after2, one_after2 = position2, reg2

        if reg1 == reg2:
            positions1.append(position1)
            positions2.append(position2)
        # deal with  addition or subtraction of a sentence ending
        elif one_after1 == reg2:
            i += 1
            positions1.append(pos_after1)
            positions2.append(position2)
        elif reg1 == one_after2:
            j += 1
            positions1.append(position1)
            positions2.append(pos_after2)
        i += 1
        j += 1

    # add last sentence in case skipped
    position1, reg1 = _choose_ending_position(p1, broken1[-1])
    position2, reg2 = _choose_ending_position(p2, broken2[-1])
    if (not positions1) or (not positions2) or (
                    positions1[-1] != position1 and positions2[-1] != position2):
        positions1.append(broken1[-1])
        positions2.append(broken2[-1])
    elif positions1[-1] != position1 and positions2[-1] == position2:
        positions1[-1] = position1
    elif positions1[-1] == position1 and positions2[-1] != position2:
        positions2[-1] = broken2[-1]
    return positions1, positions2


def reverse_mapping(word2word):
    """gets an iterator of tuples and returns a set of the reveresed mapping"""
    return set((j, i) for (i, j) in word2word)


def align_yields(p1, p2):
    """finds the best alignment of words from two passages
    Note: this function is symetrical
    consider using reverse_mapping instead of calling it twice

    returns iterator of tuples (i,j)
            mapping from i - p1 positions 
                    to j - aligned p2 positions"""
    positions1, positions2 = break2common_sentences(p1, p2)
    terminals1 = extract_terminals(p1)
    terminals2 = extract_terminals(p2)

    # map the words in each sentence to each other
    if len(positions1) == len(positions2):
        mapping = set()
        sentence_start1 = 0
        sentence_start2 = 0
        for i in range(len(positions1)):
            sentence1 = terminals1[sentence_start1:positions1[i]]
            sentence2 = terminals2[sentence_start2:positions2[i]]
            for (j, k) in align(sentence1, sentence2, False)[1]:
                if j != -1:
                    j += sentence_start1
                if k != -1:
                    k += sentence_start2
                mapping.add((j, k))

            sentence_start1 = positions1[i]
            sentence_start2 = positions2[i]
        return mapping
    else:
        print("Error number of sentences aqquired from break2common_sentences dow not match")


def fully_aligned_distance(p1, p2):
    """compares each one to its' best mapping"""
    word2word = align_yields(p1, p2)
    nodes1 = set(node for node in p1.layer(layer1.LAYER_ID).all if is_comparable(node))
    nodes2 = set(node for node in p2.layer(layer1.LAYER_ID).all if is_comparable(node))
    first = align_nodes(nodes1, nodes2, word2word)
    word2word = reverse_mapping(word2word)
    second = align_nodes(nodes2, nodes1, word2word)

    count1 = len(set((i, j) for (i, j) in first.items() if compare(i, j)))
    count2 = len(set((i, j) for (i, j) in second.items() if compare(i, j)))
    print(inspect.currentframe().f_code.co_name, " returns ", two_sided_f(count1, count2, len(nodes1), len(nodes2)))
    return two_sided_f(count1, count2, len(nodes1), len(nodes2))


MAIN_RELATIONS = [layer1.EdgeTags.ParallelScene,
                  layer1.EdgeTags.Participant,
                  layer1.EdgeTags.Adverbial
                  ]


def token_matches(p1, p2, map_by):
    """returns the number of matched tokens from p1 with tag from MAIN_RELATIONS
        p1,p2 passages
        map_by a function that maps all nodes from p1 to p2 nodes
        Note: this function is noy simmetrical"""
    count = 0
    mapping = map_by(p1, p2)
    print("mapping length", len(mapping))
    for node1, node2 in mapping.items():
        if is_comparable(node1) and is_comparable(node2) and label(node1) in MAIN_RELATIONS and compare(node1, node2):
            count += 1
    return count


def token_distance(p1, p2, map_by=buttom_up_by_levels_align):
    """compares considering only the main relation of each node"""
    count1 = token_matches(p1, p2, map_by)
    count2 = token_matches(p2, p1, map_by)
    nodes1 = set(node for node in p1.layer(layer1.LAYER_ID).all
                 if is_comparable(node) and label(node) in MAIN_RELATIONS)
    nodes2 = set(node for node in p2.layer(layer1.LAYER_ID).all
                 if is_comparable(node) and label(node) in MAIN_RELATIONS)
    print(inspect.currentframe().f_code.co_name)
    print("counts", count1, count2)
    print("lens", len(nodes1), len(nodes2))
    print(two_sided_f(count1, count2, len(nodes1), len(nodes2)))
    return two_sided_f(count1, count2, len(nodes1), len(nodes2))


def tree_structure(n):
    """ gets a node and returns a tree structure from it"""
    childrn = []
    for child in n.children:
        child = tree_structure(child)
        if child:
            childrn.append(child)
    return n, childrn


def tree_structure_aligned(n1, n2, word2word):
    """ gets two nodes and returns a tree structure from each with the proper mapping"""
    tree1 = []
    tree2 = []
    mapping = align_nodes(n1.children, n2.children, word2word)

    # add matching
    for s in mapping:
        if s in mapping and mapping[s] not in tree2:
            tree1.append(s)
            tree2.append(mapping[s])

    # add not matching
    for s in n1.children:
        if s not in tree1:
            tree1.append(s)
    for s in n2.children:
        if s not in tree2:
            tree2.append(s)

    # convert recursivly
    longer = max(len(n1.children), len(n2.children))
    shorter = min(len(n1.children), len(n2.children))
    res1 = []
    res2 = []
    for i in range(shorter):
        t1, t2 = tree_structure_aligned(tree1[i], tree2[i], word2word)
        res1.append(t1)
        res2.append(t2)
    if len(n1.children) == longer:
        res1 += [tree_structure(n) for n in n1.children[shorter:]]
    if len(n2.children) == longer:
        res2 += [tree_structure(n) for n in n2.children[shorter:]]
    return (n1, res1), (n2, res2)


def convert_structure_to_zss(tree):
    lbl = label(tree[0]) if hasattr(tree[0], "ftag") else str(type(tree[0]))
    return zss.Node(lbl, [convert_structure_to_zss(n) for n in tree[1]])


def prune_leaves(tree, flter=lambda x: True):
    """ takes a tree structure and prunes the leaves, for a single nodes without leaves returns t
        tree
        filter - a boolean function that gets a node and decides whether it should be left out,
                 this filters only leaves that this function returns true for"""
    if tree[1]:
        res = []
        for t in tree[1]:
            if t[1]:
                pruned = prune_leaves(t)
                if pruned or not flter(tree[0]):
                    res.append(pruned)
        return tree[0], [n for n in res if n is not None]
    return


def create_ordered_trees(p1, p2, word2word=None):
    """ creates two trees from two passages"""
    if not word2word:
        word2word = align_yields(p1, p2)
    top1 = top_from_passage(p1)
    top2 = top_from_passage(p2)
    mapping = align_nodes(top1, top2, word2word)
    tree1 = []
    tree2 = []
    for s in top_from_passage(p1):
        if s in mapping and mapping[s] not in tree2:
            tree1.append(s)
            tree2.append(mapping[s])

    for s in top_from_passage(p1):
        if s not in tree1:
            tree1.append(s)
    for s in top_from_passage(p2):
        if s not in tree2:
            tree2.append(s)

    # convert recursivly
    longer = max(len(top1), len(top2))
    shorter = min(len(top1), len(top2))
    res1 = []
    res2 = []
    for i in range(shorter):
        t1, t2 = tree_structure_aligned(tree1[i], tree2[i], word2word)
        res1.append(t1)
        res2.append(t2)
    if len(top1) == longer:
        res1 += [tree_structure(n) for n in top1[shorter:]]
    if len(top2) == longer:
        res2 += [tree_structure(n) for n in top2[shorter:]]
    res1 = prune_leaves((p1, res1))
    res2 = prune_leaves((p2, res2))
    res1 = prune_leaves(res1, lambda x: not is_comparable(x))
    res2 = prune_leaves(res2, lambda x: not is_comparable(x))
    return res1, res2


def aligned_edit_distance(p1, p2):
    """ uses the aligned trees for labeled tree edit distance"""
    tree1, tree2 = create_ordered_trees(p1, p2)
    return zss.simple_distance(convert_structure_to_zss(tree1), convert_structure_to_zss(tree2))


def token_level_similarity(p1, p2):
    return token_level_analysis([p1, p2])


def token_level_analysis(ps):
    """ takes a list of passages and computes the different token-level analysis"""
    s, e, f = {}, {}, {}
    i = 0
    while i < len(ps):
        p1 = ps[i]
        i += 1
        p2 = ps[i]
        i += 1
        count1 = Counter((label(node) for node in p1.layer(layer1.LAYER_ID).all if is_comparable(node)))
        count2 = Counter((label(node) for node in p2.layer(layer1.LAYER_ID).all if is_comparable(node)))
        for tag in MAIN_RELATIONS:
            f[tag] = f.get(tag, 0) + count1[tag]
            e[tag] = e.get(tag, 0) + count2[tag]
            s[tag] = s.get(tag, 0) + min(count1[tag], count2[tag])
    ADs = "As + Ds"
    s[ADs] = s[layer1.EdgeTags.Participant] + s[layer1.EdgeTags.Adverbial]
    e[ADs] = e[layer1.EdgeTags.Participant] + e[layer1.EdgeTags.Adverbial]
    f[ADs] = f[layer1.EdgeTags.Participant] + f[layer1.EdgeTags.Adverbial]
    P = {}
    R = {}
    res = {}
    for tag in MAIN_RELATIONS + [ADs]:
        P[tag] = s[tag] / f[tag]
        R[tag] = s[tag] / e[tag]
        res[tag] = 2 * (P[tag] * R[tag]) / (P[tag] + R[tag])

    return res


def aligned_top_down_distance(p1, p2):
    """starts from the heads of both passages
       and finds the amount of nodes 
       containing the same labeles for children"""
    word2word = align_yields(p1, p2)
    remaining = align_nodes(top_from_passage(p1), top_from_passage(p2), word2word)
    uncounted = {}
    cut1 = set()
    cut2 = set()
    overall1 = set()
    overall2 = set()
    # top down create find the maximum cut of matching nodes
    while remaining:
        n1, n2 = remaining.popitem()
        if is_comparable(n1) and is_comparable(n2):
            # remember overall nodes
            overall1.add(n1)
            overall2.add(n2)
            if compare(n1, n2):
                cut1.add(n1)
                cut2.add(n2)
                remaining.update(align_nodes(n1.children, n2.children, word2word))
            else:
                # remember the end of the cut
                uncounted.update(align_nodes(n1.children, n2.children, word2word))
        elif (not is_terminal(n1)) and (not is_terminal(n2)):
            remaining.update(align_nodes(n1.children, n2.children, word2word))

    # check nodes not in the cut
    while uncounted:
        n1, n2 = uncounted.popitem()
        if is_comparable(n1) and is_comparable(n2):
            overall1.add(n1)
            overall2.add(n2)
        if (not is_terminal(n1)) and (not is_terminal(n2)):
            uncounted.update(align_nodes(n1.children, n2.children, word2word))

    # overall1 = set(node for node in p1.layer(layer1.LAYER_ID).all if is_comparable(node))
    # overall2 = set(node for node in p2.layer(layer1.LAYER_ID).all if is_comparable(node))
    print(inspect.currentframe().f_code.co_name)
    print(len(set(node for node in p1.layer(layer1.LAYER_ID).all if is_comparable(node))))
    print(len(set(node for node in p2.layer(layer1.LAYER_ID).all if is_comparable(node))))
    print("counts", len(cut1), len(cut2))
    print("lens", len(overall1), len(overall2))
    print(two_sided_f(len(cut1), len(cut2), len(overall1), len(overall2)))
    return two_sided_f(len(cut1), len(cut2), len(overall1), len(overall2))
