"""Testing code for the ucca package, unit-testing only."""

import operator
import unittest
import xml.etree.ElementTree as ETree

import random

from ucca import core, layer0, layer1, convert, textutil, ioutil, diffutil


class CoreTests(unittest.TestCase):

    def test_creation(self):

        p = TestUtil.create_basic_passage()

        self.assertEqual(p.ID, "1")
        self.assertEqual(p.root, p)
        self.assertDictEqual(p.attrib.copy(), {})
        self.assertEqual(p.layer("1").ID, "1")
        self.assertEqual(p.layer("2").ID, "2")
        self.assertRaises(KeyError, p.layer, "3")

        l1 = p.layer("1")
        l2 = p.layer("2")
        self.assertEqual(l1.root, p)
        self.assertEqual(l2.attrib["test"], True)
        self.assertNotEqual(l1.orderkey, l2.orderkey)
        self.assertSequenceEqual([x.ID for x in l1.all], ["1.1", "1.2", "1.3"])
        self.assertSequenceEqual([x.ID for x in l1.heads], ["1.2"])
        self.assertSequenceEqual([x.ID for x in l2.all], ["2.2", "2.1"])
        self.assertSequenceEqual([x.ID for x in l2.heads], ["2.2", "2.1"])

        node11, node12, node13 = l1.all
        node22, node21 = l2.all
        self.assertEqual(node11.ID, "1.1")
        self.assertEqual(node11.root, p)
        self.assertEqual(node11.layer.ID, "1")
        self.assertEqual(node11.tag, "1")
        self.assertEqual(len(node11), 0)
        self.assertSequenceEqual(node11.parents, [node12, node21, node22])
        self.assertSequenceEqual(node13.parents, [node12, node22])
        self.assertDictEqual(node13.attrib.copy(), {"node": True})
        self.assertEqual(len(node12), 2)
        self.assertSequenceEqual(node12.children, [node13, node11])
        self.assertDictEqual(node12[0].attrib.copy(), {"edge": True})
        self.assertSequenceEqual(node12.parents, [node22, node21])
        self.assertEqual(node21[0].ID, "2.1->1.1")
        self.assertEqual(node21[1].ID, "2.1->1.2")
        self.assertEqual(node22[0].ID, "2.2->1.1")
        self.assertEqual(node22[1].ID, "2.2->1.2")
        self.assertEqual(node22[2].ID, "2.2->1.3")

    def test_modifying(self):
        p = TestUtil.create_basic_passage()
        l1, l2 = p.layer("1"), p.layer("2")
        node11, node12, node13 = l1.all
        node22, node21 = l2.all

        # Testing attribute changes
        p.attrib["passage"] = 1
        self.assertDictEqual(p.attrib.copy(), {"passage": 1})
        del l2.attrib["test"]
        self.assertDictEqual(l2.attrib.copy(), {})
        node13.attrib[1] = 1
        self.assertDictEqual(node13.attrib.copy(), {"node": True, 1: 1})
        self.assertEqual(len(node13.attrib), 2)
        self.assertEqual(node13.attrib.get("node"), True)
        self.assertEqual(node13.attrib.get("missing"), None)

        # Testing Node changes
        node14 = core.Node(ID="1.4", root=p, tag="4")
        node15 = core.Node(ID="1.5", root=p, tag="5")
        self.assertSequenceEqual(l1.all, [node11, node12, node13, node14,
                                          node15])
        self.assertSequenceEqual(l1.heads, [node12, node14, node15])
        node15.add("test", node11)
        self.assertSequenceEqual(node11.parents, [node12, node15, node21,
                                                  node22])
        node21.remove(node12)
        node21.remove(node21[0])
        self.assertEqual(len(node21), 0)
        self.assertSequenceEqual(node12.parents, [node22])
        self.assertSequenceEqual(node11.parents, [node12, node15, node22])
        node14.add("test", node15)
        self.assertSequenceEqual(l1.heads, [node12, node14])
        node12.destroy()
        self.assertSequenceEqual(l1.heads, [node13, node14])
        self.assertSequenceEqual(node22.children, [node11, node13])

        node22.tag = "x"
        node22[0].tag = "testx"
        self.assertEqual(node22.tag, "x")
        self.assertEqual(node22[0].tag, "testx")

    def test_equals(self):
        p1 = core.Passage("1")
        p2 = core.Passage("2")
        p1l0 = layer0.Layer0(p1)
        p2l0 = layer0.Layer0(p2)
        p1l1 = layer1.Layer1(p1)
        p2l1 = layer1.Layer1(p2)
        self.assertTrue(p1.equals(p2) and p2.equals(p1))

        # Checks basic passage equality and Attrib/tag/len differences
        p1l0.add_terminal("0", False)
        p1l0.add_terminal("1", False)
        p1l0.add_terminal("2", False)
        p2l0.add_terminal("0", False)
        p2l0.add_terminal("1", False)
        p2l0.add_terminal("2", False)
        self.assertTrue(p1.equals(p2) and p2.equals(p1))
        pnct2 = p2l0.add_terminal("3", True)
        self.assertFalse(p1.equals(p2) or p2.equals(p1))
        temp = p1l0.add_terminal("3", False)
        self.assertFalse(p1.equals(p2) or p2.equals(p1))
        temp.destroy()
        pnct1 = p1l0.add_terminal("3", True)
        self.assertTrue(p1.equals(p2) and p2.equals(p1))

        # Check Edge and node equality
        ps1 = p1l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
        self.assertFalse(p1.equals(p2) or p2.equals(p1))
        ps2 = p2l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
        self.assertTrue(p1.equals(p2) and p2.equals(p1))
        p1l1.add_fnode(ps1, layer1.EdgeTags.Participant)
        self.assertFalse(p1.equals(p2) or p2.equals(p1))
        self.assertTrue(ps1.equals(ps2, recursive=False))
        p2l1.add_fnode(ps2, layer1.EdgeTags.Process)
        self.assertFalse(p1.equals(p2) or p2.equals(p1))
        p2l1.add_fnode(ps2, layer1.EdgeTags.Participant)
        self.assertFalse(p1.equals(p2) or p2.equals(p1))
        p1l1.add_fnode(ps1, layer1.EdgeTags.Process)
        self.assertTrue(p1.equals(p2) and p2.equals(p1))
        self.assertFalse(p1.equals(p2, ordered=True) or
                         p2.equals(p1, ordered=True))
        p1l1.add_fnode(ps1, layer1.EdgeTags.Adverbial, implicit=True)
        ps2d3 = p2l1.add_fnode(ps2, layer1.EdgeTags.Adverbial)
        self.assertFalse(p1.equals(p2) or p2.equals(p1))
        ps2d3.attrib["implicit"] = True
        self.assertTrue(p1.equals(p2) and p2.equals(p1))
        ps2[2].attrib["remote"] = True
        self.assertFalse(p1.equals(p2) or p2.equals(p1))
        ps1[2].attrib["remote"] = True
        self.assertTrue(p1.equals(p2) and p2.equals(p1))
        p1l1.add_punct(None, pnct1)
        self.assertFalse(p1.equals(p2) or p2.equals(p1))
        p2l1.add_punct(None, pnct2)
        self.assertTrue(p1.equals(p2) and p2.equals(p1))
        core.Layer("2", p1)
        self.assertFalse(p1.equals(p2) or p2.equals(p1))

    def test_copying(self):
        # we don't need such a complex passage, but it will work anyway
        p1 = TestUtil.create_passage()

        p2 = p1.copy(())
        self.assertEqual(p1.ID, p2.ID)
        self.assertTrue(p1.attrib.equals(p2.attrib))
        self.assertEqual(p1.extra, p2.extra)
        self.assertEqual(p1.frozen, p2.frozen)

        l0id = layer0.LAYER_ID
        p2 = p1.copy([l0id])
        self.assertTrue(p1.layer(l0id).equals(p2.layer(l0id)))

    def test_iteration(self):
        p = TestUtil.create_basic_passage()
        l1, l2 = p.layer("1"), p.layer("2")
        node11, node12, node13 = l1.all
        node22, node21 = l2.all

        self.assertSequenceEqual(list(node11.iter()), [node11])
        self.assertSequenceEqual(list(node11.iter(obj="edges")), ())
        self.assertSequenceEqual(list(node13.iter(key=lambda x: x.tag != "3")),
                                 ())
        self.assertSequenceEqual(list(node12.iter()), [node12, node13, node11])
        self.assertSequenceEqual(list(x.ID for x in node12.iter(obj="edges")),
                                 ["1.2->1.3", "1.2->1.1"])
        self.assertSequenceEqual(list(node21.iter(duplicates=True)),
                                 [node21, node11, node12, node13, node11])
        self.assertSequenceEqual(list(node21.iter()),
                                 [node21, node11, node12, node13])
        self.assertSequenceEqual(list(node22.iter(method="bfs",
                                                  duplicates=True)),
                                 [node22, node11, node12, node13, node13,
                                  node11])


class Layer0Tests(unittest.TestCase):
    """Tests module layer0 functionality."""

    def test_terminals(self):
        """Tests :class:layer0.Terminal new and inherited functionality."""
        p = core.Passage("1")
        layer0.Layer0(p)
        terms = [
            layer0.Terminal(ID="0.1", root=p,
                            tag=layer0.NodeTags.Word,
                            attrib={"text": "1",
                                    "paragraph": 1,
                                    "paragraph_position": 1}),
            layer0.Terminal(ID="0.2", root=p,
                            tag=layer0.NodeTags.Word,
                            attrib={"text": "2",
                                    "paragraph": 2,
                                    "paragraph_position": 1}),
            layer0.Terminal(ID="0.3", root=p,
                            tag=layer0.NodeTags.Punct,
                            attrib={"text": ".",
                                    "paragraph": 2,
                                    "paragraph_position": 2})
        ]

        p_copy = core.Passage("2")
        layer0.Layer0(p_copy)
        equal_term = layer0.Terminal(ID="0.1", root=p_copy,
                                     tag=layer0.NodeTags.Word,
                                     attrib={"text": "1",
                                             "paragraph": 1,
                                             "paragraph_position": 1})
        unequal_term = layer0.Terminal(ID="0.2", root=p_copy,
                                       tag=layer0.NodeTags.Word,
                                       attrib={"text": "two",
                                               "paragraph": 2,
                                               "paragraph_position": 1})

        self.assertSequenceEqual([t.punct for t in terms],
                                 [False, False, True])
        self.assertSequenceEqual([t.text for t in terms], ["1", "2", "."])
        self.assertSequenceEqual([t.position for t in terms], [1, 2, 3])
        self.assertSequenceEqual([t.paragraph for t in terms], [1, 2, 2])
        self.assertSequenceEqual([t.para_pos for t in terms], [1, 1, 2])
        self.assertFalse(terms[0] == terms[1])
        self.assertFalse(terms[0] == terms[2])
        self.assertFalse(terms[1] == terms[2])
        self.assertTrue(terms[0] == terms[0])
        self.assertTrue(terms[0].equals(equal_term))
        self.assertFalse(terms[1].equals(unequal_term))

    def test_layer0(self):
        p = core.Passage("1")
        l0 = layer0.Layer0(p)
        t1 = l0.add_terminal(text="1", punct=False)
        l0.add_terminal(text="2", punct=True, paragraph=2)
        t3 = l0.add_terminal(text="3", punct=False, paragraph=2)
        self.assertSequenceEqual([x[0] for x in l0.pairs], [1, 2, 3])
        self.assertSequenceEqual([t.para_pos for t in l0.all], [1, 1, 2])
        self.assertSequenceEqual(l0.words, (t1, t3))


class Layer1Tests(unittest.TestCase):
    """Tests layer1 module functionality and correctness."""

    def test_creation(self):
        p = TestUtil.create_passage()
        head = p.layer("1").heads[0]
        self.assertSequenceEqual([x.tag for x in head], ["L", "H", "H", "U"])
        self.assertSequenceEqual([x.child.position for x in head.children[0]],
                                 [1])
        self.assertSequenceEqual([x.tag for x in head.children[1]],
                                 ["P", "A", "U", "A"])
        self.assertSequenceEqual([x.child.position
                                  for x in head.children[1].children[0]],
                                 [2, 3, 4, 5])
        self.assertSequenceEqual([x.child.position
                                  for x in head.children[1].children[1]],
                                 [6, 7, 8, 9])
        self.assertSequenceEqual([x.child.position
                                  for x in head.children[1].children[2]],
                                 [10])
        self.assertTrue(head.children[1][3].attrib.get("remote"))

    def test_fnodes(self):
        p = TestUtil.create_passage()
        l0 = p.layer("0")
        l1 = p.layer("1")

        terms = l0.all
        head, lkg1, lkg2 = l1.heads
        link1, ps1, ps23, punct2 = head.children
        p1, a1, punct1 = [x.child for x in ps1 if not x.attrib.get("remote")]
        ps2, link2, ps3 = ps23.children
        a2, d2 = [x.child for x in ps2 if not x.attrib.get("remote")]
        p3, a3, a4 = ps3.children

        self.assertEqual(lkg1.relation, link1)
        self.assertSequenceEqual(lkg1.arguments, [ps1])
        self.assertIsNone(ps23.process)
        self.assertEqual(ps2.process, p1)
        self.assertSequenceEqual(ps1.participants, [a1, d2])
        self.assertSequenceEqual(ps3.participants, [a3, a4])

        self.assertSequenceEqual(ps1.get_terminals(), terms[1:10])
        self.assertSequenceEqual(ps1.get_terminals(punct=False, remotes=True),
                                 terms[1:9] + terms[14:15])
        self.assertEqual(ps1.end_position, 10)
        self.assertEqual(ps2.start_position, 11)
        self.assertEqual(ps3.start_position, 17)
        self.assertEqual(a4.start_position, -1)
        self.assertEqual(ps23.to_text(), "11 12 13 14 15 16 17 18 19")

        self.assertEqual(ps1.fparent, head)
        self.assertEqual(link2.fparent, ps23)
        self.assertEqual(ps2.fparent, ps23)
        self.assertEqual(d2.fparent, ps2)

    def test_layer1(self):
        p = TestUtil.create_passage()
        l1 = p.layer("1")

        head, lkg1, lkg2 = l1.heads
        link1, ps1, ps23, punct2 = head.children
        p1, a1, punct1 = [x.child for x in ps1 if not x.attrib.get("remote")]
        ps2, link2, ps3 = ps23.children

        self.assertSequenceEqual(l1.top_scenes, [ps1, ps2, ps3])
        self.assertSequenceEqual(l1.top_linkages, [lkg1, lkg2])

        # adding scene #23 to linkage #1, which makes it non top-level as
        # scene #23 isn't top level
        lkg1.add(layer1.EdgeTags.LinkArgument, ps23)
        self.assertSequenceEqual(l1.top_linkages, [lkg2])

        # adding process to scene #23, which makes it top level and discards
        # "top-levelness" from scenes #2 + #3
        l1.add_remote(ps23, layer1.EdgeTags.Process, p1)
        self.assertSequenceEqual(l1.top_scenes, [ps1, ps23])
        self.assertSequenceEqual(l1.top_linkages, [lkg1, lkg2])

        # Changing the process tag of scene #1 to A and back, validate that
        # top scenes are updates accordingly
        p_edge = [e for e in ps1 if e.tag == layer1.EdgeTags.Process][0]
        p_edge.tag = layer1.EdgeTags.Participant
        self.assertSequenceEqual(l1.top_scenes, [ps23])
        self.assertSequenceEqual(l1.top_linkages, [lkg2])
        p_edge.tag = layer1.EdgeTags.Process
        self.assertSequenceEqual(l1.top_scenes, [ps1, ps23])
        self.assertSequenceEqual(l1.top_linkages, [lkg1, lkg2])

    def test_str(self):
        p = TestUtil.create_passage()
        self.assertSequenceEqual([str(x) for x in p.layer("1").heads],
                                 ["[L 1] [H [P 2 3 4 5] [A 6 7 8 9] [U 10] "
                                  "... [A* 15] ] [H [H [P* 2 3 4 5] [A 11 12 "
                                  "13 14] [D 15] ] [L 16] [H [A IMPLICIT] [S "
                                  "17 18] [A 19] ] ] [U 20] ",
                                  "1.2-->1.3", "1.11-->1.8,1.12"])

    def test_destroy(self):
        p = TestUtil.create_passage()
        l1 = p.layer("1")

        head, lkg1, lkg2 = l1.heads
        link1, ps1, ps23, punct2 = head.children
        p1, a1, punct1 = [x.child for x in ps1 if not x.attrib.get("remote")]
        ps2, link2, ps3 = ps23.children

        ps1.destroy()
        self.assertSequenceEqual(head.children, [link1, ps23, punct2])
        self.assertSequenceEqual(p1.parents, [ps2])
        self.assertFalse(a1.parents)
        self.assertFalse(punct1.parents)

    def test_discontiguous(self):
        """Tests FNode.discontiguous and FNode.get_sequences"""
        p = TestUtil.create_discontiguous()
        l1 = p.layer("1")
        head = l1.heads[0]
        ps1, ps2, ps3 = head.children
        d1, a1, p1, f1 = ps1.children
        e1, c1, e2, g1 = d1.children
        d2, g2, p2, a2 = ps2.children
        t14, p3, a3 = ps3.children

        # Checking discontiguous property
        self.assertFalse(ps1.discontiguous)
        self.assertFalse(d1.discontiguous)
        self.assertFalse(e1.discontiguous)
        self.assertFalse(e2.discontiguous)
        self.assertTrue(c1.discontiguous)
        self.assertTrue(g1.discontiguous)
        self.assertTrue(a1.discontiguous)
        self.assertTrue(p1.discontiguous)
        self.assertFalse(f1.discontiguous)
        self.assertTrue(ps2.discontiguous)
        self.assertFalse(p2.discontiguous)
        self.assertFalse(a2.discontiguous)
        self.assertFalse(ps3.discontiguous)
        self.assertFalse(a3.discontiguous)

        # Checking get_sequences -- should return only non-remote, non-implicit
        # stretches of terminals
        self.assertSequenceEqual(ps1.get_sequences(), [(1, 10)])
        self.assertSequenceEqual(d1.get_sequences(), [(1, 4)])
        self.assertSequenceEqual(e1.get_sequences(), [(1, 1)])
        self.assertSequenceEqual(e2.get_sequences(), [(3, 3)])
        self.assertSequenceEqual(c1.get_sequences(), [(2, 2), (4, 4)])
        self.assertSequenceEqual(a1.get_sequences(), [(5, 5), (8, 8)])
        self.assertSequenceEqual(p1.get_sequences(), [(6, 7), (10, 10)])
        self.assertSequenceEqual(f1.get_sequences(), [(9, 9)])
        self.assertSequenceEqual(ps2.get_sequences(), [(11, 14), (18, 20)])
        self.assertSequenceEqual(p2.get_sequences(), [(11, 14)])
        self.assertSequenceEqual(a2.get_sequences(), [(18, 20)])
        self.assertSequenceEqual(d2.get_sequences(), ())
        self.assertSequenceEqual(g2.get_sequences(), ())
        self.assertSequenceEqual(ps3.get_sequences(), [(15, 17)])
        self.assertSequenceEqual(a3.get_sequences(), [(16, 17)])
        self.assertSequenceEqual(p3.get_sequences(), ())


class ConversionTests(unittest.TestCase):
    """Tests convert module correctness and API."""

    def _test_edges(self, node, tags):
        """Tests that the node edge tags and number match tags argument."""
        self.assertEqual(len(node), len(tags))
        for edge, tag in zip(node, tags):
            self.assertEqual(edge.tag, tag)

    def _test_terms(self, node, terms):
        """Tests that node contain the terms given, and only them."""
        for edge, term in zip(node, terms):
            self.assertEqual(edge.tag, layer1.EdgeTags.Terminal)
            self.assertEqual(edge.child, term)

    def test_site_terminals(self):
        elem = TestUtil.load_xml("test_files/site1.xml")
        passage = convert.from_site(elem)
        terms = passage.layer(layer0.LAYER_ID).all

        self.assertEqual(passage.ID, "118")
        self.assertEqual(len(terms), 15)

        # There are two punctuation signs (dots, positions 5 and 11), which
        # also serve as paragraph end points. All others are words whose text
        # is their positions, so test that both text, punctuation (yes/no)
        # and paragraphs are converted correctly
        for i, t in enumerate(terms):
            # i starts in 0, positions at 1, hence 5,11 ==> 4,10
            if i in (4, 10):
                self.assertTrue(t.text == "." and t.punct is True)
            else:
                self.assertTrue(t.text == str(i + 1) and t.punct is False)
            if i < 5:
                par = 1
            elif i < 11:
                par = 2
            else:
                par = 3
            self.assertEqual(t.paragraph, par)

    def test_site_simple(self):
        elem = TestUtil.load_xml("test_files/site2.xml")
        passage = convert.from_site(elem)
        terms = passage.layer(layer0.LAYER_ID).all
        l1 = passage.layer("1")

        # The Terminals in the passage are just like in test_site_terminals,
        # with this layer1 hierarchy: [[1 C] [2 E] L] [3 4 . H]
        # with the linker having a remark and the parallel scene is uncertain
        head = l1.heads[0]
        self.assertEqual(len(head), 12)  # including all "unused" terminals
        self.assertEqual(head[9].tag, layer1.EdgeTags.Linker)
        self.assertEqual(head[10].tag, layer1.EdgeTags.ParallelScene)
        linker = head.children[9]
        self._test_edges(linker, [layer1.EdgeTags.Center,
                                  layer1.EdgeTags.Elaborator])
        self.assertTrue(linker.extra["remarks"], '"remark"')
        center = linker.children[0]
        elab = linker.children[1]
        self._test_terms(center, terms[0:1])
        self._test_terms(elab, terms[1:2])
        ps = head.children[10]
        self._test_edges(ps, [layer1.EdgeTags.Terminal,
                              layer1.EdgeTags.Terminal,
                              layer1.EdgeTags.Punctuation])
        self.assertTrue(ps.attrib.get("uncertain"))
        self.assertEqual(ps.children[0], terms[2])
        self.assertEqual(ps.children[1], terms[3])
        self.assertEqual(ps.children[2].children[0], terms[4])

    def test_site_advanced(self):
        elem = TestUtil.load_xml("test_files/site3.xml")
        passage = convert.from_site(elem)
        terms = passage.layer(layer0.LAYER_ID).all
        l1 = passage.layer("1")

        # This passage has the same terminals as the simple and terminals test,
        # and have the same layer1 units for the first paragraph as the simple
        # test. In addition, it has the following annotation:
        # [6 7 8 9 H] [10 F] .
        # the 6-9 H has remote D which is [10 F]. Inside of 6-9, we have [8 S]
        # and [6 7 ... 9 A], where [6 E] and [7 ... 9 C].
        # [12 H] [13 H] [14 H] [15 L], where 15 linkage links 12, 13 and 14 and
        # [15 L] has an implicit Center unit
        head, lkg = l1.heads
        self._test_edges(head, [layer1.EdgeTags.Linker,
                                layer1.EdgeTags.ParallelScene,
                                layer1.EdgeTags.ParallelScene,
                                layer1.EdgeTags.Function,
                                layer1.EdgeTags.Punctuation,
                                layer1.EdgeTags.ParallelScene,
                                layer1.EdgeTags.ParallelScene,
                                layer1.EdgeTags.ParallelScene,
                                layer1.EdgeTags.Linker])

        # we only take what we haven"t checked already
        ps1, func, punct, ps2, ps3, ps4, link = head.children[2:]
        self._test_edges(ps1, [layer1.EdgeTags.Participant,
                               layer1.EdgeTags.Process,
                               layer1.EdgeTags.Adverbial])
        self.assertTrue(ps1[2].attrib.get("remote"))
        ps1_a, ps1_p, ps1_d = ps1.children
        self._test_edges(ps1_a, [layer1.EdgeTags.Elaborator,
                                 layer1.EdgeTags.Center])
        self._test_terms(ps1_a.children[0], terms[5:6])
        self._test_terms(ps1_a.children[1], terms[6:9:2])
        self._test_terms(ps1_p, terms[7:8])
        self.assertEqual(ps1_d, func)
        self._test_terms(func, terms[9:10])
        self._test_terms(punct, terms[10:11])
        self._test_terms(ps2, terms[11:12])
        self._test_terms(ps3, terms[12:13])
        self._test_terms(ps4, terms[13:14])
        self.assertEqual(len(link), 2)
        self.assertEqual(link[0].tag, layer1.EdgeTags.Center)
        self.assertTrue(link.children[0].attrib.get("implicit"))
        self.assertEqual(link[1].tag, layer1.EdgeTags.Elaborator)
        self.assertEqual(link.children[1][0].tag, layer1.EdgeTags.Terminal)
        self.assertEqual(link.children[1][0].child, terms[14])
        self.assertEqual(lkg.relation, link)
        self.assertSequenceEqual(lkg.arguments, [ps2, ps3, ps4])

    def test_to_standard(self):
        passage = convert.from_site(TestUtil.load_xml("test_files/site3.xml"))
        ref = TestUtil.load_xml("test_files/standard3.xml")
        root = convert.to_standard(passage)
        self.assertEqual(ETree.tostring(ref), ETree.tostring(root))

    def test_from_standard(self):
        passage = convert.from_standard(TestUtil.load_xml("test_files/standard3.xml"))
        ref = convert.from_site(TestUtil.load_xml("test_files/site3.xml"))
        self.assertTrue(passage.equals(ref, ordered=True))

    def test_from_text(self):
        sample = ["Hello . again", "nice", " ? ! end", ""]
        passage = next(convert.from_text(sample))
        terms = passage.layer(layer0.LAYER_ID).all
        pos = 0
        for i, par in enumerate(sample):
            for text in par.split():
                self.assertEqual(terms[pos].text, text)
                self.assertEqual(terms[pos].paragraph, i + 1)
                pos += 1

    def test_from_text_long(self):
        sample = """
            After graduation, John moved to New York City.
            
            He liked it there. He played tennis.
            And basketball.
            
            And he lived happily ever after.
            """
        passages = list(convert.from_text(sample))
        self.assertEqual(len(passages), 3, list(map(convert.to_text, passages)))

    def test_to_text(self):
        passage = convert.from_standard(TestUtil.load_xml("test_files/standard3.xml"))
        self.assertEqual(convert.to_text(passage, False)[0],
                         "1 2 3 4 . 6 7 8 9 10 . 12 13 14 15")
        self.assertSequenceEqual(convert.to_text(passage, True),
                                 ["1 2 3 4 .", "6 7 8 9 10 .", "12 13 14 15"])

    def test_to_site(self):
        passage = convert.from_standard(TestUtil.load_xml("test_files/standard3.xml"))
        root = convert.to_site(passage)
        copy = convert.from_site(root)
        self.assertTrue(passage.equals(copy))

    def test_to_conll(self):
        passage = convert.from_standard(TestUtil.load_xml("test_files/standard3.xml"))
        converted = convert.to_conll(passage)
        with open("test_files/standard3.conll", encoding="utf-8") as f:
            # f.write("\n".join(converted))
            self.assertSequenceEqual(converted, f.read().splitlines() + [""])
        converted_passage = next(convert.from_conll(converted, passage.ID))
        # ioutil.passage2file(converted_passage, "test_files/standard3.conll.xml")
        ref = convert.from_standard(TestUtil.load_xml("test_files/standard3.conll.xml"))
        self.assertTrue(converted_passage.equals(ref))
        # Put the same sentence twice and try converting again
        for converted_passage in convert.from_conll(converted * 2, passage.ID):
            ref = convert.from_standard(TestUtil.load_xml("test_files/standard3.conll.xml"))
        self.assertTrue(converted_passage.equals(ref))

    def test_to_sdp(self):
        passage = convert.from_standard(TestUtil.load_xml("test_files/standard3.xml"))
        converted = convert.to_sdp(passage)
        with open("test_files/standard3.sdp", encoding="utf-8") as f:
            # f.write("\n".join(converted))
            self.assertSequenceEqual(converted, f.read().splitlines() + [""])
        converted_passage = next(convert.from_sdp(converted, passage.ID))
        # ioutil.passage2file(converted_passage, "test_files/standard3.sdp.xml")
        ref = convert.from_standard(TestUtil.load_xml("test_files/standard3.sdp.xml"))
        self.assertTrue(converted_passage.equals(ref))

    def test_to_export(self):
        passage = convert.from_standard(TestUtil.load_xml("test_files/standard3.xml"))
        converted = convert.to_export(passage)
        with open("test_files/standard3.export", encoding="utf-8") as f:
            # f.write("\n".join(converted))
            self.assertSequenceEqual(converted, f.read().splitlines())
        converted_passage = next(convert.from_export(converted, passage.ID))
        # ioutil.passage2file(converted_passage, "test_files/standard3.export.xml")
        ref = convert.from_standard(TestUtil.load_xml("test_files/standard3.export.xml"))
        self.assertTrue(converted_passage.equals(ref))


class UtilTests(unittest.TestCase):
    """Tests the util module functions and classes."""

    def test_break2sentences(self):
        """Tests identifying correctly sentence ends.
        """
        p = TestUtil.create_multi_passage()
        self.assertSequenceEqual(textutil.break2sentences(p), [4, 7, 11])

    def test_split2sentences(self):
        """Tests splitting a passage by sentence ends.
        """
        p = TestUtil.create_multi_passage()
        split = convert.split2sentences(p)
        self.assertEqual(len(split), 3)
        terms = [[t.text for t in s.layer(layer0.LAYER_ID).all] for s in split]
        self.assertSequenceEqual(terms[0], ["1", "2", "3", "."])
        self.assertSequenceEqual(terms[1], ["5", "6", "."])
        self.assertSequenceEqual(terms[2], ["8", ".", "10", "."])
        self.assertTrue(all(t.paragraph == 1 for s in split[0:2]
                            for t in s.layer(layer0.LAYER_ID).all))
        self.assertTrue(all(t.paragraph == 2
                            for t in split[2].layer(layer0.LAYER_ID).all))
        top_scenes = [s.layer(layer1.LAYER_ID).top_scenes for s in split]
        for t in top_scenes:
            self.assertEqual(len(t), 1)
            self.assertEqual(t[0].incoming[0].tag, layer1.EdgeTags.ParallelScene)

    def test_split2paragraphs(self):
        """Tests splitting a passage by paragraph ends.
        """
        p = TestUtil.create_multi_passage()
        split = convert.split2paragraphs(p)
        self.assertEqual(len(split), 2)
        terms = [[t.text for t in s.layer(layer0.LAYER_ID).all] for s in split]
        self.assertSequenceEqual(terms[0], ["1", "2", "3", ".", "5", "6", "."])
        self.assertSequenceEqual(terms[1], ["8", ".", "10", "."])
        self.assertTrue(all(t.paragraph == 1
                            for t in split[0].layer(layer0.LAYER_ID).all))
        self.assertTrue(all(t.paragraph == 2
                            for t in split[1].layer(layer0.LAYER_ID).all))
        top_scenes = [s.layer(layer1.LAYER_ID).top_scenes for s in split]
        self.assertEqual(len(top_scenes[0]), 2)
        self.assertEqual(len(top_scenes[1]), 1)
        for t in top_scenes:
            for n in t:
                self.assertEqual(n.incoming[0].tag, layer1.EdgeTags.ParallelScene)

    def test_split_join_sentences(self):
        p = TestUtil.create_multi_passage()
        split = convert.split2sentences(p, remarks=True)
        copy = convert.join_passages(split)
        diffutil.diff_passages(p, copy)
        self.assertTrue(p.equals(copy))

    def test_split_join_paragraphs(self):
        p = TestUtil.create_multi_passage()
        split = convert.split2paragraphs(p, remarks=True)
        copy = convert.join_passages(split)
        diffutil.diff_passages(p, copy)
        self.assertTrue(p.equals(copy))

    # def test_split_join_sentences_crossing(self):
    #     """Test that splitting and joining a passage by sentences results in the same passage,
    #     when the passage has edges crossing sentences.
    #     """
    #     p = TestUtil.create_crossing_passage()
    #     split = textutil.split2sentences(p, remarks=True)
    #     copy = textutil.join_passages(split)
    #     diffutil.diff_passages(p, copy)
    #     self.assertTrue(p.equals(copy))
    #
    # def test_split_join_paragraphs_crossing(self):
    #     """Test that splitting and joining a passage by paragraphs results in the same passage
    #     when the passage has edges crossing paragraphs.
    #     """
    #     p = TestUtil.create_crossing_passage()
    #     split = textutil.split2paragraphs(p, remarks=True)
    #     copy = textutil.join_passages(split)
    #     diffutil.diff_passages(p, copy)
    #     self.assertTrue(p.equals(copy))

    def test_shuffle_passages(self):
        """Test lazy-loading passages and shuffling them"""
        files = ["test_files/standard3.%s" % s for s in ("xml", "conll", "export", "sdp")]
        passages = ioutil.read_files_and_dirs(files)
        print("Passages:\n" + "\n".join(str(p.layer(layer1.LAYER_ID).heads[0]) for p in passages))
        random.shuffle(passages)
        print("Shuffled passages:\n" + "\n".join(str(p.layer(layer1.LAYER_ID).heads[0]) for p in passages))
        self.assertEqual(len(files), len(passages))


class TestUtil:
    """Utilities for tests."""
    
    @staticmethod
    def create_basic_passage():
        """Creates a basic :class:Passage to tinker with.

        Passage structure is as follows:
            Layer1: order by ID, heads = [1.2], all = [1.1, 1.2, 1.3]
            Layer2: order by node unique ID descending,
                    heads = all = [2.2, 2.1], attrib={"test": True}
            Nodes (tag):
                1.1 (1)
                1.3 (3), attrib={"node": True}
                1.2 (x), order by edge tag
                    children: 1.3 Edge: tag=test1, attrib={"Edge": True}
                              1.1 Edge: tag=test2
                2.1 (2), children [1.1, 1.2] with edge tags [test, test2]
                2.2 (2), children [1.1, 1.2, 1.3] with tags [test, test1, test]

        """
        p = core.Passage(ID="1")
        core.Layer(ID="1", root=p)
        core.Layer(ID="2", root=p, attrib={"test": True},
                   orderkey=lambda x: -1 * int(x.ID.split(".")[1]))

        # Order is explicitly different in order to break the alignment between
        # the ID/Edge ordering and the order of creation/addition
        node11 = core.Node(ID="1.1", root=p, tag="1")
        node13 = core.Node(ID="1.3", root=p, tag="3", attrib={"node": True})
        node12 = core.Node(ID="1.2", root=p, tag="x",
                           orderkey=operator.attrgetter("tag"))
        node21 = core.Node(ID="2.1", root=p, tag="2")
        node22 = core.Node(ID="2.2", root=p, tag="2")
        node12.add("test2", node11)
        node12.add("test1", node13, edge_attrib={"edge": True})
        node21.add("test2", node12)
        node21.add("test", node11)
        node22.add("test1", node12)
        node22.add("test", node13)
        node22.add("test", node11)
        return p

    @staticmethod
    def create_passage():
        """Creates a Passage to work with using layer1 objects.

        Annotation layout (what annotation each terminal has):
            1: Linker, linked with the first parallel scene
            2-10: Parallel scene #1, 2-5 ==> Participant #1
                6-9 ==> Process #1, 10 ==> Punctuation, remote Participant is
                Adverbial #2
            11-19: Parallel scene #23, which encapsulated 2 scenes and a linker
                (not a real scene, has no process, only for grouping)
            11-15: Parallel scene #2 (under #23), 11-14 ==> Participant #3,
                15 ==> Adverbial #2, remote Process is Process #1
            16: Linker #2, links Parallel scenes #2 and #3
            17-19: Parallel scene #3, 17-18 ==> Process #3,
                19 ==> Participant #3, implicit Participant
            20: Punctuation (under the head)

        """

        p = core.Passage("1")
        l0 = layer0.Layer0(p)
        l1 = layer1.Layer1(p)
        # 20 terminals (1-20), #10 and #20 are punctuation
        terms = [l0.add_terminal(text=str(i), punct=(i % 10 == 0))
                 for i in range(1, 21)]

        # Linker #1 with terminal 1
        link1 = l1.add_fnode(None, layer1.EdgeTags.Linker)
        link1.add(layer1.EdgeTags.Terminal, terms[0])

        # Scene #1: [[2 3 4 5 P] [6 7 8 9 A] [10 U] H]
        ps1 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
        p1 = l1.add_fnode(ps1, layer1.EdgeTags.Process)
        a1 = l1.add_fnode(ps1, layer1.EdgeTags.Participant)
        p1.add(layer1.EdgeTags.Terminal, terms[1])
        p1.add(layer1.EdgeTags.Terminal, terms[2])
        p1.add(layer1.EdgeTags.Terminal, terms[3])
        p1.add(layer1.EdgeTags.Terminal, terms[4])
        a1.add(layer1.EdgeTags.Terminal, terms[5])
        a1.add(layer1.EdgeTags.Terminal, terms[6])
        a1.add(layer1.EdgeTags.Terminal, terms[7])
        a1.add(layer1.EdgeTags.Terminal, terms[8])
        l1.add_punct(ps1, terms[9])

        # Scene #23: [[11 12 13 14 15 H] [16 L] [17 18 19 H] H]
        # Scene #2: [[11 12 13 14 P] [15 D]]
        ps23 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
        ps2 = l1.add_fnode(ps23, layer1.EdgeTags.ParallelScene)
        a2 = l1.add_fnode(ps2, layer1.EdgeTags.Participant)
        a2.add(layer1.EdgeTags.Terminal, terms[10])
        a2.add(layer1.EdgeTags.Terminal, terms[11])
        a2.add(layer1.EdgeTags.Terminal, terms[12])
        a2.add(layer1.EdgeTags.Terminal, terms[13])
        d2 = l1.add_fnode(ps2, layer1.EdgeTags.Adverbial)
        d2.add(layer1.EdgeTags.Terminal, terms[14])

        # Linker #2: [16 L]
        link2 = l1.add_fnode(ps23, layer1.EdgeTags.Linker)
        link2.add(layer1.EdgeTags.Terminal, terms[15])

        # Scene #3: [[16 17 S] [18 A] (implicit participant) H]
        ps3 = l1.add_fnode(ps23, layer1.EdgeTags.ParallelScene)
        p3 = l1.add_fnode(ps3, layer1.EdgeTags.State)
        p3.add(layer1.EdgeTags.Terminal, terms[16])
        p3.add(layer1.EdgeTags.Terminal, terms[17])
        a3 = l1.add_fnode(ps3, layer1.EdgeTags.Participant)
        a3.add(layer1.EdgeTags.Terminal, terms[18])
        l1.add_fnode(ps3, layer1.EdgeTags.Participant, implicit=True)

        # Punctuation #20 - not under a scene
        l1.add_punct(None, terms[19])

        # adding remote argument to scene #1, remote process to scene #2
        # creating linkages L1->H1, H2<-L2->H3
        l1.add_remote(ps1, layer1.EdgeTags.Participant, d2)
        l1.add_remote(ps2, layer1.EdgeTags.Process, p1)
        l1.add_linkage(link1, ps1)
        l1.add_linkage(link2, ps2, ps3)

        return p

    @staticmethod
    def create_multi_passage():
        """Creates a :class:Passage with multiple sentences and paragraphs.

        Passage: [1 2 [3 P] H] . [[5 6 . P] H]
                 [[8 P] . 10 . H]

        """
        p = core.Passage("1")
        l0 = layer0.Layer0(p)
        l1 = layer1.Layer1(p)
        terms = [l0.add_terminal(str(i), False) for i in range(1, 4)]
        terms.append(l0.add_terminal(".", True))
        terms.append(l0.add_terminal("5", False))
        terms.append(l0.add_terminal("6", False))
        terms.append(l0.add_terminal(".", True))
        terms.append(l0.add_terminal("8", False, paragraph=2))
        terms.append(l0.add_terminal(".", True, paragraph=2))
        terms.append(l0.add_terminal("10", False, paragraph=2))
        terms.append(l0.add_terminal(".", True, paragraph=2))
        h1 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
        h2 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
        h3 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
        p1 = l1.add_fnode(h1, layer1.EdgeTags.Process)
        p2 = l1.add_fnode(h2, layer1.EdgeTags.Process)
        p3 = l1.add_fnode(h3, layer1.EdgeTags.Process)
        h1.add(layer1.EdgeTags.Terminal, terms[0])
        h1.add(layer1.EdgeTags.Terminal, terms[1])
        p1.add(layer1.EdgeTags.Terminal, terms[2])
        l1.add_punct(None, terms[3])
        p2.add(layer1.EdgeTags.Terminal, terms[4])
        p2.add(layer1.EdgeTags.Terminal, terms[5])
        l1.add_punct(p2, terms[6])
        p3.add(layer1.EdgeTags.Terminal, terms[7])
        l1.add_punct(h3, terms[8])
        h3.add(layer1.EdgeTags.Terminal, terms[9])
        l1.add_punct(h3, terms[10])
        return p

    @staticmethod
    def create_crossing_passage():
        """Creates a :class:Passage with multiple sentences and paragraphs, with crossing edges.

        Passage: [1 2 [3 P(remote)] H] .
                 [[3 P] . 4 . H]

        """
        p = core.Passage("1")
        l0 = layer0.Layer0(p)
        l1 = layer1.Layer1(p)
        terms = [
            l0.add_terminal("1", False),
            l0.add_terminal("2", False),
            l0.add_terminal(".", True),
            l0.add_terminal("3", False, paragraph=2),
            l0.add_terminal(".", True, paragraph=2),
            l0.add_terminal("4", False, paragraph=2),
            l0.add_terminal(".", True, paragraph=2),
        ]
        h1 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
        h2 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
        p1 = l1.add_fnode(h2, layer1.EdgeTags.Process)
        l1.add_remote(h1, layer1.EdgeTags.Process, p1)
        h1.add(layer1.EdgeTags.Terminal, terms[0])
        h1.add(layer1.EdgeTags.Terminal, terms[1])
        l1.add_punct(None, terms[2])
        p1.add(layer1.EdgeTags.Terminal, terms[3])
        l1.add_punct(h2, terms[4])
        h2.add(layer1.EdgeTags.Terminal, terms[5])
        l1.add_punct(h2, terms[6])
        return p

    @staticmethod
    def create_discontiguous():
        """Creates a highly-discontiguous Passage object."""
        p = core.Passage("1")
        l0 = layer0.Layer0(p)
        l1 = layer1.Layer1(p)
        # 20 terminals (1-20), #10 and #20 are punctuation
        terms = [l0.add_terminal(text=str(i), punct=(i % 10 == 0))
                 for i in range(1, 21)]

        # First parallel scene, stretching on terminals 1-10
        # The dashed edge tags (e.g. -C, C-) mean discontiguous units
        # [PS [D [E 0] [C- 1] [E 2] [-C 3]]
        #     [A- 4] [P- 5 6] [-A 7] [F 8] [-P [U 9]]]
        # In addition, D takes P as a remote G
        ps1 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
        d1 = l1.add_fnode(ps1, layer1.EdgeTags.Adverbial)
        e1 = l1.add_fnode(d1, layer1.EdgeTags.Elaborator)
        c1 = l1.add_fnode(d1, layer1.EdgeTags.Center)
        e2 = l1.add_fnode(d1, layer1.EdgeTags.Elaborator)
        a1 = l1.add_fnode(ps1, layer1.EdgeTags.Participant)
        p1 = l1.add_fnode(ps1, layer1.EdgeTags.Process)
        f1 = l1.add_fnode(ps1, layer1.EdgeTags.Function)
        l1.add_remote(d1, layer1.EdgeTags.Ground, p1)
        e1.add(layer1.EdgeTags.Terminal, terms[0])
        c1.add(layer1.EdgeTags.Terminal, terms[1])
        e2.add(layer1.EdgeTags.Terminal, terms[2])
        c1.add(layer1.EdgeTags.Terminal, terms[3])
        a1.add(layer1.EdgeTags.Terminal, terms[4])
        p1.add(layer1.EdgeTags.Terminal, terms[5])
        p1.add(layer1.EdgeTags.Terminal, terms[6])
        a1.add(layer1.EdgeTags.Terminal, terms[7])
        f1.add(layer1.EdgeTags.Terminal, terms[8])
        l1.add_punct(p1, terms[9])

        # Second parallel scene, stretching on terminals 11-14 + 18-20
        # [PS- [D IMPLICIT] [G IMPLICIT] [P 10 11 12 13]]
        # [-PS [A 17 18 [U 19]]]
        ps2 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
        l1.add_fnode(ps2, layer1.EdgeTags.Adverbial, implicit=True)
        l1.add_fnode(ps2, layer1.EdgeTags.Ground, implicit=True)
        p2 = l1.add_fnode(ps2, layer1.EdgeTags.Process)
        a2 = l1.add_fnode(ps2, layer1.EdgeTags.Participant)
        p2.add(layer1.EdgeTags.Terminal, terms[10])
        p2.add(layer1.EdgeTags.Terminal, terms[11])
        p2.add(layer1.EdgeTags.Terminal, terms[12])
        p2.add(layer1.EdgeTags.Terminal, terms[13])
        a2.add(layer1.EdgeTags.Terminal, terms[17])
        a2.add(layer1.EdgeTags.Terminal, terms[18])
        l1.add_punct(a2, terms[19])

        # Third parallel scene, stretching on terminals 15-17
        # [PS [P IMPLICIT] 14 [A 15 16]]
        ps3 = l1.add_fnode(None, layer1.EdgeTags.ParallelScene)
        ps3.add(layer1.EdgeTags.Terminal, terms[14])
        l1.add_fnode(ps3, layer1.EdgeTags.Process, implicit=True)
        a3 = l1.add_fnode(ps3, layer1.EdgeTags.Participant)
        a3.add(layer1.EdgeTags.Terminal, terms[15])
        a3.add(layer1.EdgeTags.Terminal, terms[16])

        return p

    @staticmethod
    def load_xml(path):
        """XML file path ==> root element
        :param path: path to XML file
        """
        with open(path, encoding="utf-8") as f:
            return ETree.ElementTree().parse(f)
