"""
The evaluation library for UCCA layer 1.
v1.2
2016-12-25: move common Fs to root before evaluation
2017-01-04: flatten centers, do not add 1 (for root) to mutual
2017-01-16: fix bug in moving common Fs
"""
from collections import Counter, defaultdict, OrderedDict
from operator import attrgetter

from ucca import layer0, layer1
from ucca.constructions import extract_edges, get_by_names, PRIMARY, DEFAULT
from ucca.layer1 import EdgeTags, NodeTags

UNLABELED = "unlabeled"
WEAK_LABELED = "weak_labeled"
LABELED = "labeled"

EVAL_TYPES = (LABELED, UNLABELED, WEAK_LABELED)

# Pairs that are considered as equivalent for the purposes of evaluation
EQUIV = ((EdgeTags.Process, EdgeTags.State),
         (EdgeTags.ParallelScene, EdgeTags.Center),
         (EdgeTags.Connector, EdgeTags.Linker),
         (EdgeTags.Function, EdgeTags.Relator))


def flatten_centers(p):
    """
    Whenever there are Cs inside Cs, remove the external C.
    """
    for unit in p.layer(layer1.LAYER_ID).all:
        if unit.tag == NodeTags.Foundational and unit.ftag == EdgeTags.Center and \
                                len(unit.centers) == len(unit.fparent.centers) == 1:
            for e in unit.incoming:
                if e.attrib.get("remote"):
                    e.parent.add(e.tag, unit.centers[0], edge_attrib=e.attrib)
            for e in unit.outgoing:
                unit.fparent.add(e.tag, e.child, edge_attrib=e.attrib)
            unit.destroy()


def move_functions(p1, p2):
    """
    Move any common Fs to the root
    """
    f1, f2 = [{get_yield(u): u for u in p.layer(layer1.LAYER_ID).all
               if u.tag == NodeTags.Foundational and u.ftag == EdgeTags.Function} for p in (p1, p2)]
    for positions in f1.keys() & f2.keys():
        for (p, unit) in ((p1, f1[positions]), (p2, f2[positions])):
            for parent in unit.parents:
                tag = unit.ftag
                parent.remove(unit)
                p.layer(layer1.LAYER_ID).heads[0].add(tag, unit)


def get_text(p, positions):
    l0 = p.layer(layer0.LAYER_ID)
    return [l0.by_position(i).text for i in range(1, len(l0.all) + 1) if i in positions]


def create_passage_yields(p, constructions=None, reference=None, verbose=False):
    """
    :returns dict: Construction ->
                   dict: set of terminal indices (excluding punctuation) ->
                         list of edges of the Construction whose yield (excluding remotes and punctuation) is that set
    """
    yield_tags = OrderedDict()
    for construction, edges in extract_edges(
            p, constructions=constructions, reference=reference, verbose=verbose).items():
        yield_tags[construction] = {}
        for edge in edges:
            yield_tags[construction].setdefault(get_yield(edge.child), []).append(edge.tag)
    return yield_tags


def get_yield(unit):
    try:
        return frozenset(t.position for t in unit.get_terminals(punct=False))
    except ValueError:
        return frozenset()


def find_mutuals(m1, m2, eval_type):
    mutual_tags = dict()
    error_counter = Counter()
    for y in m1.keys() & m2.keys():
        if eval_type == UNLABELED:
            mutual_tags[y] = ()
        else:
            tags1 = set(m1[y])
            tags2 = set(m2[y])
            if eval_type == WEAK_LABELED:
                tags1 = expand_equivalents(tags1)
            intersection = tags1 & tags2
            if intersection:  # non-empty intersection
                mutual_tags[y] = intersection
            else:
                error_counter[(str(sorted(tags1)), str(sorted(tags2)))] += 1
    return mutual_tags, error_counter


def print_tags_and_text(p, yield_tags):
    for y, tags in sorted(yield_tags.items(), key=lambda x: min(x[0])):
        text = " ".join(get_text(p, y))
        print((",".join(sorted(filter(None, tags))) + ": " + text) if tags else text)


def expand_equivalents(tag_set):
    """
    Returns a set of all the tags in the tag set or those equivalent to them
    :param tag_set: set of tags (strings) to expand
    """
    return tag_set.union(t1 for t in tag_set for pair in EQUIV for t1 in pair if t in pair and t != t1)


class Evaluator(object):
    def __init__(self, verbose, constructions, units, fscore, errors):
        """
        :param verbose: whether to print the scores
        :param constructions: names of construction types to include in the evaluation
        :param units: whether to calculate and print the mutual and exclusive units in the passages
        :param fscore: whether to find and return the scores
        :param errors: whether to calculate and print the confusion matrix of errors
        """
        self.verbose = verbose
        self.constructions = list(DEFAULT.values()) + [c for c in get_by_names(constructions)
                                                       if c not in DEFAULT.values()]
        self.units = units
        self.fscore = fscore
        self.errors = errors

    def get_scores(self, p1, p2, eval_type):
        """
        prints the relevant statistics and f-scores. eval_type can be 'unlabeled', 'labeled' or 'weak_labeled'.
        calculates a set of all the yields such that both passages have a unit with that yield.
        :param p1: passage to compare
        :param p2: reference passage object
        :param eval_type: evaluation type to use, out of EVAL_TYPES
        1. UNLABELED: it doesn't matter what labels are there.
        2. LABELED: also requires tag match (if there are multiple units with the same yield, requires one match)
        3. WEAK_LABELED: also requires weak tag match (if there are multiple units with the same yield,
                         requires one match)
        :returns EvaluatorResults object if self.fscore is True, otherwise None
        """
        maps = [defaultdict(dict), create_passage_yields(p2, self.constructions)]
        mutual = defaultdict(dict)
        error_counters = defaultdict(Counter)
        if p1 is not None:
            maps[0] = create_passage_yields(p1, self.constructions, reference=p2)
            for construction, yield_tags1 in maps[0].items():
                yield_tags2 = maps[1][construction]
                mutual[construction], error_counters[construction] = find_mutuals(yield_tags1, yield_tags2, eval_type)

        if self.verbose:
            print("Evaluation type: (" + eval_type + ")")

        only = [{c: {y: tags for y, tags in yt.items() if y not in mutual[c]} for c, yt in m.items()} for m in maps]
        if self.verbose and self.units and p1 is not None:
            print("==> Mutual Units:")
            print_tags_and_text(p1, mutual[PRIMARY])
            print("==> Only in guessed:")
            print_tags_and_text(p1, only[0][PRIMARY])
            print("==> Only in reference:")
            print_tags_and_text(p2, only[1][PRIMARY])

        res = None
        if self.fscore:
            res = EvaluatorResults((c, SummaryStatistics(len(mutual[c]), len(only[0][c]), len(only[1][c])))
                                   for c in self.constructions)
            if self.verbose:
                res.print()

        if self.verbose and self.errors and error_counters:
            print("\nConfusion Matrix:\n")
            for error, freq in error_counters[PRIMARY].most_common():
                print(error[0], "\t", error[1], "\t", freq)

        return res


class Scores(object):
    def __init__(self, evaluator_results):
        """
        :param evaluator_results: dict: eval_type -> EvaluatorResults
        """
        self.evaluators = dict(evaluator_results)

    def average_f1(self, mode=LABELED):
        """
        Calculate the average F1 score across primary and remote edges
        :param mode: LABELED, UNLABELED or WEAK_LABELED
        :return: a single number, the average F1
        """
        return float(self.evaluators[mode].aggregate_default().f1)

    @staticmethod
    def aggregate(scores):
        """
        Aggregate multiple Scores instances
        :param scores: iterable of Scores
        :return: new Scores with aggregated scores
        """
        evaluators = [s.evaluators for s in scores]
        return Scores((t, EvaluatorResults.aggregate(filter(None, (e.get(t) for e in evaluators)))) for t in EVAL_TYPES)

    def print(self, **kwargs):
        for eval_type in EVAL_TYPES:
            evaluator = self.evaluators.get(eval_type)
            if evaluator is not None:
                print("Evaluation type: (" + eval_type + ")", **kwargs)
                evaluator.print(**kwargs)

    def fields(self):
        e = self.evaluators[LABELED]
        return ["%.3f" % float(getattr(x, y)) for x in e.results.values() for y in ("p", "r", "f1")]

    def titles(self):
        return self.field_titles(self.evaluators[LABELED].results.keys())

    @staticmethod
    def field_titles(constructions=DEFAULT):
        return ["%s_labeled_%s" % (x, y) for x in constructions for y in ("precision", "recall", "f1")]


class EvaluatorResults(object):
    def __init__(self, results, default=None):
        """
        :param results: dict: Construction -> SummaryStatistics
        :param default: map of default constructions (default is primary and remote)
        """
        self.results = OrderedDict(results)
        self.default = default or DEFAULT

    def print(self, **kwargs):
        for construction, stats in self.results.items():
            if len(self.results) > 1:
                print("\n%s:" % construction.description, **kwargs)
            stats.print(**kwargs)
        print(**kwargs)

    @classmethod
    def aggregate(cls, results):
        """
        :param results: iterable of EvaluatorResults
        :return: new EvaluatorResults with aggregates scores
        """
        collected = OrderedDict()
        default = OrderedDict()
        for evaluator_results in results:
            for c, r in evaluator_results.results.items():
                collected.setdefault(c, []).append(r)
            default.update(evaluator_results.default)
        return EvaluatorResults(((c, SummaryStatistics.aggregate(r)) for c, r in collected.items()), default=default)

    def aggregate_default(self):
        """
        Aggregate primary and remote SummaryStatistics in this EvaluatorResults instance
        :return: SummaryStatistics object representing aggregation over primary and remote
        """
        try:
            return SummaryStatistics.aggregate([self.results[c] for c in self.default.values()])
        except KeyError as e:
            raise ValueError("Default constructions missing from evaluation results: " +
                             ", ".join(map(str, self.results.keys()))) from e


class SummaryStatistics(object):
    def __init__(self, num_matches, num_only_guessed, num_only_ref):
        self.num_matches = num_matches
        self.num_only_guessed = num_only_guessed
        self.num_only_ref = num_only_ref
        self.num_guessed = num_matches + num_only_guessed
        self.num_ref = num_matches + num_only_ref
        self.p = 1.0 if self.num_guessed == 0 else 1.0 * num_matches / self.num_guessed
        self.r = 1.0 if self.num_ref == 0 else 1.0 * num_matches / self.num_ref
        self.f1 = 0.0 if 0.0 in (self.p, self.r) else 2.0 * self.p * self.r / float(self.p + self.r)

    def print(self, **kwargs):
        print("Precision: {:.3} ({}/{})".format(self.p, self.num_matches, self.num_guessed), **kwargs)
        print("Recall: {:.3} ({}/{})".format(self.r, self.num_matches, self.num_ref), **kwargs)
        print("F1: {:.3}".format(self.f1), **kwargs)

    @classmethod
    def aggregate(cls, stats):
        """
        :param stats: iterable of SummaryStatistics
        :return: new SummaryStatistics with aggregated scores
        """
        return SummaryStatistics(*map(sum, [map(attrgetter(attr), stats)
                                            for attr in ("num_matches", "num_only_guessed", "num_only_ref")]))


def evaluate(guessed, ref, converter=None, verbose=False, constructions=DEFAULT,
             units=False, fscore=True, errors=False, normalize=True, **kwargs):
    """
    Compare two passages and return requested diagnostics and scores, possibly printing them too.
    NOTE: since normalize=True by default, this method is destructive: it modifies the given passages before evaluation.
    :param guessed: Passage object to evaluate
    :param ref: reference Passage object to compare to
    :param converter: optional function to apply to passages before evaluation
    :param verbose: whether to print the results
    :param constructions: names of construction types to include in the evaluation
    :param units: whether to evaluate common units
    :param fscore: whether to compute precision, recall and f1 score
    :param errors: whether to print the mistakes
    :param normalize: flatten centers and move common functions to root before evaluation - modifies passages
    :return: Scores object
    """
    del kwargs
    if converter is not None:
        guessed = converter(guessed)
        ref = converter(ref)
    if normalize:
        for passage in (guessed, ref):
            flatten_centers(passage)  # flatten Cs inside Cs
        move_functions(guessed, ref)  # move common Fs to be under the root

    evaluator = Evaluator(verbose, constructions, units, fscore, errors)
    return Scores((evaluation_type, evaluator.get_scores(guessed, ref, evaluation_type))
                  for evaluation_type in EVAL_TYPES)
