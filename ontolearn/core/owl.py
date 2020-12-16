import operator
from abc import ABCMeta
from functools import reduce
from typing import Dict, Iterable, Tuple, overload

from ontolearn.owlapy.model import OWLClass, OWLReasoner
from ontolearn.owlapy.render import DLSyntaxRenderer
from ontolearn.owlapy.utils import NamedFixedSet, iter_bits


class ClassHierarchy(metaclass=ABCMeta):
    """Representation of a class hierarchy

    Args:
        hierarchy_down: a downwards hierarchy given as a mapping of Class to sub-classes
        reasoner: alternatively, a reasoner whose root_ontology is queried for classes and sub-classes
        """
    __slots__ = '_cls_enc', '_parents_map', '_parents_map_trans', '_children_map', '_children_map_trans', \
                '_leaf_set', '_root_set', \
                # '_eq_set'

    _cls_enc: NamedFixedSet[OWLClass]
    _parents_map: Dict[int, int]  # Class => classes
    _children_map: Dict[int, int]  # Class => classes
    _children_map_trans: Dict[int, int]  # Class => classes
    # _eq_set: Dict[int, int]  # Class => classes  # TODO
    _root_set: int  # root classes
    _leaf_set: int  # leaf classes

    @overload
    def __init__(self, hierarchy_down: Iterable[Tuple[OWLClass, Iterable[OWLClass]]]): ...

    @overload
    def __init__(self, reasoner: OWLReasoner): ...

    def __init__(self, arg):
        if isinstance(arg, OWLReasoner):
            hier_down_gen = ((_, arg.sub_classes(_, direct=True))
                             for _ in arg.get_root_ontology().classes_in_signature())
            self._init(hier_down_gen)
        else:
            self._init(arg)

    @staticmethod
    def restrict(hierarchy: 'ClassHierarchy', *, remove: Iterable[OWLClass] = None, allow: Iterable[OWLClass] = None) \
            -> 'ClassHierarchy':
        """Restrict a given class hierarchy to a set of allowed/removed classes

        Args:
            hierarchy: an existing Class hierarchy to restrict
            remove: set of Classes which should be ignored
            allow: set of Classes which should be used

        Returns:
            the restricted class hierarchy

        """
        remove_set = frozenset(remove) if remove is not None else None
        allow_set = frozenset(allow) if allow is not None else None

        def _filter(_: OWLClass):
            if remove_set is None or _ not in remove_set:
                if allow_set is None or _ in allow_set:
                    return True
            return False

        _gen = ((_, filter(_filter, hierarchy.children(_, direct=False)))
                for _ in filter(_filter, hierarchy.items()))

        return ClassHierarchy(_gen)

    def restrict_and_copy(self, *, remove: Iterable[OWLClass] = None, allow: Iterable[OWLClass] = None) \
            -> 'ClassHierarchy':
        """Restrict this class hierarchy

        See restrict for more info
        """
        return ClassHierarchy.restrict(self, remove=remove, allow=allow)

    def _init(self, hierarchy_down: Iterable[Tuple[OWLClass, Iterable[OWLClass]]]) -> None:
        self._parents_map_trans = dict()
        self._children_map_trans = dict()
        # self._eq_set = dict()

        cls_to_sub_classes = dict(hierarchy_down)
        enc = self._cls_enc = NamedFixedSet(OWLClass, cls_to_sub_classes.keys())

        for cls, sub_it in cls_to_sub_classes.items():
            cls_enc = enc(cls)
            self._children_map_trans[cls_enc] = enc(sub_it)
            self._parents_map_trans[cls_enc] = 0  # create empty parent entry for all classes

        del cls_to_sub_classes  # exhausted

        # calculate transitive children
        for cls_enc in self._children_map_trans:
            _children_transitive(self._children_map_trans, cls_enc=cls_enc, seen_set=0)

        # TODO handling of eq_sets
        # sccs = list(_strongly_connected_components(self._children_map_trans))
        # for scc in sccs:
        #     sub_classes = 0
        #     for cls_enc in iter_bits(scc):
        #         self._eq_set[cls_enc] = scc
        #         sub_classes |= self._children_map_trans[cls_enc]
        #         del self._children_map_trans[cls_enc]
        #         del self._parents_map_trans[cls_enc]
        #     self._children_map_trans[scc] = sub_classes
        #     self._parents_map_trans[scc] = 0

        # fill transitive parents
        for cls_enc, sub_classes_enc in self._children_map_trans.items():
            for sub_enc in iter_bits(sub_classes_enc):
                self._parents_map_trans[sub_enc] |= cls_enc

        self._children_map, self._leaf_set = _reduce_transitive(self._children_map_trans, self._parents_map_trans)
        self._parents_map, self._root_set = _reduce_transitive(self._parents_map_trans, self._children_map_trans)

    def parents(self, entity: OWLClass, direct: bool = True) -> Iterable[OWLClass]:
        """Parents of a class

        Args:
            entity: class for which to query parent classes
            direct: False to return transitive parents

        Returns:
            super-classes

        """
        if not direct:
            yield from self._cls_enc(self._parents_map_trans[self._cls_enc(entity)])
        else:
            yield from self._cls_enc(self._parents_map[self._cls_enc(entity)])

    def children(self, entity: OWLClass, direct: bool = True) -> Iterable[OWLClass]:
        """Children of a class

        Args:
            entity: class for which to query child classes
            direct: False to return transitive children

        Returns:
            sub-classes

        """
        if not direct:
            yield from self._cls_enc(self._children_map_trans[self._cls_enc(entity)])
        else:
            yield from self._cls_enc(self._children_map[self._cls_enc(entity)])

    def siblings(self, entity: OWLClass) -> Iterable[OWLClass]:
        seen_set = {entity}
        for parent in self.parents(entity, direct=True):
            for sibling in self.children(parent, direct=True):
                if sibling not in seen_set:
                    yield sibling
                    seen_set.add(sibling)

    def items(self) -> Iterable[OWLClass]:
        for _, i in self._cls_enc.items():
            yield i

    def roots(self) -> Iterable[OWLClass]:
        yield from self._cls_enc(self._root_set)


def _children_transitive(map_trans: Dict[int, int], cls_enc: int, seen_set: int):
    """add transitive links to map_trans

    Note:
        changes map_trans

    Args:
        map_trans: map to which transitive links are added
        cls_enc: encoded class in map_trans for which to add transitive sub-classes

    """
    sub_classes_enc = map_trans[cls_enc]
    for sub_enc in iter_bits(sub_classes_enc):
        if not sub_enc & seen_set:
            _children_transitive(map_trans, sub_enc, seen_set | cls_enc)
            seen_set = seen_set | sub_enc | map_trans[sub_enc]
            map_trans[cls_enc] |= map_trans[sub_enc]


def _reduce_transitive(map: Dict[int, int], map_inverse: Dict[int, int]) -> Tuple[Dict[int, int], int]:
    """Remove all transitive links

    Takes a downward hierarchy and an upward hierarchy with transitive links, and removes all links that can be
    implicitly detected since they are transitive

    Args:
         map: downward hierarchy with all transitive links, from Class => sub-classes
         map_inverse: upward hierarchy with all transitive links, from Class => super-classes

    Returns:
        thin map with only direct sub-classes
        set of classes without sub-classes

    """
    result_map: Dict[int, int] = dict()
    leaf_set = 0
    for enc, set_enc in map.items():
        direct_set = 0
        for item_enc in iter_bits(set_enc):
            if not map_inverse[item_enc] & (set_enc ^ item_enc):
                direct_set |= item_enc
        result_map[enc] = direct_set
        if not direct_set:
            leaf_set |= enc
    return result_map, leaf_set


def _strongly_connected_components(graph: Dict[int, int]) -> Iterable[int]:
    """Strongly connected component algorithm

    Args:
        graph: Directed Graph dictionary, vertex => set of vertices (there is an edge from v to each V)

    Returns:
        the strongly connected components

    Author: Mario Alviano
    Source: https://github.com/alviano/python/blob/master/rewrite_aggregates/scc.py
    Licence: GPL-3.0
    """
    identified = 0
    stack = []
    index = {}
    boundaries = []

    def dfs(v):
        nonlocal identified
        index[v] = len(stack)
        stack.append(v)
        boundaries.append(index[v])

        for w in iter_bits(graph[v]):
            if w not in index:
                yield from dfs(w)
            elif not w & identified:
                while index[w] < boundaries[-1]:
                    boundaries.pop()

        if boundaries[-1] == index[v]:
            boundaries.pop()
            scc = reduce(operator.or_, stack[index[v]:])
            del stack[index[v]:]
            identified |= scc
            yield scc

    for v in graph:
        if v not in index:
            yield from dfs(v)