import operator
from abc import ABCMeta, abstractmethod
from functools import reduce
from typing import Dict, Iterable, Tuple, overload, TypeVar, Generic, Type, Union, cast

from ontolearn.owlapy import HasIRI
from ontolearn.owlapy.model import OWLClass, OWLReasoner, OWLObjectProperty, OWLDataProperty
from ontolearn.owlapy.render import DLSyntaxRenderer
from ontolearn.owlapy.utils import NamedFixedSet, iter_bits

_S = TypeVar('_S', bound=HasIRI)
_U = TypeVar('_U', bound='AbstractHierarchy')


class AbstractHierarchy(Generic[_S], metaclass=ABCMeta):
    """Representation of an abstract hierarchy which can be used for classes or properties

    Args:
        hierarchy_down: a downwards hierarchy given as a mapping of Entities to sub-entities
        reasoner: alternatively, a reasoner whose root_ontology is queried for entities
        """
    __slots__ = '_Type', '_ent_enc', '_parents_map', '_parents_map_trans', '_children_map', '_children_map_trans', \
                '_leaf_set', '_root_set', \
        # '_eq_set'

    _ent_enc: NamedFixedSet[_S]
    _parents_map: Dict[int, int]  # Entity => parent entities
    _parents_map_trans: Dict[int, int]  # Entity => parent entities
    _children_map: Dict[int, int]  # Entity => child entities
    _children_map_trans: Dict[int, int]  # Entity => child entities
    # _eq_set: Dict[int, int]  # Entity => equivalent entities  # TODO
    _root_set: int  # root entities
    _leaf_set: int  # leaf entities

    @overload
    def __init__(self, factory: Type[_S], hierarchy_down: Iterable[Tuple[_S, Iterable[_S]]]):
        ...

    @overload
    def __init__(self, factory: Type[_S], reasoner: OWLReasoner):
        ...

    @abstractmethod
    def __init__(self, factory: Type[_S], arg):
        self._Type = factory
        if isinstance(arg, OWLReasoner):
            hier_down_gen = self.hierarchy_down_generator(arg)
            self._init(hier_down_gen)
        else:
            self._init(arg)

    @abstractmethod
    def hierarchy_down_generator(self, reasoner: OWLReasoner) -> Iterable[Tuple[_S, Iterable[_S]]]:
        """Generate the suitable downwards hierarchy based on the reasoner"""
        pass

    @staticmethod
    def restrict(hierarchy: _U, *, remove: Iterable[_S] = None, allow: Iterable[_S] = None) \
            -> _U:
        """Restrict a given hierarchy to a set of allowed/removed entities

        Args:
            hierarchy: an existing Entity hierarchy to restrict
            remove: set of entities which should be ignored
            allow: set of entities which should be used

        Returns:
            the restricted hierarchy

        """
        remove_set = frozenset(remove) if remove is not None else None
        allow_set = frozenset(allow) if allow is not None else None

        def _filter(_: _S):
            if remove_set is None or _ not in remove_set:
                if allow_set is None or _ in allow_set:
                    return True
            return False

        _gen = ((_, filter(_filter, hierarchy.children(_, direct=False)))
                for _ in filter(_filter, hierarchy.items()))

        return ClassHierarchy(_gen)

    def restrict_and_copy(self: _U, *, remove: Iterable[_S] = None, allow: Iterable[_S] = None) \
            -> _U:
        """Restrict this hierarchy

        See restrict for more info
        """
        return type(self).restrict(self, remove=remove, allow=allow)

    def _init(self, hierarchy_down: Iterable[Tuple[_S, Iterable[_S]]]) -> None:
        self._parents_map_trans = dict()
        self._children_map_trans = dict()
        # self._eq_set = dict()

        ent_to_sub_entities = dict(hierarchy_down)
        enc = self._ent_enc = NamedFixedSet(self._Type, ent_to_sub_entities.keys())

        for ent, sub_it in ent_to_sub_entities.items():
            ent_enc = enc(ent)
            self._children_map_trans[ent_enc] = enc(sub_it)
            self._parents_map_trans[ent_enc] = 0  # create empty parent entry for all classes

        del ent_to_sub_entities  # exhausted

        # calculate transitive children
        for ent_enc in self._children_map_trans:
            _children_transitive(self._children_map_trans, ent_enc=ent_enc, seen_set=0)

        # TODO handling of eq_sets
        # sccs = list(_strongly_connected_components(self._children_map_trans))
        # for scc in sccs:
        #     sub_entities = 0
        #     for ent_enc in iter_bits(scc):
        #         self._eq_set[ent_enc] = scc
        #         sub_entities |= self._children_map_trans[ent_enc]
        #         del self._children_map_trans[ent_enc]
        #         del self._parents_map_trans[ent_enc]
        #     self._children_map_trans[scc] = sub_entities
        #     self._parents_map_trans[scc] = 0

        # fill transitive parents
        for ent_enc, sub_entities_enc in self._children_map_trans.items():
            for sub_enc in iter_bits(sub_entities_enc):
                self._parents_map_trans[sub_enc] |= ent_enc

        self._children_map, self._leaf_set = _reduce_transitive(self._children_map_trans, self._parents_map_trans)
        self._parents_map, self._root_set = _reduce_transitive(self._parents_map_trans, self._children_map_trans)

    def parents(self, entity: _S, direct: bool = True) -> Iterable[_S]:
        """Parents of an entity

        Args:
            entity: entity for which to query parent entities
            direct: False to return transitive parents

        Returns:
            super-entities

        """
        if not direct:
            yield from self._ent_enc(self._parents_map_trans[self._ent_enc(entity)])
        else:
            yield from self._ent_enc(self._parents_map[self._ent_enc(entity)])

    def children(self, entity: _S, direct: bool = True) -> Iterable[_S]:
        """Children of an entitiy

        Args:
            entity: entity for which to query child entities
            direct: False to return transitive children

        Returns:
            sub-entities

        """
        if not direct:
            yield from self._ent_enc(self._children_map_trans[self._ent_enc(entity)])
        else:
            yield from self._ent_enc(self._children_map[self._ent_enc(entity)])

    def siblings(self, entity: _S) -> Iterable[_S]:
        seen_set = {entity}
        for parent in self.parents(entity, direct=True):
            for sibling in self.children(parent, direct=True):
                if sibling not in seen_set:
                    yield sibling
                    seen_set.add(sibling)

    def items(self) -> Iterable[_S]:
        for _, i in self._ent_enc.items():
            yield i

    def roots(self) -> Iterable[_S]:
        yield from self._ent_enc(self._root_set)


class ClassHierarchy(AbstractHierarchy[OWLClass]):
    """Representation of a class hierarchy

    Args:
        hierarchy_down: a downwards hierarchy given as a mapping of Class to sub-classes
        reasoner: alternatively, a reasoner whose root_ontology is queried for classes and sub-classes
        """

    def hierarchy_down_generator(self, reasoner: OWLReasoner) -> Iterable[Tuple[OWLClass, Iterable[OWLClass]]]:
        return ((_, reasoner.sub_classes(_, direct=True))
                for _ in reasoner.get_root_ontology().classes_in_signature())

    @overload
    def __init__(self, hierarchy_down: Iterable[Tuple[OWLClass, Iterable[OWLClass]]]): ...

    @overload
    def __init__(self, reasoner: OWLReasoner): ...

    def __init__(self, arg):
        super().__init__(OWLClass, arg)


class ObjectPropertyHierarchy(AbstractHierarchy[OWLObjectProperty]):
    def hierarchy_down_generator(self, reasoner: OWLReasoner) \
            -> Iterable[Tuple[OWLObjectProperty, Iterable[OWLObjectProperty]]]:
        return ((_, map(lambda _: cast(OWLObjectProperty, _),
                        filter(lambda _: isinstance(_, OWLObjectProperty),
                               reasoner.sub_object_properties(_, direct=True))))
                for _ in reasoner.get_root_ontology().object_properties_in_signature())

    @overload
    def __init__(self, hierarchy_down: Iterable[Tuple[OWLObjectProperty, Iterable[OWLObjectProperty]]]): ...

    @overload
    def __init__(self, reasoner: OWLReasoner): ...

    def __init__(self, arg):
        super().__init__(OWLObjectProperty, arg)


class DataPropertyHierarchy(AbstractHierarchy[OWLDataProperty]):
    def hierarchy_down_generator(self, reasoner: OWLReasoner) \
            -> Iterable[Tuple[OWLDataProperty, Iterable[OWLDataProperty]]]:
        return ((_, reasoner.sub_data_properties(_, direct=True))
                for _ in reasoner.get_root_ontology().data_properties_in_signature())

    @overload
    def __init__(self, hierarchy_down: Iterable[Tuple[OWLDataProperty, Iterable[OWLDataProperty]]]): ...

    @overload
    def __init__(self, reasoner: OWLReasoner): ...

    def __init__(self, arg):
        super().__init__(OWLDataProperty, arg)


def _children_transitive(map_trans: Dict[int, int], ent_enc: int, seen_set: int):
    """add transitive links to map_trans

    Note:
        changes map_trans

    Args:
        map_trans: map to which transitive links are added
        ent_enc: encoded class in map_trans for which to add transitive sub-classes

    """
    sub_classes_enc = map_trans[ent_enc]
    for sub_enc in iter_bits(sub_classes_enc):
        if not sub_enc & seen_set:
            _children_transitive(map_trans, sub_enc, seen_set | ent_enc)
            seen_set = seen_set | sub_enc | map_trans[sub_enc]
            map_trans[ent_enc] |= map_trans[sub_enc]


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
