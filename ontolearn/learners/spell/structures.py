from dataclasses import dataclass
import functools
from typing import Any

from lxml import etree

from ontolearn.learners.spell import o2p_ontology, o2p_owl_parser
from ontolearn.learners.spell.o2p_ontology import (
    ClassIdentifier,
    Intersection,
    NameFactory,
    Ontology,
    Restriction,
    SomeValues,
    SubClassOf,
    Thing,
    TopClass,
)

namespaces = {
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
}

Signature = tuple[list[str], list[str]]


@dataclass(slots=True)
class Structure:
    max_ind: int
    cn_ext: dict[str, set[int]]
    rn_ext: dict[int, set[tuple[int, str]]]
    indmap: dict[str, int]
    nsmap: dict[str | None, str]


def ind(A: Structure) -> range:
    return range(A.max_ind)


def conceptnames(sigma: Signature) -> list[str]:
    return sigma[0]


def rolenames(sigma: Signature) -> list[str]:
    return sigma[1]


def conceptname_ext(A: Structure, cn: str) -> set[int]:
    return A.cn_ext[cn]


def expand_namespace(namespace: str, item: str):
    return "{%s}%s" % (namespaces.get(namespace), item)


@functools.cache
def tag2name(tag: str):
    q = etree.QName(tag)
    res = "{}{}".format(q.namespace, q.localname)
    return res


def name2sparql(name: str):
    name = name.replace("{", "")
    name = name.replace("}", "")
    return "<{}>".format(name)


def expand_curie(curie, nsmap):
    assert ":" in curie
    s = curie.split(":")
    assert s[0] in nsmap
    s[0] = nsmap[s[0]]
    return "".join(s)


def map_ind_name(A: Structure, name: str) -> int:
    if "://" not in name and ":" in name:
        name = expand_curie(name, A.nsmap)
    return A.indmap[name]


def add_ns(n: str):
    a = n.split("/")
    an = a[: len(a) - 1]

    return "{{{}/}}{}".format("/".join(an), a[len(a) - 1])


class ABoxBuilder:
    A: Structure
    indmap: dict[str, int]
    role_names = set()

    def __init__(self):
        self.indmap = {}
        self.A = Structure(max_ind=0, cn_ext={}, rn_ext={}, indmap={}, nsmap={})

    def map_ind(self, a: str):
        if a not in self.indmap:
            n = self.A.max_ind
            self.indmap[a] = n
            self.A.max_ind += 1
            self.A.rn_ext[n] = set()
            self.A.indmap[a] = n

        return self.indmap[a]

    def declare_cn(self, cn):
        assert "{" not in cn
        if cn not in self.A.cn_ext:
            self.A.cn_ext[cn] = set()
        return

    def declare_rn(self, rn):
        self.role_names.add(rn)
        return

    def concept_assertion(self, a: int, concept: str):
        self.declare_cn(concept)
        self.A.cn_ext[concept].add(a)

    def role_assertion(self, idx1: int, ind2: str, role: str):
        assert "{" not in role
        idx2 = self.indmap.get(ind2)

        if not idx2:
            idx2 = self.map_ind(ind2)

        self.A.rn_ext[idx1].add((idx2, role))


tag_onto = expand_namespace("owl", "Ontology")
tag_ni = expand_namespace("owl", "NamedIndividual")
tag_type = expand_namespace("rdf", "type")
tag_class = expand_namespace("owl", "Class")
tag_thing = expand_namespace("owl", "Thing")
tag_object_prop = expand_namespace("owl", "ObjectProperty")
tag_data_prop = expand_namespace("owl", "DatatypeProperty")
tag_annotation_prop = expand_namespace("owl", "AnnotationProperty")
attr_resource = expand_namespace("rdf", "resource")
attr_about = expand_namespace("rdf", "about")
attr_datatype = expand_namespace("rdf", "datatype")

def make_res_absolute(nsmap: dict[Any, str], res: str) -> str:
    if res[0] == "#":
        assert None in nsmap
        return nsmap[None] + res[1:]
    else:
        return res


def load_owl(file: str):
    reader = o2p_owl_parser.OWLReader("")
    onto = o2p_ontology.Ontology()

    facts = 0
    nsmap = {}
    abox = ABoxBuilder()
    num = len(abox.indmap)
    print("Loaded {} individuals".format(num), end="\r")

    abox.declare_cn("http://www.w3.org/2002/07/owl#NamedIndividual")

    for _, elem in etree.iterparse(file, events=("end",), remove_blank_text=True):
        # We are only interested in top-level statements
        if elem.getparent() != None and elem.getparent().getparent() != None:
            continue
        if elem.tag == tag_onto:
            nsmap = elem.nsmap
        if elem.tag == tag_class:
            if attr_about in elem.attrib:
                abox.declare_cn(make_res_absolute(nsmap, elem.attrib[attr_about]))
            for r in reader.parse_rule(elem):
                onto.add_rule(r)
        elif elem.tag == tag_object_prop:
            abox.declare_rn(make_res_absolute(nsmap, elem.attrib[attr_about]))
            onto.add_property(reader.parse_property(elem))
            elem.clear()
        elif elem.tag == tag_data_prop:
            # TODO: handle dataproperties here
            elem.clear()
        elif elem.tag == tag_annotation_prop:
            elem.clear()
        elif (
            elem.tag == tag_ni
            or elem.tag == tag_thing
            or (attr_about in elem.attrib and elem.tag != tag_onto)
        ):
            a = make_res_absolute(nsmap, elem.attrib[attr_about])
            ind_idx = abox.map_ind(a)

            if elem.tag != tag_ni and elem.tag != tag_thing:
                facts += 1
                abox.concept_assertion(ind_idx, tag2name(elem.tag))

            num = len(abox.indmap)
            if num % 100000 == 0:
                print("\rLoaded {} individuals".format(num), end="\r")

            for child in elem:
                if child.tag in tag_type:
                    # TODO: handle complex concepts here
                    if attr_resource not in child.attrib:
                        continue
                    conceptname = make_res_absolute(nsmap, child.attrib[attr_resource])
                    facts += 1
                    abox.concept_assertion(ind_idx, conceptname)
                elif attr_datatype in child.attrib:
                    # TODO: handle dataproperties here
                    continue
                elif attr_resource in child.attrib:
                    role = tag2name(child.tag)
                    other = make_res_absolute(nsmap, child.attrib[attr_resource])
                    facts += 1
                    if role in abox.role_names:
                        abox.role_assertion(ind_idx, other, role)

            elem.clear()

    num = len(abox.indmap)
    abox.A.nsmap = nsmap
    print("\rLoaded {} individuals and {} facts".format(num, facts))
    return onto, abox


@functools.cache
def structure_from_owl(file) -> Structure:
    onto, abox = load_owl(file)
    tbox = construct_normalized_tbox(onto)
    tbox.saturate()

    compact_canonical_model(abox, tbox)

    return abox.A


# ELTBox
class TBox:
    fresh_names: set[str]
    implic: dict[str, set[str]]
    conjs: dict[str, set[frozenset[str]]]
    top: str
    cns: set[str]
    rns: set[str]
    rBrhs: dict[str, set[tuple[str, str]]]
    rBlhs: dict[str, set[tuple[str, str]]]
    range_cn_ctr: int
    ranges: dict[str, set[str]]
    role_incs: dict[str, set[str]]

    def __init__(self, top: str):
        self.fresh_names = set()
        self.implic = {}
        self.conjs = {}

        self.top = top

        self.rBrhs = {}
        self.rBlhs = {}
        self.ranges = {}
        self.range_cn_ctr = 0

        self.role_incs = {}

        self.cns = set()
        self.rns = set()
        self.register_cn(top)
        self.register_rn(tag2name(expand_namespace("owl", "sameAs")))

    def non_empty_conjs(self):
        return {A for A in self.conjs.keys() if len(self.conjs[A]) > 0}

    def non_empty_lhs(self):
        return {A for A in self.rBlhs.keys() if len(self.rBlhs[A]) > 0}

    def non_empty_rhs(self):
        return {A for A in self.rBrhs.keys() if len(self.rBrhs[A]) > 0}

    def register_cn(self, A: str):
        if A not in self.cns:
            assert "{" not in A
            self.cns.add(A)
            self.implic[A] = set([A, self.top])
            self.conjs[A] = set()
            self.rBrhs[A] = set()
            self.rBlhs[A] = set()

    def register_rn(self, r: str):
        if r not in self.rns:
            self.rns.add(r)
            self.ranges[r] = set()
            self.role_incs[r] = set([r])

    def add_axiom1(self, A: str, B: str):
        self.register_cn(A)
        self.register_cn(B)
        self.implic[A].add(B)

    def add_axiom2(self, A1: str, A2: str, B: str):
        self.register_cn(A1)
        self.register_cn(A2)
        self.register_cn(B)
        self.conjs[B].add(frozenset([A1, A2]))

    def add_axiom3(self, A: str, r: str, B: str):
        self.register_cn(A)
        self.register_cn(B)
        self.register_rn(r)
        self.rBrhs[A].add((r, B))

    def add_axiom4(self, r: str, A: str, B: str):
        self.register_cn(A)
        self.register_cn(B)
        self.register_rn(r)
        self.rBlhs[B].add((r, A))

    def add_range_restriction(self, r: str, A: str):
        self.register_cn(A)
        self.register_rn(r)
        self.ranges[r].add(A)

    def add_role_inc(self, r: str, s: str):
        self.register_rn(r)
        self.register_rn(s)
        self.role_incs[r].add(s)

    def fresh_cn(self) -> str:
        self.range_cn_ctr += 1
        name = "Fresh#R{}".format(self.range_cn_ctr)
        self.fresh_names.add(name)
        return name

    def saturate_role_incs(self):
        for r in self.rns:
            if r not in self.role_incs:
                self.role_incs[r] = set()

        change = True
        while change:
            change = False

            for r in self.rns:
                toAdd = set()
                for s in self.role_incs[r]:
                    for t in self.role_incs[s]:
                        if t not in self.role_incs[r]:
                            change = True
                            toAdd.add(t)
                self.role_incs[r] |= toAdd

        # Add implied range restrictions
        for r in self.rns:
            for s in self.role_incs[r]:
                for A in self.ranges[s]:
                    self.add_range_restriction(r, A)

        # Add implied domain restrictions
        for B in self.cns:
            toAdd = set()
            for r, A in self.rBlhs[B]:
                for s in self.rns:
                    if r != s and r in self.role_incs[s]:
                        toAdd.add((s, A))
            self.rBlhs[B] |= toAdd

    def saturate(self):

        self.saturate_role_incs()
        # Extra rule to handle range restrictions
        # From
        # A \sqsubseteq \exists r.B and \exists r-.\top implies C
        # It follows that
        # A \sqsubseteq \exists r.X and X \sqsubseteq B \sqcap C
        for A, S in list(self.rBrhs.items()):
            to_add = set()
            for r, B in S:
                if r not in self.ranges.keys():
                    continue
                X = self.fresh_cn()
                self.add_axiom1(X, B)
                for C in self.ranges[r]:
                    self.add_axiom1(X, C)
                to_add.add((r, X))
            for r, X in to_add:
                self.add_axiom3(A, r, X)

        # TODO: implement faster algorithm
        change = True
        while change == True:
            change = False

            # CR3
            for A1 in self.cns:
                add = set()
                for A2 in self.implic[A1]:
                    for A3 in self.implic[A2]:
                        if A3 not in self.implic[A1]:
                            add.add(A3)
                if len(add) > 0:
                    change = True
                    self.implic[A1] |= add
            # CR4
            for B in self.non_empty_conjs():
                for A in self.cns:
                    if B not in self.implic[A]:
                        for s in self.conjs[B]:
                            if s.issubset(self.implic[A]):
                                self.add_axiom1(A, B)
                                change = True

            # CR5
            for A in self.non_empty_rhs():
                for B in self.non_empty_lhs():
                    if B not in self.implic[A]:
                        for r, A1 in self.rBrhs[A]:
                            for r2, B1 in self.rBlhs[B]:
                                if r == r2 and B1 in self.implic[A1]:
                                    self.add_axiom1(A, B)
                                    change = True


def construct_normalized_tbox(onto: Ontology):

    ignored_rules = 0
    ignored_domain = 0
    ignored_range = 0
    cis = 0
    onto = onto.normalize()

    # TODO: functional roles, reasoning rules for ELI, etc
    # TODO: inverse roles in ABox

    t = TBox(tag2name(expand_namespace("owl", "Thing")))
    t.fresh_names = set(NameFactory.created_names)
    for rule in onto.rules:
        if type(rule) != SubClassOf:
            ignored_rules += 1
            # print("Ignoring tbox rule {}".format(rule))
            continue
        if type(rule.subject) == ClassIdentifier:
            if type(rule.object) == ClassIdentifier:
                cis += 1
                t.add_axiom1(rule.subject.identifier, rule.object.identifier)

            if type(rule.object) == Restriction:
                if (
                    type(rule.object.quantifier) == SomeValues
                    and rule.object.prop.identifier != None
                ):
                    role = rule.object.prop.identifier
                    B = rule.object.quantifier.from_class.identifier
                    cis += 1
                    t.add_axiom3(rule.subject.identifier, role, B)
                else:
                    ignored_rules += 1
                    # print("Ignoring tbox rule with rhs {}".format(rule.object))
                    pass
                    # TODO handle inverse roles here

        elif type(rule.subject) == Restriction:
            if (
                type(rule.subject.quantifier) == SomeValues
                and rule.subject.prop.identifier != None
            ):
                role = rule.subject.prop.identifier
                B = rule.subject.quantifier.from_class.identifier

                cis += 1
                t.add_axiom4(role, B, rule.object.identifier)
            else:
                ignored_rules += 1
                # print("Ignoring tbox rule with lhs {}".format(rule.subject))
                # Inverse roles here
        elif type(rule.subject) == Intersection:
            assert len(rule.subject.children) == 2
            A1 = rule.subject.children[0].identifier
            A2 = rule.subject.children[1].identifier
            B = rule.object.identifier
            cis += 1
            t.add_axiom2(A1, A2, B)
        else:
            ignored_rules += 1
            # print("Ignoring tbox rule with lhs {}".format(rule.subject))

    for r in onto.properties:
        t.register_rn(r.identifier)

    for s in onto.subproperties:
        t.add_role_inc(s.subject.identifier, s.object.identifier)

    for a, b in onto.property_domains:
        if type(b) != ClassIdentifier:

            ignored_domain += 1
            # print("Ignoring domain restriction with concept {}".format(b))
            continue
        role = a
        B = t.top
        A = b.identifier
        t.add_axiom4(role, B, A)

    for a, b in onto.property_ranges:
        if type(b) == Thing or type(b) == TopClass:
            continue
        if type(b) != ClassIdentifier:
            ignored_range += 1
            # print("Ignoring range restriction with concept {}".format(b))
            continue
        t.add_range_restriction(a, b.identifier)
    if ignored_rules > 0:
        print(
            "Ignoring {} TBox statements due to unsupported features".format(
                ignored_rules
            )
        )
    if ignored_domain > 0:
        print(
            "Ignoring {} domain restrictions due to unsupported features".format(
                ignored_domain
            )
        )
    if ignored_range > 0:
        print(
            "Ignoring {} range restrictions due to unsupported features".format(
                ignored_range
            )
        )

    print(
        "Loaded {} concept names, {} role names, {} concept inclusions".format(
            len(t.cns), len(t.rns), cis
        )
    )
    return t


def compact_canonical_model(abox: ABoxBuilder, tbox: TBox):
    # TODO refactor. Should be more obviously correct
    for cn in tbox.cns:
        abox.declare_cn(cn)

    # Saturate concept names in ABox
    for A in tbox.implic.keys():
        for B in tbox.implic[A]:
            if A == B:
                continue
            for a in conceptname_ext(abox.A, A):
                if a in conceptname_ext(abox.A, B):
                    continue
                abox.concept_assertion(a, B)

    # Apply range restrictions to ABox
    for a in ind(abox.A):
        for b, r in abox.A.rn_ext[a]:
            if r in tbox.ranges.keys():
                for B in tbox.ranges[r]:
                    abox.concept_assertion(b, B)

    rev_succs: dict[int, dict[str, set[int]]] = {b: {} for b in ind(abox.A)}
    for a in ind(abox.A):
        for b, r in abox.A.rn_ext[a]:
            if r not in rev_succs[b]:
                rev_succs[b][r] = set()
            rev_succs[b][r].add(a)

    # Propagate concept names through ABox
    # TODO faster algorithm for ABox saturation
    change = True
    while change:
        change = False
        for A, S in tbox.rBlhs.items():
            for r, B in S:
                B_ext = set(conceptname_ext(abox.A, B))
                for b in B_ext:
                    if r not in rev_succs[b]:
                        continue
                    for a in rev_succs[b][r]:
                        if a not in conceptname_ext(abox.A, A):
                            change = True
                            for C in tbox.implic[A]:
                                abox.concept_assertion(a, C)

    # Create an anonymous individual for every \exists r. B on the rhs CIs
    for A, S in tbox.rBrhs.items():
        for r, B in S:
            idx = abox.map_ind(B)
            for C in tbox.implic[B]:
                abox.concept_assertion(idx, C)

    # Connect anonymous individuals
    for A, S in tbox.rBrhs.items():
        for r, B in S:
            for a in conceptname_ext(abox.A, A):
                abox.role_assertion(a, B, r)

    # Saturate with role inclusions
    for a in ind(abox.A):
        toadd = set()
        for b, r in abox.A.rn_ext[a]:
            for s in tbox.role_incs[r]:
                toadd.add((b, s))
        abox.A.rn_ext[a] |= toadd

    # Remove fresh concept names from model
    for A in tbox.fresh_names:
        abox.A.cn_ext[A] = set()


def structure_to_dot(A: Structure, indmap: dict[str, int]):
    print("digraph D {")

    for name, val in indmap.items():
        if "#" in name:
            print('N{} [label="{}"];'.format(val, name.split("#")[1]))
        else:
            print('N{} [label="{}"];'.format(val, name))

    for a in ind(A):
        for b, r in A.rn_ext[a]:
            if "#" in r:
                r = r.split("#")[1]
            print('N{} -> N{} [label="{}"];'.format(a, b, r))
    print("}")


def not_owl_thing(cn):
    return "/Thing>" not in cn and "#Thing>" not in cn


def solution2sparql(q: Structure):
    clauses: list[str] = []

    for a in ind(q):
        for cn in q.cn_ext.keys():
            if a in q.cn_ext[cn] and not_owl_thing(name2sparql(cn)):
                clauses.append("?{} a {} .".format(a, name2sparql(cn)))
        for b, rn in q.rn_ext[a]:
            clauses.append("?{} {} ?{} .".format(a, name2sparql(rn), b))

    if len(clauses) == 0:
        clauses.append("?0 a <http://www.w3.org/2002/07/owl#Thing> .")

    return "SELECT DISTINCT ?0 WHERE {{\n {}\n}}".format("\n ".join(clauses))


# Returns A restricted to individuals that can be reached in k steps from a
# Renames individuals
def restrict_to_neighborhood(k: int, A: Structure, starts: list[int]):
    cns = [cn for cn in A.cn_ext.keys() if A.cn_ext[cn]]

    # This has its own distance calculation to avoid computing the distance
    # for the entirety of A
    inds = set(starts)
    dist = {a: 0 for a in starts}
    for r in range(k):
        step = set()
        for i1 in inds:
            for i2, rn in A.rn_ext[i1]:
                step.add(i2)
        inds = inds.union(step)
        for i in step:
            if i in dist:
                dist[i] = min(r + 1, dist[i])
            else:
                dist[i] = r + 1

    mapping = {old_ind: new_ind for (new_ind, old_ind) in enumerate(inds)}

    n_indmap = {
        name: mapping[old_ind]
        for name, old_ind in A.indmap.items()
        if old_ind in mapping
    }

    B = Structure(
        max_ind=len(inds),
        cn_ext={cn: set() for cn in cns},
        rn_ext={a: set() for a in range(len(inds))},
        indmap=n_indmap,
        nsmap=A.nsmap,
    )

    for cn in cns:
        B.cn_ext[cn] = {mapping[ind] for ind in A.cn_ext[cn] & inds}

    for i1 in inds:
        B.rn_ext[mapping[i1]] = set()
        for i2, rn in A.rn_ext[i1]:
            if i2 in inds and dist[i1] < k:
                B.rn_ext[mapping[i1]].add((mapping[i2], rn))

    return (B, mapping)


def generate_all_trees(order: int):
    layout = list(range(order))

    while layout is not None:
        yield levels_to_preds(layout)
        layout = next_rooted_tree(layout)


def next_rooted_tree(predecessor: list[int]):
    p = len(predecessor) - 1
    while predecessor[p] == 1:
        p -= 1
    if p == 0:
        return None

    q = p - 1
    while predecessor[q] != predecessor[p] - 1:
        q -= 1
    result = list(predecessor)
    for i in range(p, len(result)):
        result[i] = result[i - p + q]
    return result


def levels_to_preds(layout: list[int]) -> list[int]:
    result = [0] * (len(layout) - 1)

    stack = []
    for i in range(len(layout)):
        if stack:
            while layout[stack[-1]] >= layout[i]:
                stack.pop()
            result[i - 1] = stack[-1]
        stack.append(i)
    return result


def copy_structure(A: Structure) -> Structure:
    cns = {}
    for cn in A.cn_ext.keys():
        cns[cn] = set(A.cn_ext[cn])

    rns = {}
    for a in ind(A):
        rns[a] = set(A.rn_ext[a])
    # TODO not a deep copy
    return Structure(
        max_ind=A.max_ind, cn_ext=cns, rn_ext=rns, indmap=A.indmap, nsmap=A.nsmap
    )
