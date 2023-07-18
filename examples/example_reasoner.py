from owlapy.model import OWLSubClassOfAxiom, OWLEquivalentClassesAxiom, \
    OWLEquivalentObjectPropertiesAxiom
from owlapy.owlready2._base import OWLReasoner_Owlready2
from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.model import OWLObjectProperty, IRI, OWLObjectSomeValuesFrom, \
    OWLObjectIntersectionOf, OWLClass, OWLNamedIndividual


data_file = '../KGs/test_ontology.owl'
NS = 'http://www.semanticweb.org/stefan/ontologies/2023/1/untitled-ontology-11#'

"""
---------Object Properties--------- 

Domain(r1) = H, Range(r1) = G
r2 ⊑ r1
r3 ⊑ r4, r4 ⊑ r3
r7
Added through axioms in the code:
r5 ≡ r6

---------Classes-----------

AB ≡ (A ⊓ B), AB ⊑ C
D ⊑ (r7.E ⊓ B)
F ≡ r2.G, F ⊑ H
I ⊑ (J ⊓ K)
L ⊓ M = ∅
N ≡ Q
O ⊑ P, P ⊑ O
Added through axioms in the code:
R ⊑ r5.Q
(S ⊓ T) ⊑ U

---------Individuals-----------

o is O
p is P
a is A ^ B
b is B, b has r1.f
c is I
d is D
e is AB
f is E
g is G
n is N, n has r3.q, r4.l, r6.l
m is M
l is L
q is Q
ind1 has r5.q, r2.g
r is R
s is S ^ T

"""

a = OWLNamedIndividual(IRI(NS, "a"))
b = OWLNamedIndividual(IRI(NS, "b"))
c = OWLNamedIndividual(IRI(NS, "c"))
d = OWLNamedIndividual(IRI(NS, "d"))
e = OWLNamedIndividual(IRI(NS, "e"))
g = OWLNamedIndividual(IRI(NS, "g"))
m = OWLNamedIndividual(IRI(NS, "m"))
l = OWLNamedIndividual(IRI(NS, "l"))
n = OWLNamedIndividual(IRI(NS, "n"))
o = OWLNamedIndividual(IRI(NS, "o"))
p = OWLNamedIndividual(IRI(NS, "p"))
q = OWLNamedIndividual(IRI(NS, "q"))
r = OWLNamedIndividual(IRI(NS, "r"))
s = OWLNamedIndividual(IRI(NS, "s"))
ind1 = OWLNamedIndividual(IRI(NS, "ind1"))

r1 = OWLObjectProperty(IRI(NS, "r1"))
r2 = OWLObjectProperty(IRI(NS, "r2"))
r3 = OWLObjectProperty(IRI(NS, "r3"))
r4 = OWLObjectProperty(IRI(NS, "r4"))
r5 = OWLObjectProperty(IRI(NS, "r5"))
r6 = OWLObjectProperty(IRI(NS, "r6"))
r7 = OWLObjectProperty(IRI(NS, "r7"))

A = OWLClass(IRI(NS, 'A'))
B = OWLClass(IRI(NS, 'B'))
C = OWLClass(IRI(NS, 'C'))
AB = OWLClass(IRI(NS, 'AB'))
D = OWLClass(IRI(NS, 'D'))
E = OWLClass(IRI(NS, 'E'))
F = OWLClass(IRI(NS, 'F'))
G = OWLClass(IRI(NS, 'G'))
J = OWLClass(IRI(NS, 'J'))
K = OWLClass(IRI(NS, 'K'))
H = OWLClass(IRI(NS, 'H'))
I = OWLClass(IRI(NS, 'I'))
L = OWLClass(IRI(NS, 'L'))
M = OWLClass(IRI(NS, 'M'))
N = OWLClass(IRI(NS, 'N'))
O = OWLClass(IRI(NS, 'O'))
P = OWLClass(IRI(NS, 'P'))
Q = OWLClass(IRI(NS, 'Q'))
R = OWLClass(IRI(NS, 'R'))
S = OWLClass(IRI(NS, 'S'))
T = OWLClass(IRI(NS, 'T'))
U = OWLClass(IRI(NS, 'U'))

r2G = OWLObjectSomeValuesFrom(property=r2, filler=G)
r5Q = OWLObjectSomeValuesFrom(property=r5, filler=Q)
ST = OWLObjectIntersectionOf([S, T])
ABint = OWLObjectIntersectionOf([A, B])
r7E = OWLObjectSomeValuesFrom(property=r7, filler=E)
r7EB = OWLObjectIntersectionOf([r7E, B])
JK = OWLObjectIntersectionOf([J, K])

kb = KnowledgeBase(path=data_file)
onto = kb.ontology()
manager = onto.get_owl_ontology_manager()


manager.add_axiom(onto, OWLEquivalentObjectPropertiesAxiom([r5, r6]))
manager.add_axiom(onto, OWLSubClassOfAxiom(R, r5Q))
manager.add_axiom(onto, OWLSubClassOfAxiom(ST, U))

# manager.save_ontology(onto, IRI.create('file:/' + 'test' + '.owl'))

base_reasoner = OWLReasoner_Owlready2(onto)
reasoner = OWLReasoner_FastInstanceChecker(
                    onto,
                    base_reasoner,
                    negation_default=True,
                    sub_properties=True)

t1 = list(reasoner.instances(N))
t2 = list(reasoner.instances(r7E))
t3 = list(reasoner.instances(D))
t4 = list(reasoner.instances(H))
t5 = list(reasoner.instances(JK))
t6 = list(reasoner.instances(C))
t7 = list(reasoner.instances(r2G))
t8 = list(reasoner.instances(F))
t9 = list(reasoner.instances(ABint))
t10 = list(reasoner.instances(AB))
t11 = list(reasoner.instances(r7EB))
t67 = list(reasoner.instances(P))
t68 = list(reasoner.instances(O))
t75 = list(reasoner.instances(N))
t76 = list(reasoner.instances(Q))
t91 = list(reasoner.instances(r5Q))
t92 = list(reasoner.instances(R))
t99 = list(reasoner.instances(ST))
t100 = list(reasoner.instances(U))
t105 = list(reasoner.instances(r2G))

t12 = list(reasoner.equivalent_classes(A))
t13 = list(reasoner.equivalent_classes(L))
t26 = list(reasoner.equivalent_classes(AB, only_named=False))
t27 = list(reasoner.equivalent_classes(ABint))
t28 = list(reasoner.equivalent_classes(F, only_named=False))
t29 = list(reasoner.equivalent_classes(r2G))
t69 = list(reasoner.equivalent_classes(O))
t70 = list(reasoner.equivalent_classes(P))
t71 = list(reasoner.equivalent_classes(N))
t72 = list(reasoner.equivalent_classes(Q))

t30 = list(reasoner.equivalent_object_properties(r1))
t31 = list(reasoner.equivalent_object_properties(r2))
t83 = list(reasoner.equivalent_object_properties(r5))
t84 = list(reasoner.equivalent_object_properties(r6))
t55 = list(reasoner.equivalent_object_properties(r3))
t56 = list(reasoner.equivalent_object_properties(r4))

t16 = list(reasoner.sub_classes(r7E))
t17 = list(reasoner.sub_classes(r7EB))
t18 = list(reasoner.sub_classes(H, only_named=False))
t19 = list(reasoner.sub_classes(JK))
t20 = list(reasoner.sub_classes(C, only_named=False))
t32 = list(reasoner.sub_classes(A, only_named=False))
t61 = list(reasoner.sub_classes(P))
t62 = list(reasoner.sub_classes(O))
t77 = list(reasoner.sub_classes(N, only_named=False))
t78 = list(reasoner.sub_classes(Q, only_named=False))
t89 = list(reasoner.sub_classes(r5Q, only_named=False))
t102 = list(reasoner.sub_classes(R, only_named=False))
t97 = list(reasoner.sub_classes(U, only_named=False))

t59 = list(reasoner.sub_object_properties(r3))
t60 = list(reasoner.sub_object_properties(r4))
t85 = list(reasoner.sub_object_properties(r5))
t86 = list(reasoner.sub_object_properties(r6))

t21 = list(reasoner.super_classes(D, only_named=False))
t22 = list(reasoner.super_classes(r2G))
t23 = list(reasoner.super_classes(I, only_named=False))
t24 = list(reasoner.super_classes(AB))
t25 = list(reasoner.super_classes(ABint))
t63 = list(reasoner.super_classes(P))
t64 = list(reasoner.super_classes(O))
t79 = list(reasoner.super_classes(N, only_named=False))
t80 = list(reasoner.super_classes(Q, only_named=False))
t90 = list(reasoner.super_classes(r5Q, only_named=False))
t103 = list(reasoner.super_classes(R, only_named=False))
t98 = list(reasoner.super_classes(ST, only_named=False))
t104 = list(reasoner.super_classes(F))

t14 = list(reasoner.types(d))
t15 = list(reasoner.types(g))
t65 = list(reasoner.types(p))
t66 = list(reasoner.types(o))
t73 = list(reasoner.types(n))
t74 = list(reasoner.types(q))
t93 = list(reasoner.types(r))
t94 = list(reasoner.types(ind1))
t95 = list(reasoner.types(e))
t96 = list(reasoner.types(c))
t101 = list(reasoner.types(s))
t106 = list(reasoner.types(a))

t39 = list(reasoner.different_individuals(a))
t40 = list(reasoner.different_individuals(b))
t41 = list(reasoner.different_individuals(c))
t42 = list(reasoner.different_individuals(d))
t43 = list(reasoner.different_individuals(g))
t44 = list(reasoner.different_individuals(m))
t45 = list(reasoner.different_individuals(l))
t46 = list(reasoner.different_individuals(n))

t47 = list(reasoner.same_individuals(a))
t48 = list(reasoner.same_individuals(b))
t49 = list(reasoner.same_individuals(c))
t50 = list(reasoner.same_individuals(d))
t51 = list(reasoner.same_individuals(g))
t52 = list(reasoner.same_individuals(m))
t53 = list(reasoner.same_individuals(l))
t54 = list(reasoner.same_individuals(n))

t36 = list(reasoner.disjoint_classes(M, only_named=False))
t37 = list(reasoner.disjoint_classes(N, only_named=False))
t38 = list(reasoner.disjoint_classes(L, only_named=False))

t57 = list(reasoner.disjoint_object_properties(r3))
t58 = list(reasoner.disjoint_object_properties(r4))

t33 = list(reasoner.object_property_domains(r1))
t34 = list(reasoner.object_property_domains(r2))
t35 = list(reasoner.object_property_ranges(r1))

t81 = list(reasoner.object_property_values(n, r3, direct=False))
t82 = list(reasoner.object_property_values(n, r4, direct=False))
t87 = list(reasoner.object_property_values(n, r5))
t88 = list(reasoner.object_property_values(n, r6))

debug_breakpoint = "Place a breakpoint at this line"










