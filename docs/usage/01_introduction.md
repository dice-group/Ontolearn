# Ontolearn

**Version:** ontolearn 0.6.0

**GitHub repository:** [https://github.com/dice-group/Ontolearn](https://github.com/dice-group/Ontolearn)

**Publisher and maintainer:** [DICE](https://dice-research.org/) - data science research group of [Paderborn University](https://www.uni-paderborn.de/en/university).

**Contact**: [onto-learn@lists.uni-paderborn.de](mailto:onto-learn@lists.uni-paderborn.de)

**License:** GNU Affero General Public License v3 or later (AGPLv3+)

--------------------------------------------------------------------------------------------

Ontolearn is an open-source software library for explainable structured machine learning in Python.

Ontolearn started with the goal of using _Explainable Structured Machine Learning_ 
in OWL 2.0 ontologies and this
exactly what our library offers. The main contribution are the exclusive concept learning
algorithms that are part of this library. Currently, we have 4 fully functioning algorithms that 
learn concept in description logics. Papers can be found [here](09_further_resources.md).

For the base (core) module of Ontolearn we use [owlapy](https://github.com/dice-group/owlapy)
combined with [Owlready2](https://owlready2.readthedocs.io/en/latest/index.html). _Owlapy_ is a python package
based on owlapi and implemented by us, the Ontolearn team. For the sake of modularization we
have moved it in a separate repository. On the other side _Owlready2_ is a package for manipulating 
OWL 2.0 ontologies in Python. So in the end we have implemented
a higher degree of code for manipulation OWL 2.0 ontologies, in pursuit of making it 
easier to implement and understand, and of course, having this all in Python. This adaptation of 
Owlready2 library made it possible to build more complex algorithms.

---------------------------------------

**Ontolearn (including owlapy) can do the following:**

- Load/save ontologies in RDF/XML, OWL/XML.
- Modify ontologies by adding/removing axioms.
- Access individuals/classes/properties of an ontology (and a lot more).
- Define learning problems.
- Construct class expressions.
- Use concept learning algorithms to classify positive examples in a learning problem.
- Use local datasets or datasets that are hosted on a triplestore server, for the learning task.
- Reason over an ontology.
- Other convenient functionalities like converting OWL class expressions to SPARQL or DL syntax.

------------------------------------

The rest of the content is build as a top-to-bottom guide, but nevertheless self-containing, where
you can learn more in depth about the capabilities of Ontolearn.

If you want to quickly view the concept learning algorithms in action check 
[_Quick try-out_](06_concept_learners.md#quick-try-out).



