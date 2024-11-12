# About Ontolearn

**Version:** ontolearn 0.8.1

**GitHub repository:** [https://github.com/dice-group/Ontolearn](https://github.com/dice-group/Ontolearn)

**Publisher and maintainer:** [DICE](https://dice-research.org/) - data science research group of [Paderborn University](https://www.uni-paderborn.de/en/university).

**Contact**: [onto-learn@lists.uni-paderborn.de](mailto:onto-learn@lists.uni-paderborn.de)

**License:** MIT License

--------------------------------------------------------------------------------------------

Ontolearn is an open-source software library for explainable structured machine learning in Python.

Ontolearn started with the goal of using _Explainable Structured Machine Learning_ 
in OWL 2.0 ontologies and this
exactly what our library offers. The main contribution are the exclusive concept learning
algorithms that are part of this library. Currently, we have 6 fully functioning algorithms that 
learn concept in description logics. Papers can be found [here](09_further_resources.md).

For the base (core) module of Ontolearn we use [owlapy](https://github.com/dice-group/owlapy)
which on its end uses [Owlready2](https://owlready2.readthedocs.io/en/latest/index.html). _Owlapy_ is a python package
based on owlapi (the java counterpart), and implemented by us, the Ontolearn team. 
For the sake of modularization we have moved it in a separate repository.
The modularization aspect helps us to increase readability and reduce complexity.
So now we use owlapy not only for OWL 2 entities representation but
for ontology manipulation and reasoning as well.

---------------------------------------

**Ontolearn (including owlapy and ontosample) can do the following:**

- Load/save ontologies in RDF/XML, OWL/XML.
- Modify ontologies by adding/removing axioms.
- Access individuals/classes/properties of an ontology (and a lot more).
- Define learning problems.
- Sample ontologies.
- Construct class expressions.
- Use concept learning algorithms to classify positive examples in a learning problem.
- Use local datasets or datasets that are hosted on a triplestore server, for the learning task.
- Reason over an ontology.
- Other convenient functionalities like converting OWL class expressions to SPARQL or DL syntax.

------------------------------------

The rest of content after "examples" is build as a top-to-bottom guide, but nevertheless self-containing, where
you can learn more in depth about the capabilities of Ontolearn.
