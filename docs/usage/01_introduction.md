# About Ontolearn

**Version:** ontolearn 0.9.0

**GitHub repository:** [https://github.com/dice-group/Ontolearn](https://github.com/dice-group/Ontolearn)

**Publisher and maintainer:** [DICE](https://dice-research.org/) - data science research group of [Paderborn University](https://www.uni-paderborn.de/en/university).

**Contact**: [cdemir@mail.uni-paderborn.de](mailto:cdemir@mail.uni-paderborn.de), [alkid@mail.uni-paderborn.de](mailto:alkid@mail.uni-paderborn.de)

**License:** MIT License

--------------------------------------------------------------------------------------------

OntoLearn is an open-source software library designed for explainable structured machine learning in OWL 2.0 ontologies.
Our primary objective is to leverage structured learning techniques within the OWL framework, providing a robust and 
interpretable approach to ontology-based machine learning.

One of OntoLearn’s key contributions is its exclusive concept learning algorithms, specifically tailored for Description 
Logics (DL). The library currently includes nine fully functional algorithms capable of learning complex concepts in DL. 
For further details and references, relevant research papers can be found [here](09_further_resources.md).

At the core of OntoLearn lies [Owlapy]((https://github.com/dice-group/owlapy)), a Python package inspired by the OWL API (its Java counterpart) and developed by 
the OntoLearn team. To enhance modularity, readability, and maintainability, we have separated Owlapy from Ontolearn into an 
independent repository. This modular approach allows Owlapy to serve not only as a framework for representing OWL 2 
entities, but also as a tool for ontology manipulation and reasoning.

---------------------------------------

**Ontolearn (including owlapy and ontosample) can do the following:**

- **Use concept learning algorithms to generate hypotheses for classifying positive examples in a learning problem**.
- **Use local datasets or datasets that are hosted on a triplestore server, for the learning task.**
- Construct/Generate class expressions and evaluate them using different metrics.
- Define learning problems.
- Load/create/save ontologies in RDF/XML, OWL/XML.
- Modify ontologies by adding/removing axioms.
- Access individuals/classes/properties of an ontology (and a lot more).
- Reason over an ontology.
- Convenient functionalities like converting OWL class expressions to SPARQL or DL syntax.
- Sample ontologies.

------------------------------------

The rest of content after "examples" is build as a top-to-bottom guide, but nevertheless self-containing, where
you can learn more in depth about the components of Ontolearn.
