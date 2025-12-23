# About Ontolearn

**Version:** ontolearn 0.9.2

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

At the core of OntoLearn lies [OWLAPY](https://github.com/dice-group/owlapy), a Python package inspired by the OWL API (its Java counterpart) and developed by 
the OntoLearn team. To enhance modularity, readability, and maintainability, we have separated Owlapy from Ontolearn into an 
independent repository. This modular approach allows Owlapy to serve not only as a framework for representing OWL 2 
entities, but also as a tool for ontology manipulation and reasoning.

---------------------------------------

**Ontolearn offers:**

- **Diverse concept learning algorithms to generate hypotheses for classifying positive examples in a learning problem**.
- **Support for local datasets and datasets hosted on triplestore servers.**
- Perform operation on generated class expressions like evaluating them using different metrics, verbalizing, etc.
- Generate learning problems.
- An extensible and modular architecture that allows easy integration of new algorithms and functionalities.

Via [OWLAPY](https://github.com/dice-group/owlapy) you can also perform ontology manipulations and reasoning.

[OntoSample](https://github.com/alkidbaci/OntoSample) is another library closely related with the task of concept learning, where sampling is used 
to accelerate the learning process.

------------------------------------

The rest of content after "examples" is build as a top-to-bottom guide, but nevertheless self-containing, where
you can learn more in depth about the components of Ontolearn.
