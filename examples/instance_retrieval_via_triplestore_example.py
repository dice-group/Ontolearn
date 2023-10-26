from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.owlapy.model import OWLClass, IRI

"""
This is an example to show how simply you can retrieve instances for a class expression via a triplestore.

Prerequisite:
- Dataset
- Server hosting the dataset as a triplestore

For this example you can fulfill the prerequisites as follows:
1 - Download and unzip the datasets. See the commands under #Datasets in `download_external_resources.sh`
2 - Load and launch the triplestore server. See https://ontolearn-docs-dice-group.netlify.app/usage/05_reasoner#loading-and-launching-a-triplestore
    2.1 - The example in this script is for 'family' dataset, make the changes accordingly when setting up the triplestore server.

"""

# Create a knowledge base object for the Family benchmark using the default reasoner that will use the triplestore
# in the specified URL.
kb = KnowledgeBase(path="../KGs/Family/family-benchmark_rich_background.owl", use_triplestore=True,
                   triplestore_address="http://localhost:3030/family/sparql")

# Creating a class expression (in this case for 'Brother')
brother_class = OWLClass(IRI.create("http://www.benchmark.org/family#", "Brother"))

# Retrieving the instances for the class expression (behind the scene this is done via triplestore)
brother_individuals = kb.reasoner().instances(brother_class)

# Printing instances
[print(ind) for ind in brother_individuals]

