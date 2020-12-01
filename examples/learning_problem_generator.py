import os
import urllib.request
from hashlib import md5

from ontolearn import KnowledgeBase, LearningProblemGenerator
from ontolearn.static_funcs import export_concepts

# change pwd to script location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# download dbpedia ontology if not already present
path_dbpedia = '../data/dbpedia_2016-10.owl'
url_dbpedia = "http://downloads.dbpedia.org/2016-10/dbpedia_2016-10.owl"
md5_dbpedia = '7903e8b4cef72633b16816f548fe1a9a'
if not os.path.isfile(path_dbpedia) or md5(open(path_dbpedia, 'rb').read()).hexdigest() != md5_dbpedia:
    urllib.request.urlretrieve(url_dbpedia, path_dbpedia)

# actual script
kb = KnowledgeBase(path_dbpedia)
kb.describe()
lp = LearningProblemGenerator(knowledge_base=kb, num_problems=100, min_length=3, max_length=5)
export_concepts(kb, lp.concepts, path="generated_concepts.owl")
