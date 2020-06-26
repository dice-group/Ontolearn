from ontolearn import KnowledgeBase
from ontolearn.concept_learner import CELOE
from ontolearn.metrics import Recall
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.search import CELOESearchTree
from ontolearn.refinement_operators import ModifiedCELOERefinement


kb = KnowledgeBase(path='/home/demir/Desktop/Process.owl')


p = {'http://siemens.com/knowledge_graph/cyber_physical_systems/sma/processl-00076-ex#OP_2',
     "http://siemens.com/knowledge_graph/cyber_physical_systems/sma/product#OP_1",
     "http://siemens.com/knowledge_graph/cyber_physical_systems/sma/product#OP_4",
     "http://siemens.com/knowledge_graph/cyber_physical_systems/sma/processl-00076-ex#OP_3",
     "http://siemens.com/knowledge_graph/cyber_physical_systems/sma/product#029614"}
n = {"http://siemens.com/knowledge_graph/cyber_physical_systems/sma/product#031106",
     "http://siemens.com/knowledge_graph/cyber_physical_systems/sma/processl-00076-ex#OA_1_4",
     "http://siemens.com/knowledge_graph/cyber_physical_systems/sma/processl-00076-ex#OA_2_4",
     "http://siemens.com/knowledge_graph/cyber_physical_systems/sma/processl-00076-ex#OA_3_4",
     "http://siemens.com/knowledge_graph/cyber_physical_systems/sma/processl-00076-ex#OA_4_4"}

model = CELOE(knowledge_base=kb,
              refinement_operator=ModifiedCELOERefinement(kb=kb),
              quality_func=Recall(),
              min_horizontal_expansion=0,
              heuristic_func=CELOEHeuristic(),
              search_tree=CELOESearchTree(),
              terminate_on_goal=True,
              iter_bound=10,
              verbose=False)

model.predict(pos=p, neg=n)



model.search_tree.save_current_top_n_nodes(key=len,n=10,path='dmmm.owl')

# TODO include some functions in util to save and load desired list of concepts with individuals

exit(1)
