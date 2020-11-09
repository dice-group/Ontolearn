from ontolearn import KnowledgeBase
PATH_KG = '../data/carcinogenesis.owl'
kb = KnowledgeBase(PATH_KG)
kb.describe()
kb.save('carcinogenesis')
