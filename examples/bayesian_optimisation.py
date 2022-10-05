from tabnanny import verbose
from bayes_opt import BayesianOptimization
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1, Accuracy

def bayesian(pbounds, str_target_concept, lp, target_kb):    
    def black_box_function(max_runtime, tournament_size):
            space = dict()
            space['max_runtime'] = int(max_runtime)
            space['tournament_size'] = int(tournament_size)            
            model = EvoLearner(knowledge_base=target_kb, quality_func=F1(), **space)
            model.fit(lp, verbose=False)
            model.save_best_hypothesis(n=3, path='Predictions_{0}'.format(str_target_concept))      
            hypotheses = list(model.best_hypotheses(n=1))   
            quality = hypotheses[0].quality
            return quality
     
    optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
    )

    optimizer.maximize(
    init_points=2,
    n_iter=3,
    )
    
    print(optimizer.max)
