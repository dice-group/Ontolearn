from owlready2 import *
import random



# def make_kb_incomplete(kb_path, output_path, rate, seed):
#     """
#     Makes the knowledge base incomplete by removing a certain percentage of statements (triples).

#     Inputs:
#     ---------------

#     kb_path: Path to the input knowledge base.
#     output_path: Path to save the modified (incomplete) knowledge base.
#     rate: Percentage of statements to remove (0-100).
#     seed: random seed for reproducibility.

#     Output:
#     ---------------

#     Incomplete KB at level rate %
#     """

#     random.seed(seed)

#     # Load the ontology
#     kb = get_ontology(kb_path).load()
    
#     # Get all individuals in the ontology
#     all_individuals = list(kb.individuals())
    
#     # Collect all triples (subject-predicate-object) related to the individuals
#     all_triples = []
#     for individual in all_individuals:
#         for prop in individual.get_properties():
#             for value in prop[individual]:
#                 all_triples.append((individual, prop, value))
    
#     # Calculate the number of triples to remove based on the rate
#     num_to_remove = int(len(all_triples) * (rate / 100))
    
#     # Randomly select triples to remove
#     triples_to_remove = random.sample(all_triples, num_to_remove)

#     print(len(triples_to_remove))
#     # exit(0)
    
#     # Remove the selected triples
#     for subject, predicate, obj in triples_to_remove:

    
       
#         predicate[subject].remove(obj)
        

    
#     # Save the modified ontology to a new file
#     kb.save(file=output_path, format="rdfxml")





def make_kb_incomplete(kb_path, output_path, rate, seed):
    """
    Makes the knowledge base incomplete by removing a certain percentage of individuals.

    
    Inputs:
    ---------------

    kb_path: Path to the input knowledge base.
    output_path: Path to save the modified (incomplete) knowledge base.
    rate: Percentage of individuals to remove (0-100).
    seed: random seed for reproducibility.

    Output:
    ---------------

    Incomplete KB at level rate %
    """

    random.seed(seed)

    # Load the ontology
    kb = get_ontology(kb_path).load()
    
    # Get all individuals (instances) in the ABox
    all_individuals = list(kb.individuals())
    
    # Calculate the number of individuals to remove based on the rate
    num_to_remove = int(len(all_individuals) * (rate / 100))
    
    # Randomly select individuals to remove
    individuals_to_remove = random.sample(all_individuals, num_to_remove)
    
    # Remove the selected individuals
    for individual in individuals_to_remove:
        destroy_entity(individual)
    
    # Save the modified ontology to a new file
    kb.save(file=output_path, format="rdfxml")


# seed = 1
# rate = 10  
# kb_path = "KGs/Family/father.owl"  
# output_path = f"incomplete_father_{rate}.owl"  

# make_kb_incomplete(kb_path, output_path, rate, seed)
