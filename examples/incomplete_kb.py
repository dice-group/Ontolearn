from owlready2 import *
import random



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
