from nltk import CFG, ChartParser
from random import choice
import requests
import sys
from owlapy.converter import Owl2SparqlConverter
from owlapy.parser import DLSyntaxParser

roles = []
concepts = []
with open('/home/nkouagou/Research/Ontolearn/KGs/DBpedia-2022-12/dbpedia2022-12_properties.txt') as f:
    roles = f.read().splitlines()
with open('/home/nkouagou/Research/Ontolearn/KGs/DBpedia-2022-12/dbpedia2022-12_classes.txt') as f:
    concepts = f.read().splitlines()

url = 'https://dbpedia-2022-12.data.dice-research.org/sparql'
payload = ("SELECT DISTINCT ?c WHERE { "
           "add_role_here <http://www.w3.org/2000/01/rdf-schema#range> ?c_temp. "
           "?c <http://www.w3.org/2000/01/rdf-schema#subClassOf>+ ?c_temp .}")
headers = {'content-type': 'application/sparql-query', 'Accept': 'application/sparql-results+json'}

# source: https://stackoverflow.com/questions/603687/how-do-i-generate-sentences-from-a-formal-grammar/3292027#3292027
def produce(inp_grammar, symbol, concepts, role=None, depth=1, max_depth=5):
    if depth > max_depth:
        return [choice(concepts)] # terminal
    words = []
    productions = inp_grammar.productions(lhs=symbol)
    production = choice(productions)
    cur_concepts = concepts
    changed = False
    for sym in production.rhs():
        if isinstance(sym, str):
            # print(sym)
            # print('str', sym)
            if 'R' in sym:
                words.append(sym.replace('R', role))
            else:
                words.append(sym.replace('A', choice(cur_concepts)))
        else:
            if str(sym) in ['Existential', 'Universal']:
                while True:
                    try:
                        random_role = choice(roles)
                        r = requests.post(url, data=payload.replace('add_role_here', random_role), headers=headers)
                        print(r.json())
                        role_concepts = ['<' + b['c']['value'] + '>' for b in r.json()['results']['bindings']]
                        if len(role_concepts) == 0:
                            continue
                        words.extend(produce(inp_grammar, sym, role_concepts, random_role, depth + 1, max_depth))
                        break
                    except:
                        continue
            else:
                words.extend(produce(inp_grammar, sym, cur_concepts, role, depth+1, max_depth))
    return words


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: [output directory]")
        sys.exit()
    output_dir = str(sys.argv[1])
    # C1: avoid double negation
    grammar = CFG.fromstring('''
        C -> Complement | Intersection | Union | Existential | Universal | Terminal
        Complement -> '¬'C1
        Intersection -> '('C ' ⊓ ' C')'
        Union -> '('C ' ⊔ ' C')'
        Existential -> '(∃ R.'C')'
        Universal -> '(∀ R.'C')'
        C1 -> Intersection | Union | Existential | Universal | Terminal
        Terminal -> 'A'
    ''')

    parser = ChartParser(grammar)

    gr = parser.grammar()
    count = 0
    # file to write all class expressions
    with open(f"{output_dir}/class_expressions.txt", 'w+') as ce_file:
        with open(f"{output_dir}/queries.txt", 'w+') as q_file:
            while True:
                if count >= 100:
                    break
                ce_list = produce(gr, gr.start(), concepts)
                # skip single concepts (just a scan)
                if len(ce_list) == 1:
                    continue
                count += 1
                ce_str = ''.join(ce_list)
                ce_parsed = DLSyntaxParser(namespace=None).parse_expression(expression_str=ce_str)
                sparql_query = Owl2SparqlConverter().as_query(root_variable='?x',
                                                                ce=ce_parsed,
                                                                for_all_de_morgan=True,
                                                                count=False,
                                                                values=None,
                                                                named_individuals=False)
                ce_file.write(ce_str)
                ce_file.write(",")
                ce_file.write(" ".join(sparql_query.split()))
                ce_file.write("\n")
                q_file.write(" ".join(sparql_query.split()))
                q_file.write("\n")


"""

from ontolearn.triple_store import TripleStore

kb = TripleStore(url="https://dbpedia.data.dice-research.org/sparql")

atomic_concepts = list(kb.ontology.classes_in_signature())

print('Total number of atomic concepts in DBpedia: ', len(atomic_concepts))
print('\nShowing some...\n\n', atomic_concepts[:10])

selected_concepts_str = ['http://dbpedia.org/ontology/Journalist', 'http://dbpedia.org/ontology/HistoricPlace', 'http://dbpedia.org/ontology/Lipid', 'http://dbpedia.org/ontology/Profession', 'http://dbpedia.org/ontology/Model', 'http://dbpedia.org/ontology/President', 'http://dbpedia.org/ontology/Academic', 'http://dbpedia.org/ontology/Actor', 'http://dbpedia.org/ontology/Place', 'http://dbpedia.org/ontology/FootballMatch']

selected_concepts = [c for c in atomic_concepts if any(s==c.str for s in selected_concepts_str)]

print('\nSelected concepts ', selected_concepts, '\n')
print('\nTotal selected: ', len(selected_concepts))

"""