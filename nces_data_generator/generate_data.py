import argparse, json
from helper_classes import RDFTriples, KB2Data
parser = argparse.ArgumentParser()

parser.add_argument('--kbs', type=str, nargs='+', default=['carcinogenesis'], help='Knowledge base name')
parser.add_argument('--num_rand_samples', type=int, default=200, help='The number of random samples at each step of the generation process')
parser.add_argument('--depth', type=int, default=3, help='The depth of refinements')
parser.add_argument('--max_child_len', type=int, default=15, help='Maximum child length')
parser.add_argument('--refinement_expressivity', type=float, default=0.5)
parser.add_argument('--rho', type=str, default='ExpressRefinement', choices=['ExpressRefinement'], help='Refinement operator to use')

args = parser.parse_args()

for kb in args.kbs:
    triples = RDFTriples(source_kg_path=f'../NCESData/{kb}/{kb}.owl')
    triples.export_triples()
    with open(f'../NCESData/{kb}/data_generation_settings.json', "w") as setting:
        json.dump(vars(args), setting)
    DataGen = KB2Data(path=f'../NCESData/{kb}/{kb}.owl', rho_name=args.rho, depth=args.depth, max_child_length=args.max_child_len, refinement_expressivity=args.refinement_expressivity, downsample_refinements=True, num_rand_samples=args.num_rand_samples, min_num_pos_examples=1)
    DataGen.generate_descriptions().save_data()