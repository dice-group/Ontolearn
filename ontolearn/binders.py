import subprocess
from typing import List, Dict
from .util import create_experiment_folder, create_logger
import re
import time


class DLLearnerBinder:
    def __init__(self, binary_path=None, model=None, kb_path=None, max_runtime=3):
        assert binary_path
        assert model
        assert kb_path
        self.binary_path = binary_path
        self.kb_path = kb_path

        self.name = model
        self.max_runtime = max_runtime
        self.storage_path, _ = create_experiment_folder()
        self.logger = create_logger(name=self.name, p=self.storage_path)
        self.best_predictions = None

    def generate_config(self, positives: List[str], negatives: List[str]):
        assert len(positives) > 0
        assert len(negatives) > 0

        Text = list()
        pos_string = "{ "
        neg_string = "{ "
        for i in positives:
            pos_string += "\"" + str(
                i) + "\","
        for j in negatives:
            neg_string += "\"" + str(
                j) + "\","

        pos_string = pos_string[:-1]
        pos_string += "}"

        neg_string = neg_string[:-1]
        neg_string += "}"

        Text.append("rendering = \"dlsyntax\"")
        Text.append("// knowledge source definition")

        # perform cross validation
        Text.append("cli.type = \"org.dllearner.cli.CLI\"")
        # Text.append("cli.performCrossValidation = \"true\"")
        # Text.append("cli.nrOfFolds = 10\n")
        Text.append("ks.type = \"OWL File\"")
        Text.append("\n")

        Text.append("// knowledge source definition")

        Text.append(
            "ks.fileName = \"" + self.kb_path + '\"')
        # Text.append(
        #    "ks.fileName = \"" + '/home/demir/Desktop/DL/dllearner-1.4.0/examples/carcinogenesis/carcinogenesis.owl\"')  # carcinogenesis/carcinogenesis.ow

        Text.append("\n")
        Text.append("reasoner.type = \"closed world reasoner\"")
        Text.append("reasoner.sources = { ks }")
        Text.append("\n")

        Text.append("lp.type = \"PosNegLPStandard\"")
        Text.append("accuracyMethod.type = \"fmeasure\"")

        Text.append("\n")

        Text.append("lp.positiveExamples =" + pos_string)
        Text.append("\n")

        Text.append("lp.negativeExamples =" + neg_string)
        Text.append("\n")
        Text.append("alg.writeSearchTree = \"true\"")

        Text.append("op.type = \"rho\"")

        Text.append("op.useCardinalityRestrictions = \"false\"")

        # Text.append(
        #     "alg.searchTreeFile =\"" + config_path + '_search_tree.txt\"')  # carcinogenesis/carcinogenesis.ow
        # Text.append("alg.maxClassExpressionTests = " + str(num_of_concepts_tested))

        if self.name == 'celoe':
            Text.append("alg.type = \"celoe\"")
            Text.append("alg.stopOnFirstDefinition = \"true\"")
        elif self.name == 'ocel':
            Text.append("alg.type = \"ocel\"")
            Text.append("alg.showBenchmarkInformation = \"true\"")
        elif self.name == 'eltl':
            Text.append("alg.type = \"eltl\"")
            Text.append("alg.maxNrOfResults = \"1\"")
            Text.append("alg.stopOnFirstDefinition = \"true\"")
        else:
            raise ValueError('Wrong algorithm choosen.')

        Text.append("alg.maxExecutionTimeInSeconds = " + str(self.max_runtime))

        Text.append("\n")
        pathToConfig = self.storage_path + '/' + self.name + '.conf'  # /home/demir/Desktop/DL/DL-Learner-1.3.0/examples/family-benchmark

        with open(pathToConfig, "wb") as wb:
            for i in Text:
                wb.write(i.encode("utf-8"))
                wb.write("\n".encode("utf-8"))
        return pathToConfig

    def fit(self, pos: List[str], neg: List[str], max_runtime: int = None):
        """

        @param pos:
        @param neg:
        @param max_runtime:
        @return:
        """
        assert len(pos) > 0
        assert len(neg) > 0

        if max_runtime:
            self.max_runtime = max_runtime

        pathToConfig = self.generate_config(positives=pos, negatives=neg)

        total_runtime = time.time()
        res = subprocess.run([self.binary_path + 'bin/cli', pathToConfig], stdout=subprocess.PIPE,
                             universal_newlines=True)
        total_runtime = round(time.time() - total_runtime, 3)

        self.best_predictions = self.parse_dl_learner_output(res.stdout.splitlines())
        self.best_predictions['Runtime'] = total_runtime
        return self

    def best_hypotheses(self):
        return self.best_predictions

    def parse_dl_learner_output(self, output_of_dl_learner) -> Dict:
        """

        @param output_of_dl_learner:
        @return:
        """

        solutions = None
        best_concept_str = None
        acc = -1.0
        f_measure = -1.0

        # (1) Store output of dl learner and extract solutions.
        with open(self.storage_path + '/output_' + self.name + '.txt', 'w') as w:
            for th, sentence in enumerate(output_of_dl_learner):
                w.write(sentence + '\n')
                if 'solutions' in sentence and '1:' in output_of_dl_learner[th + 1]:
                    solutions = output_of_dl_learner[th:]

            # check whether solutions found
            if solutions:  # if solution found, check the correctness of relevant part of dl-learner output.
                try:
                    assert isinstance(solutions, list)
                    assert 'solutions' in solutions[0]
                    assert len(solutions) > 0
                    assert '1: ' in solutions[1][:5]
                except AssertionError as ast:
                    print(type(solutions))
                    print('####')
                    print(solutions[0])
                    print('####')
                    print(len(solutions))
            else:
                # no solution found.
                print('#################')
                print('#######{}##########'.format(self.name))
                print('#################')
                for i in output_of_dl_learner[-3:-1]:
                    print(i)
                print('#################')
                print('#######{}##########'.format(self.name))
                print('#################')
                return {'Model': self.name, 'Prediction': best_concept_str, 'Accuracy': float(acc),
                        'F-measure': float(f_measure)}

        # top_predictions must have the following form
        """solutions ......:
        1: Parent(pred.acc.: 100.00 %, F - measure: 100.00 %)
        2: ⊤ (pred.acc.: 50.00 %, F-measure: 66.67 %)
        3: Person(pred.acc.: 50.00 %, F - measure: 66.67 %)
        """
        best_solution = solutions[1]

        if self.name == 'ocel':
            """ parse differently"""
            token = '(accuracy '
            start_index = len('1: ')
            end_index = best_solution.index(token)
            best_concept_str = best_solution[start_index:end_index - 1]  # -1 due to white space between *) (*.
            quality_info = best_solution[end_index:]
            # best_concept_str => *Sister ⊔ (Female ⊓ (¬Granddaughter))*
            # quality_info     => *(accuracy 100%, length 16, depth 2)*

            # Create a list to hold the numbers
            predicted_accuracy_info = re.findall(r'accuracy \d*%', quality_info)

            assert len(predicted_accuracy_info) == 1
            assert predicted_accuracy_info[0][-1] == '%'  # percentage sign
            acc = re.findall(r'\d+\.?\d+', predicted_accuracy_info[0])[0]

        elif self.name in ['celoe', 'eltl']:
            # e.g. => 1: Sister ⊔ (∃ married.Brother) (pred. acc.: 90.24%, F-measure: 91.11%)
            # Heuristic => Quality info start with *(pred. acc.: *
            token = '(pred. acc.: '
            start_index = len('1: ')
            end_index = best_solution.index(token)
            best_concept_str = best_solution[start_index:end_index - 1]  # -1 due to white space between *) (*.
            quality_info = best_solution[end_index:]
            # best_concept_str => *Sister ⊔ (Female ⊓ (¬Granddaughter))*
            # quality_info     => *(pred. acc.: 79.27%, F-measure: 82.83%)*

            # Create a list to hold the numbers
            predicted_accuracy_info = re.findall(r'pred. acc.: \d+.\d+%', quality_info)
            f_measure_info = re.findall(r'F-measure: \d+.\d+%', quality_info)

            assert len(predicted_accuracy_info) == 1
            assert len(f_measure_info) == 1

            assert predicted_accuracy_info[0][-1] == '%'  # percentage sign
            assert f_measure_info[0][-1] == '%'  # percentage sign

            acc = re.findall(r'\d+\.?\d+', predicted_accuracy_info[0])[0]
            f_measure = re.findall(r'\d+\.?\d+', f_measure_info[0])[0]
        else:
            raise ValueError

        return {'Prediction': best_concept_str, 'Accuracy': float(acc), 'F-measure': float(f_measure)}

    @staticmethod
    def train(dataset: List = None) -> None:
        """ do nothing """

    def fit_from_iterable(self, dataset: List = None, max_runtime=None) -> List[Dict]:
        """
        @param dataset:
        @param max_runtime:
        @return:
        """
        assert len(dataset) > 0
        if max_runtime is None:
            print('Max run time is set to 3')
            self.max_runtime = 3
        results = []
        for (s, p, n) in dataset:
            best_pred = self.fit(pos=p, neg=n, max_runtime=3).best_hypotheses()
            print(best_pred)
            exit(1)
            results.append(best_pred)

        return results
