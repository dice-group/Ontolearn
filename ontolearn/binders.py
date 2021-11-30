import subprocess
from datetime import datetime
from typing import List, Dict
from .utils import create_experiment_folder
import re
import time


class DLLearnerBinder:
    """
    dl-learner python binder.
    """

    def __init__(self, binary_path=None, model=None, kb_path=None, max_runtime=3):
        try:
            assert binary_path
            assert model
            assert kb_path
        except AssertionError:
            print(f'binary_path:{binary_path}, model:{model} kb_path{kb_path} can not be None')
            raise
        self.binary_path = binary_path
        self.kb_path = kb_path
        self.name = model
        self.max_runtime = max_runtime
        self.storage_path, _ = create_experiment_folder()
        self.best_predictions = None
        self.config_name_identifier = None

    def write_dl_learner_config(self, pos: List[str], neg: List[str]) -> str:
        """
        Writes config file for dl-learner.
        @param pos: A list of URIs of individuals indicating positive examples in concept learning problem.
        @param neg: A list of URIs of individuals indicating negatives examples in concept learning problem.
        @return: path of generated config file.
        """
        assert len(pos) > 0
        assert len(neg) > 0

        Text = list()
        pos_string = "{ "
        neg_string = "{ "
        for i in pos:
            pos_string += "\"" + str(
                i) + "\","
        for j in neg:
            neg_string += "\"" + str(
                j) + "\","

        pos_string = pos_string[:-1]
        pos_string += "}"

        neg_string = neg_string[:-1]
        neg_string += "}"

        Text.append("rendering = \"dlsyntax\"")
        Text.append("// knowledge source definition")

        Text.append("cli.type = \"org.dllearner.cli.CLI\"")
        Text.append("ks.type = \"OWL File\"")
        Text.append("\n")

        Text.append("// knowledge source definition")
        Text.append(
            "ks.fileName = \"" + self.kb_path + '\"')
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
        Text.append("op.useNumericDatatypes = \"false\"")
        Text.append("op.useCardinalityRestrictions = \"false\"")

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
            raise ValueError('Wrong algorithm chosen.')

        Text.append("alg.maxExecutionTimeInSeconds = " + str(self.max_runtime))
        Text.append("\n")

        pathToConfig = self.storage_path + '/' + self.name + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.conf'

        with open(pathToConfig, "wb") as wb:
            for i in Text:
                wb.write(i.encode("utf-8"))
                wb.write("\n".encode("utf-8"))
        return pathToConfig

    def fit(self, pos: List[str], neg: List[str], max_runtime: int = None):
        """
        Fit dl-learner model on a given positive and negative examples.
        @param pos: A list of URIs of individuals indicating positive examples in concept learning problem
        @param neg: A list of URIs of individuals indicating negatives examples in concept learning problem.
        @param max_runtime:
        @return: self.
        """
        try:
            assert len(pos) > 0
            assert len(neg) > 0
        except AssertionError:
            print(f'Positive and Negative Examples can not be 0 length:|Pos|={len(pos)},|Neg|{len(neg)}')
            raise
        if max_runtime:
            self.max_runtime = max_runtime
        pathToConfig = self.write_dl_learner_config(pos=pos, neg=neg)
        total_runtime = time.time()
        res = subprocess.run([self.binary_path + 'bin/cli', pathToConfig], stdout=subprocess.PIPE,
                             universal_newlines=True)
        total_runtime = round(time.time() - total_runtime, 3)
        self.best_predictions = self.parse_dl_learner_output(res.stdout.splitlines(), pathToConfig)
        self.best_predictions['Runtime'] = total_runtime
        return self

    def best_hypothesis(self):
        """
        return predictions if exists.
        """
        if self.best_predictions:
            return self.best_predictions
        else:
            print('Not prediction found.')

    def parse_dl_learner_output(self, output_of_dl_learner, file_path) -> Dict:
        """
        Parse the output received from executing dl-learner.
        @return: A dictionary of {'Prediction': ..., 'Accuracy': ..., 'F-measure': ...}
        """
        solutions = None
        best_concept_str = None
        acc = -1.0
        f_measure = -1.0
        search_info = None
        num_expression_tested = -1

        # DL-learner does not provide a unified output :(
        # ELTL  => No info pertaining to the number of concept tested, number of retrieval etc.
        # CELOE => Algorithm terminated successfully (time: 245ms, 188 descriptions tested, 69 nodes in the search
        #          tree).
        # OCEL  => Algorithm stopped (4505 descriptions tested).

        time.time()
        txt_path = file_path + '.txt'  # self.storage_path + '/output_' + self.name + '_' + str(time.time()) + '.txt'

        # (1) Store output of dl learner and extract solutions.
        with open(txt_path, 'w') as w:
            for th, sentence in enumerate(output_of_dl_learner):
                w.write(sentence + '\n')
                if 'solutions' in sentence and '1:' in output_of_dl_learner[th + 1]:
                    solutions = output_of_dl_learner[th:]

                if 'Algorithm' in sentence:
                    search_info = sentence

            # check whether solutions found
            if solutions:  # if solution found, check the correctness of relevant part of dl-learner output.
                try:
                    assert isinstance(solutions, list)
                    assert 'solutions' in solutions[0]
                    assert len(solutions) > 0
                    assert '1: ' in solutions[1][:5]
                except AssertionError:
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
                    if 'descriptions' in i:
                        search_info = i
                print('#################')
                print('#######{}##########'.format(self.name))
                print('#################')

                _ = re.findall(r'\d+ descriptions tested', search_info)
                assert len(_) == 1
                # Get the numbers
                num_expression_tested = int(re.findall(r'\d+', _[0])[0])

                return {'Model': self.name, 'Prediction': best_concept_str, 'Accuracy': float(acc) * .01,
                        'F-measure': float(f_measure) * .01, 'NumClassTested': int(num_expression_tested)}

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
            _ = re.findall(r'\d+ descriptions tested', search_info)
            assert len(_) == 1
            # Get the numbers
            num_expression_tested = int(re.findall(r'\d+', _[0])[0])

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

            if search_info is not None:
                # search_info is expected to be " Algorithm terminated successfully (time: 252ms, 188 descriptions
                # tested, 69 nodes in the search tree)."
                _ = re.findall(r'\d+ descriptions tested', search_info)
                if len(_) == 0:
                    assert self.name == 'eltl'
                else:
                    assert len(_) == 1
                    # Get the numbers
                    num_expression_tested = int(re.findall(r'\d+', _[0])[0])
        else:
            raise ValueError
        # 100% into range between 1.0 and 0.0
        return {'Prediction': best_concept_str, 'Accuracy': float(acc) * .01, 'F-measure': float(f_measure) * .01,
                'NumClassTested': int(num_expression_tested)}

    @staticmethod
    def train(dataset: List = None) -> None:
        """ do nothing """

    def fit_from_iterable(self, dataset: List = None, max_runtime=None) -> List[Dict]:
        """
        Fit dl-learner model on a list of given positive and negative examples.
        @param dataset:A list of tuple (s,p,n) where
        s => string representation of target concept
        p => positive examples, i.e. s(p)=1.
        n => negative examples, i.e. s(n)=0.
        @param max_runtime:
        @return:
        """
        assert len(dataset) > 0
        if max_runtime:
            assert isinstance(max_runtime, int)
            self.max_runtime = max_runtime

        return [self.fit(pos=p, neg=n, max_runtime=self.max_runtime).best_hypothesis() for (s, p, n) in dataset]
