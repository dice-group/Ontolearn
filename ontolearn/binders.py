import subprocess
from typing import List

class DLLearnerBinder:
    def __init__(self, path=None, model=None):
        assert path
        assert model
        self.execute_dl_learner_path = path

    def generate_config(self, knowledge_base_path, algorithm, positives, negatives, config_path
                        , max_run_time=10):

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
            "ks.fileName = \"" + knowledge_base_path + '\"')
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

        if algorithm == 'celoe':
            Text.append("alg.type = \"celoe\"")
            Text.append("alg.stopOnFirstDefinition = \"true\"")
        elif algorithm == 'ocel':
            Text.append("alg.type = \"ocel\"")
            Text.append("alg.showBenchmarkInformation = \"true\"")
        elif algorithm == 'eltl':
            Text.append("alg.type = \"eltl\"")
            Text.append("alg.maxNrOfResults = \"1\"")
            Text.append("alg.stopOnFirstDefinition = \"true\"")
        else:
            raise ValueError('Wrong algorithm choosen.')

        Text.append("alg.maxExecutionTimeInSeconds = " + str(max_run_time))

        Text.append("\n")

        pathToConfig = config_path + '.conf'  # /home/demir/Desktop/DL/DL-Learner-1.3.0/examples/family-benchmark

        file = open(pathToConfig, "wb")

        for i in Text:
            file.write(i.encode("utf-8"))
            file.write("\n".encode("utf-8"))
        file.close()
        return pathToConfig

    def parse_output(self, results, config_path, serialize):
        # output_of_dl.append('### ' + pathToConfig + ' ends ###')

        raise NotImplementedError
        if serialize:
            f_name = config_path + '_' + 'Result.txt'
            with open(f_name, 'w') as handle:
                for sentence in results:
                    handle.write(sentence + '\n')

        top_predictions = None
        for ith, lines in enumerate(results):
            if 'solutions' in lines:
                top_predictions = results[ith:]

        print(top_predictions[0])
        print(top_predictions[1])
        # top_predictions must have the following form
        """solutions ......:
        1: Parent(pred.acc.: 100.00 %, F - measure: 100.00 %)
        2: ⊤ (pred.acc.: 50.00 %, F-measure: 66.67 %)
        3: Person(pred.acc.: 50.00 %, F - measure: 66.67 %)
        """
        try:
            assert 'solutions' in top_predictions[0] and ':' == top_predictions[0][-1]
        except AssertionError as e:
            print('PARSING ERROR')

            for i in top_predictions:
                print(i)
            exit(1)

        print(top_predictions[1])

        str_f_measure = 'F-measure: '
        try:
            assert '1: ' in top_predictions[1]
            assert 'pred. acc.:' in top_predictions[1]
            assert str_f_measure in top_predictions[1]
        except AssertionError:
            print('Not expected value')

            print(top_predictions[1])
            exit(1)
        # Get last numerical value from first item
        best_pred_info = top_predictions[1]

        best_pred = best_pred_info[best_pred_info.index('1: ') + 3:best_pred_info.index(' (pred. acc.:')]

        f_measure = best_pred_info[best_pred_info.index(str_f_measure) + len(str_f_measure): -1]
        assert f_measure[-1] == '%'
        f_measure = float(f_measure[:-1])

        return best_pred, f_measure

    def pipeline(self, *, knowledge_base_path, algorithm, positives, negatives,
                 path_name, max_run_time=10):
        if algorithm is None:
            raise ValueError

        print('####### ', algorithm, ' starts ####### ')

        config_path = path_name + '_' + algorithm

        pathToConfig = self.generate_config(knowledge_base_path=knowledge_base_path,
                                            algorithm=algorithm, positives=positives, negatives=negatives,
                                            max_run_time=max_run_time,
                                            config_path=config_path)

        output_of_dl = list()

        output_of_dl.append('\n\n')
        output_of_dl.append('### ' + pathToConfig + ' starts ###')

        result = subprocess.run([self.execute_dl_learner_path + 'bin/cli', pathToConfig], stdout=subprocess.PIPE,
                                universal_newlines=True)

        lines = result.stdout.splitlines()
        output_of_dl.extend(lines)

        return self.parse_output(output_of_dl, config_path=config_path, serialize=False)

    def train(self, dataset: List = None)-> None:
        assert len(dataset) > 0
        print('No training for Dl-learner algorithms')

    def test(self, dataset: List = None, max_run_time=None):
        assert len(dataset) > 0
        if max_run_time is None:
            print('Max run time is set to 3')
            max_run_time = 3

        for i in dataset:
            print(i)
            exit(1)
