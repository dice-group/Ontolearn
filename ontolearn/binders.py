import subprocess

class DLLearnerBinder:
    def __init__(self, path):
        assert path
        self.execute_dl_learner_path = path

    def generate_config(self, knowledge_base_path, algorithm, positives, negatives, config_path,
                        num_of_concepts_tested):

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

        if algorithm == 'celoe':
            Text.append("alg.type = \"celoe\"")
            Text.append("alg.maxClassExpressionTests = " + str(num_of_concepts_tested))
            Text.append("alg.stopOnFirstDefinition = \"true\"")
        elif algorithm == 'ocel':
            Text.append("alg.type = \"ocel\"")
            Text.append("alg.maxClassDescriptionTests = " + str(num_of_concepts_tested))
            Text.append("alg.showBenchmarkInformation = \"true\"")
        elif algorithm == 'eltl':
            Text.append("alg.type = \"eltl\"")
            Text.append("alg.maxNrOfResults = \"1\"")
            Text.append("alg.stopOnFirstDefinition = \"true\"")
        else:
            raise ValueError('Wrong algorithm choosen.')
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

        top_predictions = None
        for ith, lines in enumerate(results):
            if 'solutions:' in lines:
                top_predictions = results[ith:]

        # top_predictions must have the following form
        """solutions:
        1: Parent(pred.acc.: 100.00 %, F - measure: 100.00 %)
        2: ⊤ (pred.acc.: 50.00 %, F-measure: 66.67 %)
        3: Person(pred.acc.: 50.00 %, F - measure: 66.67 %)
        """
        try:
            assert 'solutions:' in top_predictions[0]
        except:
            print('PARSING ERROR')

            for i in results:
                print(i)

            exit(1)
        str_f_measure = 'F-measure: '
        assert '1: ' in top_predictions[1]
        assert 'pred. acc.:' in top_predictions[1]
        assert str_f_measure in top_predictions[1]

        # Get last numerical value from first item
        best_pred_info = top_predictions[1]

        best_pred = best_pred_info[best_pred_info.index('1: ') + 3:best_pred_info.index(' (pred. acc.:')]

        f_measure = best_pred_info[best_pred_info.index(str_f_measure) + len(str_f_measure): -1]
        assert f_measure[-1] == '%'
        f_measure = float(f_measure[:-1])

        if serialize:
            f_name = config_path + '_' + 'Result.txt'
            with open(f_name, 'w') as handle:
                for sentence in results:
                    handle.write(sentence + '\n')
            handle.close()

        return best_pred, f_measure

    def pipeline(self, *, knowledge_base_path, algorithm, positives, negatives,
                 path_name, num_of_concepts_tested,
                 expand_goal_node_furhter=False,
                 name_of_Example=None, show_path=False):
        if algorithm is None:
            raise ValueError

        print('####### ', algorithm, ' starts ####### ')

        config_path = path_name + '_' + algorithm + '_' + str(num_of_concepts_tested)

        pathToConfig = self.generate_config(knowledge_base_path=knowledge_base_path,
                                            algorithm=algorithm,
                                            positives=positives, negatives=negatives,
                                            config_path=config_path, num_of_concepts_tested=num_of_concepts_tested)

        output_of_dl = list()

        output_of_dl.append('\n\n')
        output_of_dl.append('### ' + pathToConfig + ' starts ###')

        result = subprocess.run([self.execute_dl_learner_path + 'bin/cli', pathToConfig], stdout=subprocess.PIPE,
                                universal_newlines=True)

        lines = result.stdout.splitlines()
        output_of_dl.extend(lines)

        return self.parse_output(output_of_dl, config_path=config_path, serialize=False)