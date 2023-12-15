
mkdir FamilyBenchmarkResults
python examples/concept_learning_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family.owl --max_runtime 60 --report family_results1.csv && mv family_results1.csv FamilyBenchmarkResults
python examples/concept_learning_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family.owl --max_runtime 60 --report family_results2.csv && mv family_results2.csv FamilyBenchmarkResults
python examples/concept_learning_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family.owl --max_runtime 60 --report family_results3.csv && mv family_results3.csv FamilyBenchmarkResults
python examples/concept_learning_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family.owl --max_runtime 60 --report family_results4.csv && mv family_results4.csv FamilyBenchmarkResults
python examples/concept_learning_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family.owl --max_runtime 60 --report family_results5.csv && mv family_results5.csv FamilyBenchmarkResults


mkdir MutagenesisBenchmarkResults
python examples/concept_learning_evaluation.py --lps LPs/Mutagenesis/lps.json --kb KGs/Mutagenesis/mutagenesis.owl --max_runtime 60 --report mutagenesis_results1.csv && mv mutagenesis_results1.csv MutagenesisBenchmarkResults
python examples/concept_learning_evaluation.py --lps LPs/Mutagenesis/lps.json --kb KGs/Mutagenesis/mutagenesis.owl --max_runtime 60 --report mutagenesis_results2.csv && mv mutagenesis_results2.csv MutagenesisBenchmarkResults
python examples/concept_learning_evaluation.py --lps LPs/Mutagenesis/lps.json --kb KGs/Mutagenesis/mutagenesis.owl --max_runtime 60 --report mutagenesis_results3.csv && mv mutagenesis_results3.csv MutagenesisBenchmarkResults
python examples/concept_learning_evaluation.py --lps LPs/Mutagenesis/lps.json --kb KGs/Mutagenesis/mutagenesis.owl --max_runtime 60 --report mutagenesis_results4.csv && mv mutagenesis_results4.csv MutagenesisBenchmarkResults
python examples/concept_learning_evaluation.py --lps LPs/Mutagenesis/lps.json --kb KGs/Mutagenesis/mutagenesis.owl --max_runtime 60 --report mutagenesis_results5.csv && mv mutagenesis_results5.csv MutagenesisBenchmarkResults


mkdir CarcinogenesisBenchmarkResults
python examples/concept_learning_evaluation.py --lps LPs/Carcinogenesis/lps.json --kb KGs/Carcinogenesis/carcinogenesis.owl --max_runtime 60 --report carcinogenesis_results1.csv && mv carcinogenesis_results1.csv CarcinogenesisBenchmarkResults
python examples/concept_learning_evaluation.py --lps LPs/Carcinogenesis/lps.json --kb KGs/Carcinogenesis/carcinogenesis.owl --max_runtime 60 --report carcinogenesis_results2.csv && mv carcinogenesis_results2.csv CarcinogenesisBenchmarkResults
python examples/concept_learning_evaluation.py --lps LPs/Carcinogenesis/lps.json --kb KGs/Carcinogenesis/carcinogenesis.owl --max_runtime 60 --report carcinogenesis_results3.csv && mv carcinogenesis_results3.csv CarcinogenesisBenchmarkResults
python examples/concept_learning_evaluation.py --lps LPs/Carcinogenesis/lps.json --kb KGs/Carcinogenesis/carcinogenesis.owl --max_runtime 60 --report carcinogenesis_results4.csv && mv carcinogenesis_results4.csv CarcinogenesisBenchmarkResults
python examples/concept_learning_evaluation.py --lps LPs/Carcinogenesis/lps.json --kb KGs/Carcinogenesis/carcinogenesis.owl --max_runtime 60 --report carcinogenesis_results5.csv && mv carcinogenesis_results5.csv CarcinogenesisBenchmarkResults
