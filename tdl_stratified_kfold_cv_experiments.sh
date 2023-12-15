
mkdir CVFamilyBenchmarkResults
python examples/concept_learning_cv_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family.owl --max_runtime 1 --report family_results1.csv && mv family_results1.csv CVFamilyBenchmarkResults

#mkdir CVMutagenesisBenchmarkResults
#python examples/concept_learning_cv_evaluation.py --lps LPs/Mutagenesis/lps.json --kb KGs/Mutagenesis/mutagenesis.owl --max_runtime 60 --report mutagenesis_results1.csv && mv mutagenesis_results1.csv CVMutagenesisBenchmarkResults
#mkdir CVCarcinogenesisBenchmarkResults
#python examples/concept_learning_cv_evaluation.py --lps LPs/Carcinogenesis/lps.json --kb KGs/Carcinogenesis/carcinogenesis.owl --max_runtime 60 --report carcinogenesis_results1.csv && mv carcinogenesis_results1.csv CVCarcinogenesisBenchmarkResults
