mkdir CVFamilyBenchmarkResults
python examples/concept_learning_cv_evaluation.py --lps LPs/Family/lps.json --kb KGs/Family/family.owl --max_runtime 60 --report cv_family_results.csv && mv cv_family_results.csv CVFamilyBenchmarkResults
mkdir CVMutagenesisBenchmarkResults
python examples/concept_learning_cv_evaluation.py --lps LPs/Mutagenesis/lps.json --kb KGs/Mutagenesis/mutagenesis.owl --max_runtime 60 --report cv_mutagenesis_results.csv && mv cv_mutagenesis_results.csv CVMutagenesisBenchmarkResults
mkdir CVCarcinogenesisBenchmarkResults
python examples/concept_learning_cv_evaluation.py --lps LPs/Carcinogenesis/lps.json --kb KGs/Carcinogenesis/carcinogenesis.owl --max_runtime 60 --report cv_carcinogenesis_results.csv && mv cv_carcinogenesis_results.csv CVCarcinogenesisBenchmarkResults
