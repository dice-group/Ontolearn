# Ontolearn

*Ontolearn* is an open-source software library for description logic learning problem.
- **Drill** &rarr; [Neuro-Symbolic Class Expression Learning](https://www.ijcai.org/proceedings/2023/0403.pdf)
- **EvoLearner** &rarr; [EvoLearner: Learning Description Logics with Evolutionary Algorithms](https://dl.acm.org/doi/abs/10.1145/3485447.3511925)
- **NCES2** &rarr; (soon) [Neural Class Expression Synthesis in ALCHIQ(D)](https://papers.dice-research.org/2023/ECML_NCES2/NCES2_public.pdf)
- **NCES** &rarr; [Neural Class Expression Synthesis](https://link.springer.com/chapter/10.1007/978-3-031-33455-9_13) 
- **NERO** &rarr; [Learning Permutation-Invariant Embeddings for Description Logic Concepts](https://link.springer.com/chapter/10.1007/978-3-031-30047-9_9)
- **CLIP** &rarr; (soon) [Learning Concept Lengths Accelerates Concept Learning in ALC](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_14)
- **CELOE** &rarr; [Class Expression Learning for Ontology Engineering](https://www.sciencedirect.com/science/article/abs/pii/S1570826811000023)
- **OCEL** &rarr; A limited version of CELOE

You can find more details about these algorithms and what *Ontolearn* has to offer in the [documentation](https://ontolearn-docs-dice-group.netlify.app/index.html).

Quick navigation: 
- [Installation](#installation)
- [Quick try-out](#quick-try-out)
- [Usage](#usage)
- [Relevant Papers](#relevant-papers)

## Installation
For detailed instructions please refer to the [installation guide](https://ontolearn-docs-dice-group.netlify.app/usage/installation.html) in the documentation.

### Installation from source

Make sure to set up a virtual python environment like [Anaconda](https://www.anaconda.com/)  before continuing with the installation. 

To successfully pass all the tests you need to download some external resources in advance 
(see [_Download external files_](#download-external-files)). You will need
at least to download the datasets. Also, install _java_ and _curl_ if you don't have them in your system already:

```commandline
sudo apt install openjdk-11-jdk
sudo apt install curl
```

A quick start up will be as follows:

```shell
git clone https://github.com/dice-group/Ontolearn.git && conda create --name onto python=3.8 && conda activate onto 
pip3 install -r requirements.txt && python -c "import ontolearn"
# wget https://files.dice-research.org/projects/Ontolearn/KGs.zip -O ./KGs.zip && unzip KGs.zip
# python -m pytest tests # Partial test with pytest
```
or
```shell
pip install ontolearn  # more on https://pypi.org/project/ontolearn/
```

## Quick try-out

You can execute the script `deploy_cl.py` to deploy the concept learners in a local web server and try
the algorithms using an interactive interface made possible by [Gradio](https://www.gradio.app/). Currently, 
you can only deploy the following concept learners: **NCES**, **EvoLearner**, **CELOE** and **OCEL**.

> **NOTE: In case you don't have a dataset, don't worry, you can use
> the datasets we store in our data server. See _[Download external files](#download-external-files)_.**

For example the command below will launch an interface using **EvoLearner** as the model on 
the **Family** dataset which is a simple dataset with 202 individuals:

```shell
python deploy_cl.py --model evolearner --path_knowledge_base KGs/Family/family-benchmark_rich_background.owl
```

Once you run this command, a local URL where the model is deployed will be provided to you.


In the interface you need to enter the positive and the negative examples. For a quick run you can
click on the **Random Examples** checkbox, but you may as well enter some real examples for
the learning problem of **Aunt**, **Brother**, **Cousin**, etc. which
you can find in the file `examples/synthetic_problems.json`. Just copy and paste the IRIs of
positive and negative examples for a specific learning problem directly
in their respective fields.

Run the help command to see the description on this script usage:

```shell
python deploy_cl.py --help
```

## Usage

In the [examples](https://github.com/dice-group/Ontolearn/tree/develop/examples) folder, you can find examples on how to use
the learning algorithms. Also in the [tests](https://github.com/dice-group/Ontolearn/tree/develop/tests) folder you can find the test cases.

For more detailed instructions we suggest to follow the [guides](https://ontolearn-docs-dice-group.netlify.app/usage/06_concept_learners) in the documentation.

Below we give a simple example on using CELOE to learn class expressions for a small dataset.

```python
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import CELOE, OCEL, EvoLearner, Drill
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1
from ontolearn.owlapy.model import OWLNamedIndividual, IRI
from ontolearn.utils import setup_logging
from ontolearn.owlapy.render import DLSyntaxObjectRenderer
setup_logging()

renderer = DLSyntaxObjectRenderer()


kb = KnowledgeBase(path="../KGs/Family/family-benchmark_rich_background.owl")
lp = PosNegLPStandard(pos={OWLNamedIndividual(IRI.create(p)) for p in
                           {"http://www.benchmark.org/family#F10F175",
                            "http://www.benchmark.org/family#F10F177"}},
                      neg={OWLNamedIndividual(IRI.create("http://www.benchmark.org/family#F9M142"))})

max_runtime, topk=1, 3
preds_evo = list(EvoLearner(knowledge_base=kb, quality_func=F1(), max_runtime=max_runtime).fit(lp).best_hypotheses(n=topk))
preds_celoe = list(CELOE(knowledge_base=kb, quality_func=F1(), max_runtime=max_runtime).fit(lp).best_hypotheses(n=topk))
preds_ocel = list(OCEL(knowledge_base=kb, quality_func=F1(), max_runtime=max_runtime).fit(lp).best_hypotheses(n=topk))
preds_drill = list(Drill(knowledge_base=kb, quality_func=F1(), max_runtime=max_runtime).fit(lp).best_hypotheses(n=topk))

for i in range(3):
    print(f"{i+1}.Pred:\n"
          f"DRILL:{renderer.render(preds_drill[i].concept)}\n"
          f"EvoLearner:{renderer.render(preds_celoe[i].concept)}\n"
          f"CELOE:{renderer.render(preds_celoe[i].concept)}\nOCEL:{renderer.render(preds_ocel[i].concept)}\n")

```
The goal in this example is to learn a class expression for the concept "father". 
The output is as follows:
```
The result: (¬female) ⊓ (∃ hasChild.⊤) has quality 1.0
```

NCES is a powerful algorithm implemented recently.
For a quick start on how to use it, please refer to the notebook [simple usage NCES](examples/simple-usage-NCES.ipynb).

----------------------------------------------------------------------------

#### Download external files

Some resources like pre-calculated embeddings or `pre_trained_agents` and datasets (ontologies)
are not included in the repository directly. Use the command line command `wget`
 to download them from our data server.

> **NOTE: Before you run this commands in your terminal, make sure you are 
in the root directory of the project!**

To download the datasets:

```shell
wget https://files.dice-research.org/projects/Ontolearn/KGs.zip -O ./KGs.zip
```

Then depending on your operating system, use the appropriate command to unzip the files:

```shell
# Windows
tar -xf KGs.zip

# or

# macOS and Linux
unzip KGs.zip
```

Finally, remove the _.zip_ file:

```shell
rm KGs.zip
```

And for NCES data: 

```shell
wget https://files.dice-research.org/projects/NCES/NCES_Ontolearn_Data/NCESData.zip -O ./NCESData.zip
unzip NCESData.zip
rm NCESData.zip
```


----------------------------------------------------------------------------

#### Building (sdist and bdist_wheel)

You can use <code>tox</code> to build sdist and bdist_wheel packages for Ontolearn.
- "sdist" is short for "source distribution" and is useful for distribution of packages that will be installed from source.
- "bdist_wheel" is short for "built distribution wheel" and is useful for distributing packages that include large amounts of compiled code, as well as for distributing packages that have complex dependencies.

To build and compile the necessary components of Ontolearn, use:
```shell
tox -e build
```

To automatically build and test the documentation of Ontolearn, use:
```shell
tox -e docs
```

----------------------------------------------------------------------------

#### Simple Linting

Using the following command will run the linting tool [flake8](https://flake8.pycqa.org/) on the source code.
```shell
flake8
```

----------------------------------------------------------------------------

### Citing
Currently, we are working on our manuscript describing our framework. 
If you find our work useful in your research, please consider citing the respective paper:
```
# DRILL
@inproceedings{demir2023drill,
  added-at = {2023-08-01T11:08:41.000+0200},
  author = {Demir, Caglar and Ngomo, Axel-Cyrille Ngonga},
  booktitle = {The 32nd International Joint Conference on Artificial Intelligence, IJCAI 2023},
  title = {Neuro-Symbolic Class Expression Learning},
  url = {https://www.ijcai.org/proceedings/2023/0403.pdf},
 year={2023}
}

# NCES2
@inproceedings{kouagou2023nces2,
author={Kouagou, N'Dah Jean and Heindorf, Stefan and Demir, Caglar and Ngonga Ngomo, Axel-Cyrille},
title={Neural Class Expression Synthesis in ALCHIQ(D)},
url = {https://papers.dice-research.org/2023/ECML_NCES2/NCES2_public.pdf},
booktitle={Machine Learning and Knowledge Discovery in Databases},
year={2023},
publisher={Springer Nature Switzerland},
address="Cham"
}

# NCES
@inproceedings{kouagou2023neural,
  title={Neural class expression synthesis},
  author={Kouagou, N’Dah Jean and Heindorf, Stefan and Demir, Caglar and Ngonga Ngomo, Axel-Cyrille},
  booktitle={European Semantic Web Conference},
  pages={209--226},
  year={2023},
  publisher={Springer Nature Switzerland}
}

# EvoLearner
@inproceedings{heindorf2022evolearner,
  title={Evolearner: Learning description logics with evolutionary algorithms},
  author={Heindorf, Stefan and Bl{\"u}baum, Lukas and D{\"u}sterhus, Nick and Werner, Till and Golani, Varun Nandkumar and Demir, Caglar and Ngonga Ngomo, Axel-Cyrille},
  booktitle={Proceedings of the ACM Web Conference 2022},
  pages={818--828},
  year={2022}
}


# CLIP
@inproceedings{kouagou2022learning,
  title={Learning Concept Lengths Accelerates Concept Learning in ALC},
  author={Kouagou, N’Dah Jean and Heindorf, Stefan and Demir, Caglar and Ngonga Ngomo, Axel-Cyrille},
  booktitle={European Semantic Web Conference},
  pages={236--252},
  year={2022},
  publisher={Springer Nature Switzerland}
}
```

In case you have any question, please contact:  ```onto-learn@lists.uni-paderborn.de```
