# Architecture

Here we present the class hierarchies of Ontolearn.

:::{raw} latex
\let\origsphinxincludegraphics\sphinxincludegraphics
\begingroup
\renewcommand{\sphinxincludegraphics}[1]{\origsphinxincludegraphics[scale=.5]{#1}}
:::

## Knowledge Base organisation

:::{raw} latex
\begin{wrapfigure}{r}{.55\linewidth}\vspace{-60pt}\raggedleft%
:::
:::{uml}
package ontolearn.abstracts {
abstract AbstractKnowledgeBase
}
package ontolearn.concept_generator {
class ConceptGenerator
}
package ontolearn.knowledge_base {
class KnowledgeBase
}

AbstractKnowledgeBase <|-- KnowledgeBase
ConceptGenerator --o KnowledgeBase
:::
:::{raw} latex
\end{wrapfigure}
:::

The Knowledge Base keeps the OWL Ontology and Reasoner together and
manages the OWL Hierarchies. It is also responsible for delegating
and caching instance queries and encoding instances.

:::{raw} latex
\vspace{20pt}\WFclear
:::

## Learning Algorithms

:::{raw} latex
\begin{wrapfigure}{r}{.55\linewidth}\vspace{-50pt}\raggedleft%
:::
:::{uml}
package ontolearn.base_concept_learner {
abstract BaseConceptLearner
}
package ontolearn.concept_learner {
class CELOE
}

BaseConceptLearner <|-- CELOE
:::
:::{raw} latex
\end{wrapfigure}
:::

There may be several Concept Learning Algorithms to choose from. Each
may be better suited for a specific kind of Learning Problem or desired
result.

:::{raw} latex
\vspace{40pt}\WFclear
:::

## Heuristics

:::{raw} latex
\begin{wrapfigure}{r}{.3\linewidth}\vspace{-50pt}\raggedleft%
:::
:::{uml}
package ontolearn.abstracts {
abstract AbstractHeuristic
}
package ontolearn.heuristics {
class CELOEHeuristic
}

AbstractHeuristic <|-- CELOEHeuristic
:::
:::{raw} latex
\end{wrapfigure}
:::

Heuristics are an abstraction to guide the concept search process. The
included CELOE Heuristic takes as basis the [Quality
function](#quality-functions) and then adds some weights, like
penalties for extending the length of a concept or for repeatedly
searching the same position in the search space. Other heuristics can be
invented and tested.

:::{raw} latex
\vspace{\baselineskip}\WFclear
:::

## Refinement Operators

:::{raw} latex
\begin{wrapfigure}{r}{.55\linewidth}\raggedleft%
:::
:::{uml}
package ontolearn.abstracts {
abstract BaseRefinement
}
package ontolearn.refinement_operators {
class ModifiedCELOERefinement
}
package ontolearn.refinement_operators {
class LengthBasedRefinement
}

BaseRefinement <|-- ModifiedCELOERefinement
BaseRefinement <|-- LengthBasedRefinement
:::
:::{raw} latex
\vspace{2\baselineskip}\end{wrapfigure}
:::

Refinement operators suggest new concepts to evaluate based on a given
concept. Ultimately, the refinement operator decides what
Description Logics language family the learning algorithm will be able
to cover. The included modified CELOE refinement will iteratively
generate ALC concepts by generating all available atomic classes and
properties, and combine them with ⊓, ⊔, ¬, ∀ and ∃, and successively
increase their length. The length based refinement will intuitively do
the same but immediately go to a specific length instead of searching
the concepts by increasing length.

:::{raw} latex
\WFclear
:::

## Learning Problem types

:::{raw} latex
\begin{wrapfigure}{r}{.4\linewidth}\vspace{-50pt}\raggedleft%
:::
:::{uml}
package ontolearn.abstracts {
abstract AbstractLearningProblem
}
package ontolearn.learning_problem {
class PosNegLPStandard
}

AbstractLearningProblem <|-- PosNegLPStandard
:::
:::{raw} latex
\end{wrapfigure}
:::

Learning Problems encode all information pertaining to the
classification task, i.e. the positive examples (that should be
covered by the learning result) and the negative examples (that should
not be covered by the learning result).

:::{raw} latex
\vspace{3\baselineskip}\WFclear
:::

## Search trees and nodes

:::{raw} latex
\begin{wrapfigure}{r}{.65\linewidth}\vspace{-40pt}\raggedleft%
:::
:::{uml}
package ontolearn.abstracts {
abstract AbstractNode
}
package ontolearn.abstracts {
abstract AbstractOEHeuristicNode
}
package ontolearn.search {
class Node
}
package ontolearn.search {
class OENode
}
package ontolearn.search {
class LBLNode
}
package ontolearn.abstract {
abstract LBLSearchTree
}
package ontolearn.search {
class SearchTreePriorityQueue
}

OENode <|-- LBLNode
AbstractNode <|-- Node
AbstractNode <|-- OENode
AbstractOEHeuristicNode <|-- OENode

LBLSearchTree <|-- SearchTreePriorityQueue
:::
:::{raw} latex
\end{wrapfigure}
:::

Nodes are tuples of concept and (typically) quality and other related
measures or information, and are used in the search process or
possibly to present the algorithm results.

:::{raw} latex
\vspace{7\baselineskip}\WFclear
:::

## OWL Hierarchies

:::{raw} latex
\begin{wrapfigure}{r}{.7\linewidth}\vspace{-40pt}\raggedleft%
:::
:::{uml}
package ontolearn.core.owl.hierarchy {
abstract AbstractHierarchy
}
package ontolearn.core.owl.hierarchy {
class ClassHierarchy
}
package ontolearn.core.owl.hierarchy {
class ObjectPropertyHierarchy
}
package ontolearn.core.owl.hierarchy {
class DatatypePropertyHierarchy
}

AbstractHierarchy <|-- ClassHierarchy
AbstractHierarchy <|-- ObjectPropertyHierarchy
AbstractHierarchy <|-- DatatypePropertyHierarchy
:::
:::{raw} latex
\end{wrapfigure}
:::

Hierarchies order their taxonomies into a sub- and super-kind
relation, for example sub-classes and super-classes.

:::{raw} latex
\vspace{3\baselineskip}\WFclear
:::

## Quality Functions

:::{raw} latex
\begin{wrapfigure}{r}{.65\linewidth}\vspace{-40pt}\raggedleft%%
:::
:::{uml}
package ontolearn.abstracts {
abstract AbstractScorer
}
package ontolearn.metrics {
class F1
}
package ontolearn.metrics {
class Accuracy
}
package ontolearn.metrics {
class Recall
}
package ontolearn.metrics {
class Precision
}

AbstractScorer <|-- Precision
AbstractScorer <|-- F1
AbstractScorer <|-- Accuracy
AbstractScorer <|-- Recall
:::
:::{raw} latex
\end{wrapfigure}
:::

You can choose one quality function which will be the quality of a
Node in the search process. The way the quality is measured typically
influences the heuristic and thus the direction of the search process.

:::{raw} latex
\vspace{\baselineskip}\WFclear
:::

## Component architecture

:::{raw} latex
\newenvironment{noparenv}{\def\par{}}{\endgraf}
\begin{noparenv}
\obeycr\null\hfill\smash{%
  \raisebox{\dimexpr-\height+\baselineskip+30pt}{%
:::
```{uml}
package ontolearn.base_concept_learner {
abstract BaseConceptLearner << AbstractNode >>
}
package ontolearn.concept_generator {
class ConceptGenerator
}
package ontolearn.abstracts {
abstract AbstractKnowledgeBase
}
package ontolearn.abstracts {
abstract AbstractLearningProblem
}
package ontolearn.abstracts {
abstract BaseRefinement << AbstractNode >>
}
package ontolearn.abstracts {
abstract AbstractScorer
}
package ontolearn.abstracts {
abstract AbstractHeuristic << AbstractHeuristicNode >>
}
package owlapy.model {
abstract OWLOntology
}
package owlapy.model {
abstract OWLReasoner
}

OWLReasoner *-- OWLOntology

AbstractKnowledgeBase *-- OWLOntology
AbstractKnowledgeBase *-- OWLReasoner
AbstractKnowledgeBase *-- ConceptGenerator

OWLReasoner -* ConceptGenerator

BaseRefinement *-- AbstractKnowledgeBase

BaseConceptLearner *-- AbstractKnowledgeBase
BaseConceptLearner *-- BaseRefinement : refinement_operator
BaseConceptLearner *-- AbstractScorer : quality_func
BaseConceptLearner *-- AbstractHeuristic : heuristic_func
```
:::{raw} latex
}}\end{noparenv}\par\vspace*{\dimexpr-\baselineskip-\parskip-\baselineskip}%
:::

:::{raw} latex
\begin{noparenv}
\parshape 5 0pt 0.4\textwidth
0pt 0.35\textwidth
0pt 0.3\textwidth
0pt 0.3\textwidth
0pt 0.3\textwidth
:::

Here you can see how the components depend on each other. All of these
components are required in order to run the concept learning algorithm.

:::{raw} latex
\end{noparenv}

\vspace{20\baselineskip}\WFclear

\endgroup
:::
