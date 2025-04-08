from collections import Counter
from typing import Set, Generator, Dict, Any
from owlapy.class_expression import *
from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner
from owlapy.owl_property import (
    OWLDataProperty,
    OWLObjectInverseOf,
    OWLObjectProperty,
    OWLProperty,
)
from owlapy.iri import IRI
from owlapy import owl_expression_to_dl
from graphviz import Digraph





class ReasoningNode:
    """ A node representing a step in the reasoning tree """
    def __init__(self, expression: Any, instances: Set[Any] = None):
        self.expression = expression
        self.instances = instances if instances is not None else set()
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def to_dict(self):
        """ Convert tree to a dictionary format for easy visualization """
        return {
            "expression": str(self.expression),
            "instances": [str(inst) for inst in self.instances],
            "children": [child.to_dict() for child in self.children]
        }


def build_reasoning_tree(reasoner, expression: Any) -> ReasoningNode:
    """ Builds a tree representing the reasoning process without modifying the reasoner """
    root = ReasoningNode(expression)
    _populate_tree(reasoner, root, expression)
    return root


def _populate_tree(reasoner, node: ReasoningNode, expression: Any):
    """ Recursively populates the reasoning tree using the existing reasoner """
    if isinstance(expression, (OWLClass, OWLObjectOneOf)):
        node.instances = set(reasoner.instances(expression))
    elif isinstance(expression, OWLObjectUnionOf):
        for op in expression.operands():
            child_node = ReasoningNode(op)
            node.add_child(child_node)
            _populate_tree(reasoner, child_node, op)
        node.instances = set.union(*(child.instances for child in node.children))
    elif isinstance(expression, OWLObjectIntersectionOf):
        for op in expression.operands():
            child_node = ReasoningNode(op)
            node.add_child(child_node)
            _populate_tree(reasoner, child_node, op)
        node.instances = set.intersection(*(child.instances for child in node.children))
    elif isinstance(expression, OWLObjectComplementOf):
        child_node = ReasoningNode(expression.get_operand())
        node.add_child(child_node)
        _populate_tree(reasoner, child_node, expression.get_operand())
        all_individuals = {i for i in reasoner.individuals_in_signature()}
        node.instances = all_individuals - child_node.instances
    elif isinstance(expression, OWLObjectSomeValuesFrom):
        object_property = expression.get_property()
        filler_expression = expression.get_filler()
        filler_node = ReasoningNode(filler_expression)
        node.add_child(filler_node)
        _populate_tree(reasoner, filler_node, filler_expression)

        result = Counter()
        for object_individual in filler_node.instances:
            subjects = reasoner.get_individuals_with_object_property(
                obj=object_individual,
                object_property=object_property
            )
            result.update(subjects)

        node.instances = {ind for ind, count in result.items() if count >= 1}
    elif isinstance(expression, OWLObjectAllValuesFrom):
        object_property = expression.get_property()
        filler_expression = expression.get_filler()
        complement_node = ReasoningNode(OWLObjectComplementOf(filler_expression))
        node.add_child(complement_node)
        _populate_tree(reasoner, complement_node, OWLObjectComplementOf(filler_expression))

        negation_some = OWLObjectSomeValuesFrom(object_property, OWLObjectComplementOf(filler_expression))
        negation_some_node = ReasoningNode(negation_some)
        node.add_child(negation_some_node)
        _populate_tree(reasoner, negation_some_node, negation_some)

        all_individuals = {i for i in reasoner.individuals_in_signature()}
        node.instances = all_individuals - negation_some_node.instances
    elif isinstance(expression, OWLObjectMaxCardinality):
        object_property = expression.get_property()
        filler_expression = expression.get_filler()
        cardinality = expression.get_cardinality()

        filler_node = ReasoningNode(filler_expression)
        node.add_child(filler_node)
        _populate_tree(reasoner, filler_node, filler_expression)

        subject_counts = {ind: 0 for ind in reasoner.individuals_in_signature()}
        for object_ind in filler_node.instances:
            subjects = reasoner.get_individuals_with_object_property(obj=object_ind, object_property=object_property)
            for subj in subjects:
                subject_counts[subj] += 1

        node.instances = {ind for ind, count in subject_counts.items() if count <= cardinality}
    else:
        raise NotImplementedError(f"Instances for {type(expression)} are not implemented yet")
    


def format_instances(instances):
    """Format instances: Show at most 10, followed by '...' if more exist."""
    instance_names = [inst.str.split('#')[-1] for inst in instances]
    if len(instance_names) >3:
        instance_names = instance_names[:3]
        instance_names.append('...')
        return f"{instance_names}"
    return str(set(instance_names))


def visualize_reasoning_tree_graphviz(root: ReasoningNode, filename="reasoning_tree"):
    """ Uses Graphviz to visualize the reasoning tree with the concept at the top 
        and its retrieved instances displayed separately. """
    dot = Digraph(format="png")  # Create a directed graph

    root_id = str(id(root))
    
    # Root node: Show the full concept at the top in red
    concept_label = f"{owl_expression_to_dl(root.expression)}"
    dot.node(root_id, concept_label, shape="box", style="bold", color='red')

    # Instances node: Show retrieved instances in a separate node
    instance_label = format_instances(root.instances)
    instance_node_id = f"{root_id}_instances"
    dot.node(instance_node_id, instance_label, shape="ellipse", style="bold", color="red")

    # Add a horizontal edge from concept node to instances
    dot.edge(root_id, instance_node_id, label="output", dir="forward", arrowhead="normal", constraint="false", style="dashed")

    def _add_nodes(node, parent=None, edge_label=""):
        """ Recursively add nodes to the Graphviz object """
        node_id = str(id(node))

        if node is not root:  # Skip re-adding the root
            if not node.children:  
                # Leaf nodes: If it's a final retrieved instance, highlight it in red
                node_label = format_instances(node.instances)
                dot.node(node_id, node_label, shape="ellipse", style="bold", color='blue')
            else:
                # Normal child nodes: Show only instances
                node_label = format_instances(node.instances)
                dot.node(node_id, node_label, shape="ellipse")

            if parent:
                dot.edge(str(id(parent)), node_id, label=edge_label)  # Label edges with the concept breakdown

        for child in node.children:
            _add_nodes(child, node, edge_label=owl_expression_to_dl(child.expression))

    _add_nodes(root)
    dot.render(filename, view=False)



def visualize_reasoning_tree(root: ReasoningNode):
    """ Print reasoning tree in a structured way with at most 5 instances displayed per node. """
    def _print_tree(node, level=0):
        instances = format_instances(node.instances)
        print(" " * (level * 4) + f"- {owl_expression_to_dl(node.expression)} -----> {instances}")
        for child in node.children:
            _print_tree(child, level + 1)

    _print_tree(root)



EBR = TripleStoreNeuralReasoner(path_of_kb="KGs/Family/father.owl", gamma=0.9)
reasoner = EBR  
A =OWLObjectAllValuesFrom(property=OWLObjectProperty(IRI('http://example.com/father#','hasChild')),filler=OWLObjectComplementOf(OWLClass(IRI('http://example.com/father#','male'))))

# A= OWLClass(IRI('http://example.com/father#','person'))
B = OWLObjectComplementOf(OWLClass(IRI('http://example.com/father#','person')))
concept = OWLObjectIntersectionOf([A, A])  # Example OWL expression
reasoning_tree = build_reasoning_tree(reasoner, concept)
visualize_reasoning_tree(reasoning_tree) 
visualize_reasoning_tree_graphviz(reasoning_tree, "reasoning_tree_output")