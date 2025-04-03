from abc import ABC, abstractmethod
from typing import List, Optional, Union
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.class_expression import (OWLClass, OWLClassExpression, OWLObjectUnionOf, OWLObjectIntersectionOf, 
                                     OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, OWLObjectComplementOf)
import pathlib
import json

class Expr(ABC):
    @abstractmethod
    def to_string(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

class Atoms(Expr):
    def __init__(self, name):
        self.name = name

    def to_string(self):
        return self.name
    
    def __repr__(self):
        return self.to_string()
    
    def to_dict(self):
        return {"type": OWLClass.__name__, "name": self.name}

class Not(Expr):
    def __init__(self, expr: Expr):
        self.expr = expr

    def to_string(self):
        return f"¬{self.expr.to_string()}"
    
    def __repr__(self):
        return self.to_string()
    
    def to_dict(self):
        return {"type": OWLObjectComplementOf.__name__, "expr": self.expr.to_dict()}

class And(Expr):
    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

    def to_string(self):
        return f"({self.left.to_string()} ⊓ {self.right.to_string()})"
    
    def __repr__(self):
        return self.to_string()
    
    def to_dict(self):
        return {"type": OWLObjectIntersectionOf.__name__, "left": self.left.to_dict(), "right": self.right.to_dict()}

class Or(Expr):
    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

    def to_string(self):
        return f"({self.left.to_string()} ⊔ {self.right.to_string()})"
    
    def __repr__(self):
        return self.to_string()
    
    def to_dict(self):
        return {"type": OWLObjectUnionOf.__name__, "left": self.left.to_dict(), "right": self.right.to_dict()}

class Exists(Expr):
    def __init__(self, role: str, filler: Expr):
        self.role = role
        self.filler = filler

    def to_string(self):
        return f"∃{self.role}.{self.filler.to_string()}"
    
    def __repr__(self):
        return self.to_string()
    
    def to_dict(self):
        return {"type": OWLObjectSomeValuesFrom.__name__, "role": self.role, "filler": self.filler.to_dict()}

class Forall(Expr):
    def __init__(self, role: str, filler: Expr):
        self.role = role
        self.filler = filler

    def to_string(self):
        return f"∀{self.role}.{self.filler.to_string()}"
    
    def __repr__(self):
        return self.to_string()
    
    def to_dict(self):
        return {"type": OWLObjectAllValuesFrom.__name__, "role": self.role, "filler": self.filler.to_dict()}

class ConceptAbstractSyntaxTreeBuilder:
    def __init__(self, knowledge_base_path: str, concept: Optional[Union[List[str], List[List[str]]]], relax_parentheses:bool=False):       
        self.tokens = self._sanitize_tokens((token for token in concept if token.strip() not in {'(', ')'}) 
                                            if relax_parentheses else concept) 
        self.index = 0
        self.length = len(self.tokens)
        self.knowledge_base = KnowledgeBase(path=knowledge_base_path)
        ontology = self.knowledge_base.ontology

        atoms_concepts = list(ontology.classes_in_signature())
        self.unique_atom_concept_names = {'⊤', '⊥'}.union({DLSyntaxObjectRenderer().render(atom) for atom in atoms_concepts})
        self.unique_roles = {relation.iri.get_remainder() for relation in ontology.object_properties_in_signature()}

        self.negation = {"¬"}
        self.binary_operations = {"⊓", "⊔"}
        self.quantifiers = {"∃", "∀"}
        self.cardinals = {"≤", "≥"}
        self.parenthesis = {"(", ")"}
        self.dot = {'.'}

        # validity construct wrt to top and bottom concept
        # not the invalidilty with a bottom concept?
        # handle for cardinality and other extensions

    def _current_token(self):
        return self.tokens[self.index] if self.index < self.length else None

    def _advance(self):
        self.index += 1

    def _sanitize_tokens(self, tokens):
        return [token.strip() for token in tokens]

    def parse(self):
        try:
            ast = self._parse_expression()
            if self.index != self.length:
                return None, {"error": "Extra tokens remain after generation.", 
                            "expr": self.render_tokens_as_expr(self.tokens[:self.index])}

            return ast, {"type": OWLClassExpression.__name__, "concept": ast.to_dict()}

        except Exception as e:
            # can be extented to the full tokens
            return None, {"error": str(e), "expr": self.render_tokens_as_expr(self.tokens[:self.index + 1])}

    def _parse_expression(self):
        node = self._parse_term()
        
        while self._current_token() in self.binary_operations:
            operation = self._current_token()

            self._advance()
            right = self._parse_term()
            node = And(node, right) if operation == '⊓' else Or(node, right)
        return node

    def _parse_term(self):
        token = self._current_token()
        if token is None:
            raise Exception("Unexpected end of tokens during generation.")

        if token in self.negation:
            self._advance()
            return Not(self._parse_term())

        if token in self.quantifiers:
            quantifier = token
            self._advance()
            if self._current_token() not in self.unique_roles:
                raise Exception(f"Expected role after quantifier, got '{self._current_token()}'.")
            
            role = self._current_token()
            self._advance()
            if self._current_token() != '.': # list(self.dot)[0]
                raise Exception("Expected '.' after role in quantified expression.")
            
            self._advance()
            filler = self._parse_term()
            return Exists(role, filler) if quantifier == '∃' else Forall(role, filler)

        if token == '(':
            self._advance()
            expression = self._parse_expression()

            if self._current_token() != ')':
                raise Exception("Expected ')' after expression.")
            self._advance()
            return expression

        if token in self.unique_atom_concept_names:
            self._advance()
            return Atoms(token)

        raise Exception(f"Unexpected token '{token}' at position {self.index}.")
    
    def render_tokens_as_expr(self, tokens):
        formatted_tokens = []
        i = 0

        while i < len(tokens):
            token = tokens[i]

            if token in self.quantifiers:
                formatted_tokens.append(token)
                i += 1
                if i < len(tokens):
                    formatted_tokens.append(tokens[i])
                    i += 1
                    if i < len(tokens) and tokens[i] in self.dot:
                        formatted_tokens.append(tokens[i])
                        i += 1
                continue

            if token in self.negation:
                formatted_tokens.append(token)
            elif token in self.binary_operations:
                formatted_tokens.append(f" {token} ")
            elif token in self.parenthesis:
                formatted_tokens.append(token)
            else:
                if formatted_tokens and formatted_tokens[-1] not in {"(", " "}:  
                    formatted_tokens.append(token)  
                else:
                    formatted_tokens.append(token)
            i += 1
        return "".join(formatted_tokens).replace("  ", " ").strip()

def generate_class_expression(kb_path:str, leaners_prediction = None, save_as_json: bool=False, relax_parentheses=True):
    if not leaners_prediction:

        # token_sequence = ['¬', 'hasSibling', '.', '(', 'Thing', '⊓', '∃', 'hasChild', '.', 'Female', '⊓', 'Grandfather'] 
        # token_sequence =['¬', ' Brother ', ' ) ']
        # token_sequence = ['Female','⊓','(','∃','hasSibling','.','Father',')']
        # token_sequence = ['∃','hasSibling','.','Father',')']
        # token_sequence = ['Female','⊓','¬','(','∃','hasSibling','.','Father',')','⊓','(','∃','married','.','Brother',')']
        # token_sequence = ['Thing','⊓','∀','hasChild ','.','Brother','⊓','¬','Thing','⊓','Father']
        # token_sequence = ['Grandfather', '⊓', '¬', '∀', 'hasSibling', '.', 'Brother', '⊓', 'Female', '⊓', 'Brother']
        # token_sequence = ['(', '∀', 'married', '.', 'Thing', '⊔', 'Person', '⊔', '(', '∃', 'married', '.', 'Father']
        # token_sequence = ['∀', 'hasChild', '.', 'Father', '⊔', '¬', '.', 'Brother', '⊔', 'Person', ')']
        # token_sequence = ['Father', '⊓', 'Grandfather', '⊔', '∀', 'hasParent', '.', '¬', '∀', 'hasChild', '.', 'Brother']
        # token_sequence = ['Grandfather', '⊓', 'Brother', '⊓', '(', 'Thing', '⊔', '∃', 'married', '.', 'Grandfather', ')']
        # token_sequence = ['Father', '⊔', 'Female', '⊓', 'Female', '⊔', 'Thing', '⊔', 'Female', ')']
        token_sequence = ['∃', 'hasSibling', '.', 'Brother', '⊓', ' Father ', '⊓', '∀', ' hasSibling', '.', 'Grandfather', ')']

    try:
        builder = ConceptAbstractSyntaxTreeBuilder(knowledge_base_path=kb_path, concept=token_sequence, relax_parentheses=relax_parentheses)
        concept, result = builder.parse()

        if concept is not None:
            print("Constructed concept abstract syntax tree:", concept)

            if save_as_json:
                with open("concept_abstract_syntax_tree.json", "w") as f:
                    json.dump(result, f, indent=2)
                print("Saving... as concept_abstract_syntax_tree.json")
        else:
            print("Error constructing the concept abstract syntax tree:", result["error"])
            print("Invalid concept representation", result["expr"])

    except Exception as e:
        print("Unexpected error constructing the concept abstract syntax tree:", e)

if __name__ == "__main__":
    knowledge_base_path = pathlib.Path(__file__).parent.parent.parent.resolve()._str + "/data/KGs/Family/family-benchmark_rich_background.owl"
    generate_class_expression(knowledge_base_path, save_as_json=True)
    

    '''
    TODO
        - * assumed the ordering of the prediction is given
        - you can add the target_goal concept (groupby)

        - Get predicitons from all learners | nces 12 roces [By today]
        - recursive handlinding to enforce its validility [Fri]
        - new pushable branch 
        - present to Demir for feedback on [Mon.]

        - heirachy-awareness
    '''

