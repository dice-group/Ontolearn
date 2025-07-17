from abc import ABC, abstractmethod
import random
from random import choice
from typing import List, Optional
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.class_expression import (OWLClass, OWLClassExpression, OWLObjectUnionOf, OWLObjectIntersectionOf, 
                                     OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, OWLObjectComplementOf, OWLCardinalityRestriction)
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
    
class Cardinality(Expr):
    def __init__(self, kind: str, n: int, role: str, filler: Expr):
        self.kind = kind
        self.n = n
        self.role = role
        self.filler = filler
    def to_string(self):
        return f"{self.kind}{self.n} {self.role}.{self.filler.to_string()}"
    def __repr__(self):
        return f"({self.kind}{self.n} {self.role}.{self.filler})"
    def to_dict(self):
        return {
            "type": OWLCardinalityRestriction.__name__,
            "kind": self.kind,
            "n": self.n,
            "role": self.role,
            "filler": self.filler.to_dict()
        }

class ConceptAbstractSyntaxTreeBuilder:
    def __init__(self, knowledge_base_path: str, max_length: Optional[int] = None): 
        assert isinstance(knowledge_base_path, str) and knowledge_base_path.strip(), "Knowledge base path is required"
        
        self.knowledge_base = KnowledgeBase(path=knowledge_base_path)
        self.max_length = max_length
        
        ontology = self.knowledge_base.ontology
        atoms_concepts = list(ontology.classes_in_signature())
        self.unique_atom_concept_names = {'⊤', '⊥'}.union({DLSyntaxObjectRenderer().render(atom) for atom in atoms_concepts})
        # self.atom_concepts_with_negation = self.unique_atom_concept_names | {("¬", atom) for atom in self.unique_atom_concept_names}
        self.unique_roles = {relation.iri.get_remainder() for relation in ontology.object_properties_in_signature()}

        self.negation = {"¬"}
        self.binary_ops = {"⊓", "⊔"}
        self.quantifiers = {"∃", "∀"}
        self.cardinals = {"≤", "≥"}
        self.parenthesis = {"(", ")"}
        self.dot = {'.'}
        self.digits = {str(i) for i in range(10)}

        # TODO: handle concrete roles and other extended vocabs
        self.vocabs = self.unique_atom_concept_names | self.unique_roles | self.binary_ops | self.negation | self.quantifiers | self.parenthesis | self.dot | self.cardinals | self.digits
        self.atom_concepts_with_negation = None

    def _negate_unique_atomic_concepts(self, replace_with_negation=False):
        if replace_with_negation:
            return self.unique_atom_concept_names | {("¬", atom) for atom in self.unique_atom_concept_names}
        return self.unique_atom_concept_names
    
    def _current_token(self):
        return self.tokens[self.index] if self.index < self.length else None

    def _advance(self):
        self.index += 1

    def _sanitize_tokens(self, tokens):
        return [token for token in (t.strip() for t in tokens) if token]
    
    def _strip_trailing_parentheses(self, concept_str:str):
        if concept_str.startswith('(') and concept_str.endswith(')'):
            concept_str = concept_str[1:-1]
        return concept_str
    
    def _fix_mid_tokens_errors(self, tokens: list[str]) -> list[str]:
        container = []
        i = 0

        while i < len(tokens):
            prev_token = tokens[i - 1] if i - 1 >= 0 else None
            token = tokens[i]
            next_token = tokens[i + 1] if i + 1 < len(tokens) else None
            next_next_token = tokens[i + 2] if i + 2 < len(tokens) else None

            if (prev_token == '(' and token in self.binary_ops and 
                next_token == '.' and next_next_token):
                if next_next_token in self.unique_atom_concept_names | self.negation:
                    i += 2
                else:
                    container.append(choice(list(self.unique_atom_concept_names)))
                    container.append(token)
                    i += 1
                i += 1
                continue

            if prev_token in self.unique_roles and token == '(':
                if next_token:
                    if next_token in self.unique_atom_concept_names | self.negation:
                        container.append('.')
                        container.append(token)
                    elif next_token in self.dot and next_next_token and next_next_token in self.unique_atom_concept_names | self.negation:
                        i += 1
                        container.append('.')
                        container.append(token)
                    i += 1
                    continue

            if prev_token == '(' and token == '.' and next_token == ')':
                container.append(choice(list(self.unique_atom_concept_names)))
                i += 1
                continue
            
            if token == prev_token and token in self.binary_ops | self.quantifiers | self.dot:
                i += 1
                continue

            if prev_token in self.binary_ops and token in self.binary_ops:
                i += 1
                continue

            if token == '.' and prev_token in {'(', ')'}:
                i += 1
                continue

            if prev_token == '(' and token in self.binary_ops:
                i += 1
                continue

            if prev_token in self.binary_ops and token == ')':
                container.pop() 
                container.append(token)
                i += 1
                continue

            if prev_token == ')' and token == '(':
                container.append(choice(list(self.binary_ops)))
                container.append(token)
                i += 1
                continue
           
            if (prev_token == ')' and token not in self.binary_ops and next_token == '('):
                container.append(choice(list(self.binary_ops)))
                i += 1  
                continue

            if (prev_token in self.unique_atom_concept_names | {')'} and 
                token in self.unique_atom_concept_names | self.negation | {'('} | self.dot):
                container.append(choice(list(self.binary_ops)))
                if token in self.dot:
                    i += 1
                else:
                    container.append(token)
                    i += 1
                continue

            if prev_token in self.binary_ops and token in self.quantifiers and next_token in self.binary_ops:
                container.append(choice(list(self.unique_atom_concept_names)))
                i += 1
                continue

            container.append(token)
            i += 1
        return container
    
    def _postprocess_tail_fix(self, tokens: list[str], max_length: int) -> list[str]:
        def is_incomplete_tail(toks):
            if not toks:
                return True
            return toks[-1] in self.binary_ops | self.quantifiers | self.negation | {'.', '(', *self.unique_roles}

        def minimal_completion_after(toks):
            last = toks[-1] if toks else None
            remaining = max_length - len(toks)

            if last is None:
                return [choice(list(self.unique_atom_concept_names))]

            if last in self.binary_ops:
                return [choice(list(self.unique_atom_concept_names))] if remaining >= 1 else []

            if last in self.quantifiers:
                return [choice(list(self.unique_roles)), '.', choice(list(self.unique_atom_concept_names))] if remaining >= 3 else []

            # if last in self.digits:
            #     return [choice(list(self.unique_roles)), '.', choice(list(self.unique_atom_concept_names))] if remaining >= 3 else []

            # if last in self.cardinals:
            #     return ['1', choice(list(self.unique_roles)), '.', choice(list(self.unique_atom_concept_names))]

            if last in self.unique_roles:
                if len(toks) >= 2 and toks[-2] in self.quantifiers:
                    return ['.', choice(list(self.unique_atom_concept_names))] if remaining >= 2 else []
                return []

            if last == '.':
                if len(toks) >= 2 and toks[-2] in self.unique_roles:
                    return [choice(list(self.unique_atom_concept_names))] if remaining >= 1 else []
                return []

            if last in self.negation:
                return [choice(list(self.unique_atom_concept_names))] if remaining >= 1 else []

            if last == '(':
                return [choice(list(self.unique_atom_concept_names)), ')'] if remaining >= 2 else []

            return []

        if len(tokens) == max_length and not is_incomplete_tail(tokens):
            return tokens

        while len(tokens) < max_length and is_incomplete_tail(tokens):
            patch = minimal_completion_after(tokens)
            if not patch:
                break
            tokens += patch
            tokens = tokens[:max_length]

        if len(tokens) == max_length and is_incomplete_tail(tokens):
            for i in reversed(range(len(tokens))):
                if not is_incomplete_tail(tokens[:i]):
                    tokens = tokens[:i]
                    break

        return tokens
    
    def balance_flatten_parentheses(self, sequences: list[str], max_length: int = None) -> list[str]:
        stack, result = [], []

        for sequence in sequences:
            if sequence == '(':
                stack.append(len(result))
                result.append(sequence)
            elif sequence == ')':
                if stack:
                    stack.pop()
                    result.append(sequence)
            else:
                result.append(sequence)

        if stack:
            if max_length is not None:
                for pos in reversed(stack):
                    if len(result) < max_length:
                        result.append(')')
                    else:
                        result.pop(pos)
            else:
                for pos in reversed(stack):
                    result.pop(pos)

        i = 0
        while i < len(result) - 3:
            if result[i] == '(' and result[i + 1] == '(':
                j = i + 2
                depth = 1
                while j < len(result) and depth > 0:
                    if result[j] == '(':
                        depth += 1
                    elif result[j] == ')':
                        depth -= 1
                    j += 1

                if j < len(result) and result[j] == ')':
                    result = result[:i+1] + result[i+2:j] + result[j+1:]
                    continue
            i += 1

        return result

    def parse(self, token_sequence:List[str], relax_parentheses:bool=False, enforce_validity:Optional[bool]=False, replace_with_negation:bool=False):
        assert isinstance(token_sequence, list) and len(token_sequence) > 0, "Token sequence must be a non-empty list of non-empty strings"
        
        tokens = (token for token in token_sequence if token.strip() not in {'(', ')'}) if relax_parentheses else token_sequence
        self.tokens = self._sanitize_tokens(tokens)

        if not self.max_length:
            self.max_length = len(self.tokens) + 10

        if enforce_validity:
            self.tokens = self._fix_mid_tokens_errors(self._enforce(replace_with_negation=replace_with_negation))
            self.tokens = self.balance_flatten_parentheses(self._postprocess_tail_fix(self.tokens.copy(), self.max_length))

        self.index = 0
        self.length = len(self.tokens)

        try:
            ast = self._parse_expression()
            if self.index != self.length:
                return None, {"error": "Extra tokens remain after generation.", "expr": self.render_tokens_as_class_expr(self.tokens[:self.index])}
            return self._strip_trailing_parentheses(str(ast)), {"type": OWLClassExpression.__name__, "concept": ast.to_dict()}

        except Exception as e:
            # can be extented to the full tokens
            return None, {"error": str(e), "expr": self.render_tokens_as_class_expr(self.tokens[:self.index + 1])}

    def _parse_expression(self):
        node = self._parse_term()
        
        while self._current_token() in self.binary_ops:
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
        
        if token in self.cardinals:
            kind = token
            self._advance()

            num_token = self._current_token()
            if num_token is None or not num_token.isdigit():
                raise Exception(f"Expected number after '{kind}', got '{num_token}'.")
            number = int(num_token)
            self._advance()

            role = self._current_token()
            if role not in self.unique_roles:
                raise Exception(f"Expected role after number in cardinality, got '{role}'.")
            self._advance()

            if self._current_token() != '.':
                raise Exception("Expected '.' after role in cardinality.")
            self._advance()

            filler = self._parse_term()
            return Cardinality(kind, number, role, filler)

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
    
    def render_tokens_as_class_expr(self, _tokens):
        formatted_tokens = []
        indx = 0

        while indx < len(_tokens):
            token = _tokens[indx]

            if token in self.quantifiers:
                formatted_tokens.append(token)

                indx += 1
                if indx < len(_tokens):
                    formatted_tokens.append(_tokens[indx])

                    indx += 1
                    if indx < len(_tokens) and _tokens[indx] in self.dot:
                        formatted_tokens.append(_tokens[indx])
                        indx += 1
                continue

            if token in self.negation:
                formatted_tokens.append(token)
            elif token in self.binary_ops:
                formatted_tokens.append(f" {token} ")
            elif token in self.parenthesis:
                formatted_tokens.append(token)
            else:
                if formatted_tokens and formatted_tokens[-1] not in {"(", " "}:  
                    formatted_tokens.append(token)  
                else:
                    formatted_tokens.append(token)
            indx += 1
        return "".join(formatted_tokens).replace("  ", " ").strip()
    
    def _lookahead_grammar_strategy(self, context_tokens):
        if not context_tokens:
            return self.negation | self.quantifiers | self.cardinals | self.unique_atom_concept_names | {'('}
        
        last = context_tokens[-1]
        
        if last in self.quantifiers:
            return self.unique_roles

        if last in self.cardinals:
            return self.digits

        if last in self.digits:
            return self.digits | self.unique_roles

        if last in self.unique_roles:
            return self.dot

        if last == '.':
            return self.negation | self.quantifiers | self.cardinals | self.unique_atom_concept_names | {'('}

        if last == '(':
            return self.negation | self.quantifiers | self.cardinals | self.unique_atom_concept_names | {'('}

        if last in self.unique_atom_concept_names:
            return self.binary_ops | {')'}

        if last in self.binary_ops:
            return self.negation | self.quantifiers | self.cardinals | self.unique_atom_concept_names | {'('}
        
        if last in self.negation:
            return self.unique_atom_concept_names | self.quantifiers | self.cardinals | {'('}

        if last == ')':
            return self.binary_ops | {')'}

        return self.vocabs

    
    def _is_valid_next_token(self, token, context_tokens):
        valid_candidate_tokens = sorted(self._lookahead_grammar_strategy(context_tokens))
        return token in valid_candidate_tokens

    def _enforce(self, max_length:Optional[int]=None, replace_with_negation:bool=False):
        if not max_length:
            max_length = len(self.tokens) + 50

        self.atom_concepts_with_negation = self._negate_unique_atomic_concepts(replace_with_negation=replace_with_negation)

        corrected_tokens, curr_valid_cum = [], []
        choices, cap = None, None
        cap_curr_val = {'atom': 1, 'neg_atom': 2, 'role': 3, 'roleCard': 4}

        indx = 0
        while indx < len(self.tokens) and len(corrected_tokens) < max_length:
            token = self.tokens[indx]
            prev_token = self.tokens[indx-1] if indx != 0 else None 
            ahead_token = self.tokens[indx+1] if (indx+1) < len(self.tokens) else None
            next_ahead_token = self.tokens[indx+2] if (indx+2) < len(self.tokens) else None

            if not curr_valid_cum and token in self.quantifiers | self.negation | self.binary_ops | self.parenthesis | self.unique_atom_concept_names | self.dot | self.unique_roles:
                if indx != 0 and prev_token:
                    if token in self.negation:
                        if prev_token in self.negation | self.unique_atom_concept_names:
                            ops_choice = random.choice(list(self.binary_ops))
                            corrected_tokens.extend([ops_choice, token])
                            indx +=1
                            continue
                    elif token in self.dot and prev_token not in self.unique_roles:
                            if ahead_token and ahead_token not in {')'}:
                                if ahead_token in self.unique_atom_concept_names | self.negation | {'('}:
                                    ops_choice = random.choice(list(self.binary_ops))
                                    corrected_tokens.append(ops_choice)
                                    indx +=1
                                    continue
                    elif token in self.unique_roles and prev_token not in self.quantifiers:
                        if ahead_token and prev_token in {')'}:
                            ops_choice = random.choice(list(self.binary_ops))
                            corrected_tokens.append(ops_choice)

                            if ahead_token in self.dot:
                                quant_choice = random.choice(list(self.quantifiers))
                                corrected_tokens.extend([quant_choice, token, ahead_token])
                                curr_valid_cum.extend([token, ahead_token])
                                cap = 3
                                choices = self.atom_concepts_with_negation
                                indx += 2
                                continue
                            elif ahead_token in self.unique_atom_concept_names:
                                indx += 1
                                continue
                            elif ahead_token in {')'}:
                                indx +=2
                                continue
                            elif ahead_token and ahead_token in self.negation | self.unique_atom_concept_names:
                                if prev_token in self.binary_ops | self.parenthesis:
                                    indx += 1
                                    continue
                    elif token in self.quantifiers and prev_token not in self.binary_ops:
                        ops_choice = random.choice(list(self.binary_ops))

                        if prev_token in self.unique_atom_concept_names and (ahead_token in self.unique_roles or 
                        (next_ahead_token and next_ahead_token in self.unique_roles)):
                            corrected_tokens.extend([ops_choice, token])
                            choices = self.unique_roles
                            indx += 1
                            continue
                    elif token in {')'} and prev_token not in self.unique_atom_concept_names:
                        if prev_token in self.quantifiers and ahead_token in self.unique_roles:
                            self.tokens.pop(indx)
                            continue
                    elif token in {'('} and prev_token in {')'} | self.unique_atom_concept_names:
                        ops_choice = random.choice(list(self.binary_ops))
                        corrected_tokens.extend([ops_choice, token])
                        choices = self.atom_concepts_with_negation
                        indx += 1
                        continue
   
                if ahead_token:
                    if token in self.quantifiers:
                        if ahead_token in self.dot:
                            role_choice = random.choice(list(self.unique_roles))
                            corrected_tokens.extend([token, role_choice])
                            curr_valid_cum.append(role_choice)
                            cap = 3
                            indx += 1
                            continue
                        elif ahead_token in self.parenthesis:
                            if prev_token and prev_token in {')'}:
                                corrected_tokens.append(random.choice(list(self.binary_ops)))

                            role_choice = random.choice(list(self.unique_roles))
                            corrected_tokens.extend([token, role_choice])
                            curr_valid_cum.append(role_choice)
                            cap = 3
                            choices = self.dot
                            indx += 1
                            continue
                        elif ahead_token in self.binary_ops:
                            atomic_choice = random.choice(list(self.atom_concepts_with_negation))
                            if isinstance(atomic_choice, tuple):
                                corrected_tokens.extend(atomic_choice)
                            else:
                                corrected_tokens.append(atomic_choice)
                            indx += 1
                            continue
                        elif ahead_token in self.unique_atom_concept_names:
                            self.tokens.pop(indx)
                            continue

                    if token in self.binary_ops:
                        if ahead_token in self.binary_ops:
                            atomic_choice = random.choice(list(self.atom_concepts_with_negation))
                            if isinstance(atomic_choice, tuple):
                                _token = [token] + list(atomic_choice)
                            else:
                                _token = [token, atomic_choice]

                            corrected_tokens.extend(_token)
                            indx += 2
                            continue
                        elif ahead_token == ')':
                            atomic_choice = random.choice(list(self.atom_concepts_with_negation))
                            if isinstance(atomic_choice, tuple):
                                _token = [token] + list(atomic_choice)
                            else:
                                _token = [token, atomic_choice]
                            corrected_tokens.extend(_token)
                            indx += 1
                            continue
                        elif ahead_token in self.dot:
                            if next_ahead_token:

                                if next_ahead_token in self.parenthesis | self.unique_atom_concept_names:
                                    corrected_tokens.append(token)
                                    indx += 2
                                    continue

                    if token == ')':
                        if indx != 0:
                            if prev_token and prev_token in self.unique_atom_concept_names:
                                corrected_tokens.append(token)
                                indx += 1
                                continue

                            elif ahead_token in self.parenthesis | self.quantifiers | self.dot:
                                if ahead_token in {')'}:
                                    indx += 1
                                    continue

                                ops_choice = random.choice(list(self.binary_ops))
                                corrected_tokens.extend([token, ops_choice])
                                indx += 1
                                continue
                        else:
                            if ahead_token in self.unique_atom_concept_names:
                                corrected_tokens.append('(')
                                indx +=1
                                continue
                            elif ahead_token in self.binary_ops:
                                atomic_choice = random.choice(list(self.atom_concepts_with_negation))
                                if isinstance(atomic_choice, tuple):
                                    _token = list(atomic_choice)
                                else:
                                    _token = [atomic_choice]
                                corrected_tokens.extend(['('] + _token)
                                indx += 1
                                continue

                    if token in self.negation:
                        if ahead_token in self.quantifiers:
                            if next_ahead_token and next_ahead_token in self.unique_roles:
                                atomic_choice = random.choice(list(self.unique_atom_concept_names))
                                ops_choice = random.choice(list(self.binary_ops))
                                corrected_tokens.extend([token, atomic_choice, ops_choice])
                                indx += 1
                                continue
                            else:
                                corrected_tokens.append(token)
                                if ahead_token not in self.unique_atom_concept_names | self.negation:
                                    self.tokens[indx + 1] = random.choice(list(self.unique_atom_concept_names))
                                indx += 1
                                continue
                        elif ahead_token in self.unique_roles:
                            if next_ahead_token and next_ahead_token in {'.'}:
                                quant_choice = random.choice(list(self.quantifiers))
                                self.tokens[indx] = quant_choice
                                corrected_tokens.append(quant_choice)
                                indx +=1
                                continue
                            
                            corrected_tokens.append(token)
                            atomic_choice = random.choice(list(self.unique_atom_concept_names))
                            self.tokens[indx+1] = atomic_choice
                            indx +=1
                            continue
                    
                    if token in self.unique_roles and prev_token not in self.quantifiers:
                        if prev_token in self.unique_atom_concept_names and ahead_token in self.unique_atom_concept_names:
                            ops_choice = random.choice(list(self.binary_ops))
                            corrected_tokens.append(ops_choice)
                            indx +=1
                            continue
            
            if token in self.quantifiers:
                if curr_valid_cum:
                    if cap == 3 and len(curr_valid_cum) == 2:
                        atomic_choice = random.choice(list(choices))

                        if isinstance(atomic_choice, tuple):
                            corrected_tokens.extend(atomic_choice)
                            self.tokens[indx] = atomic_choice[1]
                        else:
                            corrected_tokens.append(atomic_choice)
                            self.tokens[indx] = atomic_choice
                        
                        cap, choices, curr_valid_cum = None, None, []
                        indx +=1
                        continue

            if token == '(':
                if ahead_token or prev_token:
                    if ahead_token in self.negation | self.binary_ops | {')'} | self.dot:
                        if not curr_valid_cum:
                            if indx == 0:
                                if ahead_token not in self.negation | self.unique_atom_concept_names:
                                    token = random.choice(list(self.atom_concepts_with_negation))
                                    if isinstance(token, tuple):
                                        corrected_tokens.extend(token)
                                    else:
                                        corrected_tokens.append(token)
                                    indx += 1
                                    continue
                                else:
                                    if next_ahead_token:

                                        if ahead_token in self.negation:
                                            if next_ahead_token not in self.unique_atom_concept_names:
                                                atomic_choice = random.choice(list(self.unique_atom_concept_names))

                                                if next_ahead_token in self.binary_ops:
                                                    corrected_tokens.extend([token, atomic_choice])
                                                    ahead_token = atomic_choice
                                                elif next_ahead_token in self.quantifiers:
                                                    ops_choice = random.choice(list(self.binary_ops))
                                                    corrected_tokens.extend([token, atomic_choice, ops_choice])
                                                    ahead_token = ops_choice
                                                indx +=2
                                                continue
                            else:
                                if ahead_token in self.binary_ops:
                                    if next_ahead_token and next_ahead_token in self.negation | self.unique_atom_concept_names:
                                        if prev_token and prev_token in self.binary_ops:
                                            indx +=2
                                            continue
                                elif ahead_token in {')'} | self.dot :
                                    atomic_choice = random.choice(list(self.atom_concepts_with_negation))
                                    ops_choice = random.choice(list(self.binary_ops))

                                    if isinstance(atomic_choice, tuple):
                                        _token = list(atomic_choice)
                                    else:
                                        _token = [atomic_choice]

                                    if prev_token and prev_token not in self.binary_ops:
                                        _token = [ops_choice] + _token

                                    _token = [token] + _token 
                                    
                                    corrected_tokens.extend(_token if ahead_token in self.dot else _token + [')'])
                                    indx += 2
                                    continue          
                        else:
                            if ahead_token in {')'}:
                                if all(choice in self.atom_concepts_with_negation for choice in choices):
                                    atomic_choice = random.choice(list(choices))
                                else:
                                    atomic_choice = random.choice(list(self.atom_concepts_with_negation))

                                if isinstance(atomic_choice, tuple):
                                    _token = [token] + list(atomic_choice)
                                else:
                                    _token = [token, atomic_choice]

                                if prev_token and prev_token in self.unique_roles:
                                    _token = ['.'] + _token
                                
                                corrected_tokens.extend(_token)
                                cap, choices, curr_valid_cum = None, None, []

                                indx += 1
                                continue
                            elif ahead_token in self.negation:
                                if prev_token and prev_token in self.unique_roles:
                                    if next_ahead_token and next_ahead_token in self.unique_atom_concept_names:
                                        corrected_tokens.extend([token, '.', ahead_token, next_ahead_token])
                                        cap, choices, curr_valid_cum = None, None, []
                                        indx += 3
                                        continue
                                elif (indx - 2 != 0 and self.tokens[indx-2] in self.unique_roles) and prev_token and prev_token in self.dot:
                                    corrected_tokens.extend([token, ahead_token])

                                    if next_ahead_token and (indx + 3) < len(self.tokens):
                                        if next_ahead_token in self.unique_atom_concept_names and self.tokens[indx + 3] in {')'}:
                                            corrected_tokens.extend([next_ahead_token, self.tokens[indx + 3]])
                                            indx += 2
                                    indx += 1
                                    cap, choices, curr_valid_cum = None, None, []
                                    continue
                                elif cap == 3 and len(curr_valid_cum) == 1:
                                    dot_choice = list(choices) if choices in self.dot else '.'
                                    corrected_tokens.append(dot_choice)

                                atomic_choice = random.choice(list(self.unique_atom_concept_names))
                                corrected_tokens.extend([token, ahead_token, atomic_choice])
                                cap, choices, curr_valid_cum = None, None, []

                                indx +=2
                                continue
                    elif ahead_token in self.quantifiers:
                        if curr_valid_cum:
                            if all(choice in self.atom_concepts_with_negation for choice in choices):
                                atomic_choice = random.choice(list(choices))
                            else:
                                atomic_choice = random.choice(list(self.atom_concepts_with_negation)) 

                            if isinstance(atomic_choice, tuple):
                                _token = [token] + list(atomic_choice)
                            else:
                                _token = [token, atomic_choice]    
                            
                            if next_ahead_token and next_ahead_token not in {')'} or next_ahead_token not in self.binary_ops:
                                _token += [')']
                            corrected_tokens.extend(_token)
                            cap, choices, curr_valid_cum = None, None, []

                            if next_ahead_token and next_ahead_token in self.unique_roles:
                                ops_choice = random.choice(list(self.binary_ops))
                                corrected_tokens.append(ops_choice)
                                indx +=1
                                continue

                            indx +=2
                            continue
                        else:
                            if not next_ahead_token:
                                atomic_choice = random.choice(list(self.unique_atom_concept_names))
                                corrected_tokens.append(token)
                                self.tokens[indx+1] = atomic_choice
                                indx +=1
                                continue
                    elif prev_token:
                        if prev_token in self.unique_atom_concept_names:
                            ops_choice = random.choice(list(self.binary_ops))
                            corrected_tokens.extend([ops_choice, token])
                            indx +=1
                            continue

            if token == ')':
                if curr_valid_cum:
                    if cap == 3 and len(curr_valid_cum) == 2:
                        atomic_choice = random.choice(list(self.unique_atom_concept_names))
                        corrected_tokens.append(atomic_choice)
                        cap, choices, curr_valid_cum = None, None, []
                        indx +=1
                        continue
            
            if not curr_valid_cum and token in self.unique_atom_concept_names | self.unique_roles and ahead_token:
                if token in self.unique_atom_concept_names:
                    ops_choice = random.choice(list(self.binary_ops))

                    if ahead_token in self.unique_atom_concept_names and prev_token not in {')'}:
                        corrected_tokens.extend([token, ops_choice])
                        indx +=1
                        continue
                    elif ahead_token in self.unique_roles:
                        quant_choice = random.choice(list(self.quantifiers))
                        corrected_tokens.extend([token, ops_choice, quant_choice])
                        indx +=1
                        continue
                else:
                    _token = [token, '.']
                    if ahead_token and ahead_token in self.unique_atom_concept_names:
                        corrected_tokens.extend(_token + [ahead_token])
                        indx +=2
                        continue
                    elif ahead_token and ahead_token in self.binary_ops:
                        atomic_choice = random.choice(list(self.atom_concepts_with_negation))

                        if isinstance(atomic_choice, tuple):
                            choice = list(atomic_choice)
                        else:
                            choice = [atomic_choice]

                        corrected_tokens.extend(_token + choice)
                        indx +=1
                        continue

            if token not in self.binary_ops | self.parenthesis | self.quantifiers:
                _token = token

                if _token in self.negation | self.unique_atom_concept_names:
                    if _token in self.negation:
                        if not cap:
                            cap = cap_curr_val['neg_atom']
                        choices = self.unique_atom_concept_names
                    else:
                        if not cap:
                            cap = cap_curr_val['atom']
                        choices = self.binary_ops
                elif _token in self.digits | self.dot | self.unique_roles:
                    if _token in self.digits: #TODO: Work on this later
                        if not cap:
                            cap = cap_curr_val['roleCard']
                        if curr_valid_cum and (ahead_token and ahead_token in self.digits):
                            _token += ahead_token
                            indx += 1
                        choices = self.unique_roles
                    elif _token in self.unique_roles:
                        if not cap:
                            cap = cap_curr_val['role']
                        choices = self.dot
                    elif _token in self.dot:
                        choices = self.atom_concepts_with_negation
                curr_valid_cum.append(_token)

            if not self._is_valid_next_token(token, corrected_tokens):
                token = random.choice(list(choices))
                curr_valid_cum.append(token)

            if isinstance(token, tuple):
                corrected_tokens.extend(token)
            else:
                corrected_tokens.append(token)

            if len(curr_valid_cum) == cap or (curr_valid_cum and curr_valid_cum[0] == '.' and len(curr_valid_cum) == 2):
                cap, choices, curr_valid_cum = None, None, []

            indx +=1
        return corrected_tokens
    
def generate_class_expression(kb_path:str, leaners_prediction = None, save_as_json: bool=False, relax_parentheses=False):
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
        # token_sequence = ['≥', '27', 'hasChild', '.', '(','Person', '⊔', 'Father',')', '⊔', 'Grandfather', '⊓', 'Female', '⊓', '∃', 'married', '.', 'Brother']

        # token_sequence = ['≥', '27', 'hasChild', '.', '(','Person', '⊔', 'Father',')', '⊔', '∃', 'hasSibling', '.', 'Brother', '⊓', ' Father ', '⊓', '(','∀', ' hasSibling', '.', 'Grandfather', ')']
        # token_sequence = ['Female', ' ', '⊓', ' ', '(', 'Female', ' ', '⊔', ' ', '(', '∃', ' ', 'married', '.', 'PersonWithASibling', ')', ')']
        # token_sequence = ['Female', ' ', '⊔', ' ', '(', '∃', ' ', 'married', '.', '(', ')',' ', '⊓', ' ', '(', '∃', ' ', 'hasSibling', '.', '⊤', ')']
        # token_sequence = ['Female', ' ', '⊔', ' ', '(', '∃', ' ', 'married', '.', '(', '¬','Grandfather', ')', ')']
        # token_sequence = ['Daughter', ' ', '⊔', ' ', '(', '∃', ' ', ' ', '.', 'Grandfather',')']
        # token_sequence = ['Granddaughter', ' ', '⊔', ' ', ' ', 'PersonWithASibling', ' ', ' ', '⊔', ' ', '(', '∃', ' ', 'hasParent', '.','PersonWithASibling', ')']
        # token_sequence = ['Person', '⊓', '(', 'Grandmother', '⊔', '(', '∃', 'married', '.', 'Grandfather', ')', ')', ')', ')', ')']
        # token_sequence = ['Grandparent', '⊔', '(', '∃', 'married', '.', '(', ')', ')']
        # token_sequence = ['(', '⊓', '(', '∀', 'married', '.', '(', '(', ')', ')', '(', '¬', ')', '(', ')', ')', ')']
        token_sequence = ['Person', '⊓', '(', '(', '⊔', '(', '∀', 'hasChild', '.', 'Grandfather', ')', ')', '⊓', '(', ')', ')', ')', ')']
        #['∀', 'married', '.', '(', 'Brother', '⊔', 'Sister']

    try:
        builder = ConceptAbstractSyntaxTreeBuilder(knowledge_base_path=kb_path)
        concept, result = builder.parse(token_sequence=token_sequence, relax_parentheses=relax_parentheses, enforce_validity=True, replace_with_negation=False)

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
    generate_class_expression(knowledge_base_path)
    

    '''
    TODO
        - Knowledge base fix with Alkid
        - decoding strategy - greed*, beam [constrained beam search]

        - why the choice of the number of predictions* and check its same at this dimension
        - Get unique predicitons from all learners excluding PAD | nces 12 roces [By today] i.e working with argmax
        - token masking, parser_prefixing
        
        - heirachy-awareness, best_hypothesis paradigm
        - ask jean about self.synthensizer beyond NCESTrainer
        - DD
            - where or how can I get the label for each given lps of a given dataset

        - [sequence of data] how does the adopted archs knows which e+ vs e- belongs to T since (e+, e-, T), how can naively 1, 0 markers any other ways?
            - how do you compute metrics within pred vs actual sequences vs incase of Aunt! can your reward be based on conformity with valid generated
        - unknown T via preprocessing



        "" https://huggingface.co/blog/vivien/llm-decoding-with-regex-constraints
        At decoding time, the DFA is used to determine, for each new token, which potential tokens are allowed. For this and starting from the initial 
        state of the DFA, we determine the allowed tokens with the outgoing transitions from the current state, apply the corresponding mask to the 
        next token probabilities and renormalize these probabilities. We can then sample a new token and update the state of the DFA.

        - https://arxiv.org/pdf/2405.00218   - constrained vs unconstrained beam sampling
        There are mainly two strategies for autoregressive decoding: maximization-based decoding and stochastic decoding.
        ""
    '''

