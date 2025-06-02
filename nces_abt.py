from abc import ABC, abstractmethod
import numpy as np
import json

# class ParseError(Exception):
#     def __init__(self, message, partial_ast, failed_token, pos, tokens):
#         self.message = message
#         self.partial_ast = partial_ast
#         self.failed_token = failed_token
#         self.pos = pos
#         self.tokens = tokens
#     def __str__(self):
#         return (f"{self.message} (failed token: '{self.failed_token}' at position {self.pos}).\n"
#                 f"Partial AST: {self.partial_ast}")

class Expr(ABC):
    @abstractmethod
    def to_string(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

class Atomic(Expr):
    def __init__(self, name):
        self.name = name
    def to_string(self):
        return self.name
    def __repr__(self):
        return self.name
    def to_dict(self):
        return {"type": "Atomic", "name": self.name}

class Not(Expr):
    def __init__(self, expr: Expr):
        self.expr = expr
    def to_string(self):
        return f"¬{self.expr.to_string()}"
    def __repr__(self):
        return f"¬{self.expr}"
    def to_dict(self):
        return {"type": "Not", "expr": self.expr.to_dict()}

class And(Expr):
    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right
    def to_string(self):
        return f"({self.left.to_string()} ⊓ {self.right.to_string()})"
    def __repr__(self):
        return f"({self.left} ⊓ {self.right})"
    def to_dict(self):
        return {"type": "And", "left": self.left.to_dict(), "right": self.right.to_dict()}

class Or(Expr):
    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right
    def to_string(self):
        return f"({self.left.to_string()} ⊔ {self.right.to_string()})"
    def __repr__(self):
        return f"({self.left} ⊔ {self.right})"
    def to_dict(self):
        return {"type": "Or", "left": self.left.to_dict(), "right": self.right.to_dict()}

class Exists(Expr):
    def __init__(self, role: str, filler: Expr):
        self.role = role
        self.filler = filler
    def to_string(self):
        return f"∃{self.role}.{self.filler.to_string()}"
    def __repr__(self):
        return f"(∃{self.role}.{self.filler})"
    def to_dict(self):
        return {"type": "Exists", "role": self.role, "filler": self.filler.to_dict()}

class Forall(Expr):
    def __init__(self, role: str, filler: Expr):
        self.role = role
        self.filler = filler
    def to_string(self):
        return f"∀{self.role}.{self.filler.to_string()}"
    def __repr__(self):
        return f"(∀{self.role}.{self.filler})"
    def to_dict(self):
        return {"type": "Forall", "role": self.role, "filler": self.filler.to_dict()}

class Cardinality(Expr):
    def __init__(self, kind: str, n: int, role: str, filler: Expr):
        self.kind = kind  # either "≥" or "≤" or 
        self.n = n
        self.role = role
        self.filler = filler
    def to_string(self):
        return f"{self.kind}{self.n} {self.role}.{self.filler.to_string()}"
    def __repr__(self):
        return f"({self.kind}{self.n} {self.role}.{self.filler})"
    def to_dict(self):
        return {
            "type": "Cardinality",
            "kind": self.kind,
            "n": self.n,
            "role": self.role,
            "filler": self.filler.to_dict()
        }


# --- Allowed Vocabulary ---

ATOMIC_CONCEPTS = frozenset({'Oxygen',"PersonWithASibling","Person", "Animal","Daughter", "Sister", "Thing", "Female", "Father", "Brother", "Parent", "Granddaughter", "Son", 'Mother', 'Grandson', 'Child', 'Grandchild', 'Grandmother'})
ROLES = frozenset({"hasChild", "hasParent", "hasSibling", "married", "inBond"})
BINARY_OPS = frozenset({"⊓", "⊔"})
UNARY_OPS = frozenset({"¬"})
QUANTIFIERS = frozenset({"∃", "∀"})
PARENTHESES = frozenset({"(", ")"})
DOT = frozenset({'.'})
CARDINALITY_OPS = frozenset({"≥", "≤"})
DIGITS = frozenset({str(i) for i in range(10)})
VOCAB = ATOMIC_CONCEPTS | ROLES | BINARY_OPS | UNARY_OPS | QUANTIFIERS | PARENTHESES | DOT | CARDINALITY_OPS | DIGITS

def lookahead_grammar_strategy(context_tokens):
    if not context_tokens:
        return UNARY_OPS | QUANTIFIERS | CARDINALITY_OPS | ATOMIC_CONCEPTS | {'('}
    
    last = context_tokens[-1]
    
    if last in QUANTIFIERS:
        return ROLES

    if last in CARDINALITY_OPS:
        return DIGITS

    if last in DIGITS:
        return DIGITS | ROLES

    if last in ROLES:
        return DOT

    if last == '.':
        return UNARY_OPS | QUANTIFIERS | CARDINALITY_OPS | ATOMIC_CONCEPTS | {'('}

    if last == '(':
        return UNARY_OPS | QUANTIFIERS | CARDINALITY_OPS | ATOMIC_CONCEPTS | {'('}

    if last in ATOMIC_CONCEPTS:
        return BINARY_OPS | {')'}

    if last in BINARY_OPS:
        return UNARY_OPS | QUANTIFIERS | CARDINALITY_OPS | ATOMIC_CONCEPTS | {'('}
    
    if last in UNARY_OPS:
        return ATOMIC_CONCEPTS | QUANTIFIERS | CARDINALITY_OPS | {'('}

    if last == ')':
        return BINARY_OPS | {')'}

    return VOCAB



def format_token_sequence(tokens):
    result = ""
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in QUANTIFIERS:
            result += token
            i += 1
            if i < len(tokens):
                result += tokens[i]
                i += 1
                if i < len(tokens) and tokens[i] in DOT:
                    result += tokens[i]
                    i += 1
            continue

        if token in UNARY_OPS:
            result += token
        elif token in BINARY_OPS:
            result += " " + token + " "
        elif token in PARENTHESES:
            result += token
        else:
            if result and result[-1] not in " (":
                result += "" + token
            else:
                result += token
        i += 1
    return " ".join(result.split())

class ConceptAbstractSyntaxTreeBuilder:
    def __init__(self, tokens, relax_parentheses=False):
        self.tokens = self._sanitize_tokens([token for token in tokens if token.strip() not in ('(', ')')] if relax_parentheses else tokens)
        self.index = 0
        self.length = len(self.tokens)

    def _current_token(self):
        return self.tokens[self.index] if self.index < self.length else None

    def _advance(self):
        self.index += 1

    def _sanitize_tokens(self, tokens):
        return [token.strip() for token in tokens if token.strip()]

    def parse(self):
        ast = self._parse_expression()
        
        if self.index != self.length:
            raise Exception("Extra tokens remain after generation.")
        return ast

    def _parse_expression(self):
        node = self._parse_term()
        
        while self._current_token() in BINARY_OPS:
            op = self._current_token()
            self._advance()
            right = self._parse_term()
            node = And(node, right) if op == '⊓' else Or(node, right)
        return node

    def _parse_term(self):
        token = self._current_token()
        if token is None:
            raise Exception("Unexpected end of tokens during generation.")

        if token in UNARY_OPS:
            self._advance()
            return Not(self._parse_term())

        if token in QUANTIFIERS:
            quant = token
            self._advance()
            if self._current_token() not in ROLES:
                raise Exception(f"Expected role after quantifier, got '{self._current_token()}'.")
            
            role = self._current_token()
            self._advance()
            if self._current_token() != '.':
                raise Exception("Expected '.' after role in quantified expression.")
            
            self._advance()
            filler = self._parse_term()
            return Exists(role, filler) if quant == '∃' else Forall(role, filler)
        
        if token in CARDINALITY_OPS:
            kind = token
            self._advance()

            num_token = self._current_token()
            if num_token is None or not num_token.isdigit():
                raise Exception(f"Expected number after '{kind}', got '{num_token}'.")
            number = int(num_token)
            self._advance()

            role = self._current_token()
            if role not in ROLES:
                raise Exception(f"Expected role after number in cardinality, got '{role}'.")
            self._advance()

            if self._current_token() != '.':
                raise Exception("Expected '.' after role in cardinality.")
            self._advance()

            filler = self._parse_term()
            return Cardinality(kind, number, role, filler)


        if token == '(':
            self._advance()
            expr = self._parse_expression()

            if self._current_token() != ')':
                raise Exception("Expected ')' after expression.")
            self._advance()
            return expr

        if token in ATOMIC_CONCEPTS:
            self._advance()
            return Atomic(token)

        raise Exception(f"Unexpected token '{token}' at position {self.index}.")

# def grammar_constrained_decoder(max_length=12):
#     generated_tokens = []
#     for _ in range(max_length):
#         allowed = lookahead_grammar_strategy(generated_tokens)
#         logits = {token: np.random.rand() for token in VOCAB}
#         for token in VOCAB:
#             if token not in allowed:
#                 logits[token] = -np.inf
#         next_token = max(logits, key=logits.get)
#         generated_tokens.append(next_token)
#         if generated_tokens[-1] == ')' and generated_tokens.count('(') <= generated_tokens.count(')'):
#             break
#     return generated_tokens
def grammar_constrained_decoder(max_length=12):
    generated_tokens = []
    number_buffer = ""
    t = 0

    while t < max_length:
        context = generated_tokens + ([number_buffer] if number_buffer else [])
        allowed = lookahead_grammar_strategy(context)
        print

        # Simulate logits from neural model or uniform sampling
        logits = {token: np.random.rand() for token in VOCAB}
        for token in VOCAB:
            if token not in allowed:
                logits[token] = -np.inf

        # Select the next token
        next_token = max(logits, key=logits.get)

        # If it's a digit, accumulate it into number_buffer
        if next_token.isdigit() and len(number_buffer) <= 3:
            number_buffer += next_token
            continue

        # If number_buffer is non-empty and current token is non-digit, flush it as a full number
        if number_buffer:
            generated_tokens.append(number_buffer)
            t += 1
            number_buffer = ""

            # If we hit max after flushing number, exit early
            if t >= max_length:
                break

        # Add the current non-digit token
        generated_tokens.append(next_token)
        t += 1

        # Stop if expression seems complete
        if next_token == ')' and generated_tokens.count('(') <= generated_tokens.count(')'):
            break

    # Flush any remaining number at the end
    if number_buffer and t < max_length:
        generated_tokens.append(number_buffer)

    return generated_tokens

def generate_class_expression(relax_parentheses=True, save_as_json=False):
    # Generated Token Sequence: ['Female', '⊔', '≥', '2', '9', 'married', '.', 'Thing', '⊔', 'Female', ')']
    # Error constructing AST: Expected role after number in cardinality, got '9'.
    # Incomplete AST: Female ⊔ ≥29married.Thing ⊔ Female)
    # token_sequence = ['¬', 'Person', '⊓', '≤', '3001', 'hasSibling', '.', 'Female']
    # token_sequence = ['≤', '6', 'hasSibling', '.', 'Person', '⊓', '∃', 'hasChild', '.', 'Brother', ')']
    # token_sequence = ['≥', '27', 'hasChild', '.', 'Person', '⊔', 'Father', '⊔', 'Animal', '⊓', 'Female', ')', '⊓', '∃', 'married', '.', 'Brother']
    # token_sequence = ["≥", "1", "hasChild", ".", "(", "Person", "⊔", "Animal", ")"]
    # token_sequence =  ['Person', '⊓', '(', '∀', '⊔', '(', '¬', ')', ')', 'Grandparent', ')', ')', '(', ')']
    # ['≥', '55', 'hasParent', '.', '¬', 'Person', ')']

    # token_sequence = grammar_constrained_decoder(max_length=10)
    token_sequence = ['Person', '⊔', '(', '∃', 'married', '.', '(', 'Father', ')']
    print("Generated Token Sequence:", token_sequence)
    try:
        builder = ConceptAbstractSyntaxTreeBuilder(token_sequence, relax_parentheses=relax_parentheses)
        ast = builder.parse()
        print("Constructed AST:", ast)
        print("Final Expression:", ast.to_string())
        if save_as_json:
            with open(f"concept_abstract_syntax_tree.json", "w") as f:
                json.dump(ast.to_dict(), f, indent=2)
            print(f"AST exported to concept_abstract_syntax_tree.json")
    except Exception as e:
        print("Error constructing AST:", e)
        if hasattr(e, 'partial_ast') and e.partial_ast is not None:
            print("Incomplete AST (as tokens):", e.partial_ast.to_string())
        else:
            print("Incomplete AST:", format_token_sequence(token_sequence))



if __name__ == "__main__":
    generate_class_expression()
    # generate_concept_with_logits()

