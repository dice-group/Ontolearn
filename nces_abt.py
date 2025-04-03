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

# class IncompleteBinary(Expr):
#     def __init__(self, left: Expr, op: str):
#         self.left = left
#         self.op = op
#     def to_string(self):
#         return f"{self.left.to_string()} {self.op}"
#     def __repr__(self):
#         return f"({self.left} {self.op})"
#     def to_dict(self):
#         return {"type": "IncompleteBinary", "left": self.left.to_dict(), "op": self.op}

# --- Allowed Vocabulary ---
ATOMIC_CONCEPTS = {"Person", "Animal", "Thing", "Female", "Father", "Brother"}
ROLES = {"hasChild", "hasParent", "hasSibling", "married"}
BINARY_OPS = {"⊓", "⊔"}
UNARY_OPS = {"¬"}
QUANTIFIERS = {"∃", "∀"}
PARENTHESES = {"(", ")"}
DOT = {'.'}
VOCAB = ATOMIC_CONCEPTS | ROLES | BINARY_OPS | UNARY_OPS | QUANTIFIERS | PARENTHESES | DOT

def allowed_tokens(context_tokens):
    if not context_tokens:
        return UNARY_OPS | QUANTIFIERS | ATOMIC_CONCEPTS | {"("}
    last = context_tokens[-1]
    if last in QUANTIFIERS:
        return ROLES
    if last in ROLES:
        return DOT
    if last == '.':
        return UNARY_OPS | QUANTIFIERS | ATOMIC_CONCEPTS | {"("}
    if last == '(':
        return UNARY_OPS | QUANTIFIERS | ATOMIC_CONCEPTS | {"("}
    if last in ATOMIC_CONCEPTS:
        return BINARY_OPS | {')'}
    if last in BINARY_OPS:
        return UNARY_OPS | QUANTIFIERS | ATOMIC_CONCEPTS | {"("}
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

def grammar_constrained_decoder(max_length=12):
    generated_tokens = []
    for _ in range(max_length):
        allowed = allowed_tokens(generated_tokens)
        logits = {token: np.random.rand() for token in VOCAB}
        for token in VOCAB:
            if token not in allowed:
                logits[token] = -np.inf
        next_token = max(logits, key=logits.get)
        generated_tokens.append(next_token)
        if generated_tokens[-1] == ')' and generated_tokens.count('(') <= generated_tokens.count(')'):
            break
    return generated_tokens

def generate_class_expression(relax_parentheses=True, save_as_json=False):
    token_sequence = grammar_constrained_decoder(max_length=50)
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
    generate_class_expression(save_as_json=False)

