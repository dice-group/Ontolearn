import collections
import random

# Re-define your vocabulary for clarity and direct use
ATOMIC_CONCEPTS = {'Oxygen', "Person", "Animal", "Daughter", "Sister", "Thing", "Female", "Father", "Brother", "Granddaughter", "Son", 'Mother', 'Grandson', 'Child'}
ROLES = {"hasChild", "hasParent", "hasSibling", "married", "inBond"}
BINARY_OPS = {"⊓", "⊔"}
UNARY_OPS = {"¬"}
QUANTIFIERS = {"∃", "∀"}
PARENTHESES = {"(", ")"}
DOT = {'.'}
CARDINALITY_OPS = {"≥", "≤"}
DIGITS = {str(i) for i in range(10)}

VOCAB = ATOMIC_CONCEPTS | ROLES | BINARY_OPS | UNARY_OPS | QUANTIFIERS | PARENTHESES | DOT | CARDINALITY_OPS | DIGITS
NEG_ATOMIC_CONCEPTS = ATOMIC_CONCEPTS | {("¬", atom) for atom in ATOMIC_CONCEPTS}


class Parser:
    def __init__(self, tokens, max_length):
        self.tokens = tokens
        self.max_length = max_length
        self.current_index = 0
        self.corrected_tokens = []
        self.errors = []  # To log detected errors and corrections

    def _add_token(self, token):
        if len(self.corrected_tokens) < self.max_length:
            if isinstance(token, tuple):
                self.corrected_tokens.extend(token)
            else:
                self.corrected_tokens.append(token)
            return True
        return False

    def _peek(self, offset=0):
        if self.current_index + offset < len(self.tokens):
            return self.tokens[self.current_index + offset]
        return None

    def _consume(self):
        if self.current_index < len(self.tokens):
            token = self.tokens[self.current_index]
            self.current_index += 1
            return token
        return None

    def _expect(self, expected_types, error_msg, allow_skip=False):
        """
        Attempts to consume a token of an expected type.
        If current token is not expected, attempts error recovery.
        Returns the consumed token or a corrected one.
        """
        token = self._peek()

        # If current token is exactly one of the expected types
        if token in expected_types:
            self._consume()
            if not self._add_token(token):
                return None  # Max length check
            return token

        # If current token is a tuple (e.g., ('¬', 'Person')) and matches
        if isinstance(token, tuple) and token in expected_types:
            self._consume()
            if not self._add_token(token):
                return None  # Max length check
            return token

        # --- Error Recovery ---
        self.errors.append(
            f"Error: Expected one of {expected_types}, found '{token}' at index {self.current_index}. {error_msg}")

        # Strategy 1: Insert missing token
        if not token and expected_types:  # End of input, but more expected
            choice = random.choice(list(expected_types))
            self.errors.append(f"Correction: Inserting '{choice}'.")
            if not self._add_token(choice):
                return None
            return choice

        # Strategy 2: Replace incorrect token with a random valid choice
        # This is a strong heuristic; consider refining.
        if expected_types:
            # Try to replace if the current token is in VOCAB but not allowed
            if token in VOCAB:
                choice = random.choice(list(expected_types))
                self.errors.append(f"Correction: Replacing '{token}' with '{choice}'.")
                self._consume()  # Consume the bad token
                if not self._add_token(choice):
                    return None
                return choice

        # Strategy 3: Skip unexpected token (if allowed or if it's junk)
        if allow_skip and token is not None:
            self.errors.append(f"Correction: Skipping unexpected token '{token}'.")
            self._consume()  # Skip the bad token
            # After skipping, try to find an expected token by recursive call
            return self._expect(expected_types, error_msg,
                               allow_skip=True)  # Retry after skip

        # If no recovery strategy works, return None or raise an exception
        # For this problem, we'll try to insert a default to keep parsing
        if expected_types:
            choice = random.choice(list(expected_types))
            self.errors.append(
                f"Correction: Fallback inserting '{choice}' as no other recovery worked.")
            if not self._add_token(choice):
                return None
            return choice

        return None  # Critical error, cannot recover

    def _parse_concept(self):
        """
        Concept ::= AtomicConcept
                  | UnaryOp Concept
                  | ParenthesizedConcept
                  | Quantifier Role '.' Concept
                  | CardinalityOp Digit+ Role '.' Concept
                  | Concept BinaryOp Concept
        """
        lhs_concept = self._parse_primary_concept()
        if not lhs_concept:
            return None

        # Handle binary operations (left-associative)
        while len(self.corrected_tokens) < self.max_length:
            peek_token = self._peek()
            if peek_token in BINARY_OPS:
                self._consume()  # Consume the binary operator
                if not self._add_token(peek_token):
                    return None
                rhs_concept = self._parse_concept()  # Corrected: Use _parse_concept() for recursion
                if not rhs_concept:
                    # If RHS is missing, try to insert a default atomic concept
                    self.errors.append(
                        "Error: Missing concept after binary operator. Inserting default.")
                    if not self._add_token(random.choice(list(ATOMIC_CONCEPTS))):
                        return None
                    break  # Break from loop if insertion fails or to avoid infinite loop
            else:
                break  # No more binary operators, end of this concept

        return True  # Return True for success

    def _parse_primary_concept(self):
        """
        PrimaryConcept ::= AtomicConcept
                         | UnaryOp PrimaryConcept
                         | ParenthesizedConcept
                         | Quantifier Role '.' Concept
                         | CardinalityOp Digit+ Role '.' Concept
        """
        token = self._peek()

        if token in ATOMIC_CONCEPTS or (isinstance(token, tuple) and token[0] == '¬' and token[1] in ATOMIC_CONCEPTS):
            self._consume()
            if not self._add_token(token):
                return None  # Successfully parsed an atomic concept
            return True

        elif token in UNARY_OPS:
            self._consume()
            if not self._add_token(token):
                return None
            if not self._parse_primary_concept():  # Corrected: Parse a primary concept
                self.errors.append("Error: Missing concept after unary operator.")
                if not self._add_token(random.choice(list(ATOMIC_CONCEPTS))):
                    return None
            return True

        elif token == '(':
            self._consume()
            if not self._add_token('('):
                return None
            if not self._parse_concept():  # Corrected: Use _parse_concept() here
                self.errors.append("Error: Missing concept inside parentheses.")
                if not self._add_token(random.choice(list(ATOMIC_CONCEPTS))):
                    return None

            # Ensure closing parenthesis
            if self._peek() != ')':
                self.errors.append(
                    f"Correction: Missing ')' for opening '(' at {self.corrected_tokens.count('(') - self.corrected_tokens.count(')')} imbalance. Inserting ')'.")
                if not self._add_token(')'):
                    return None
            else:
                self._consume()
                if not self._add_token(')'):
                    return None
            return True  # Indicate success for parsing this primary concept

        elif token in QUANTIFIERS:
            return self._parse_quantifier_restriction()

        elif token in CARDINALITY_OPS:
            return self._parse_cardinality_restriction()

        # --- Error Recovery for primary concept start ---
        self.errors.append(
            f"Error: Unexpected token '{token}' when expecting primary concept start. Attempting recovery.")

        # Try to insert a default atomic concept if nothing else fits
        if not self._add_token(random.choice(list(ATOMIC_CONCEPTS))):
            return None
        return True  # Indicate that a concept was "parsed" (inserted)

    def _parse_role(self):
        role = self._expect(ROLES, "Expected a role (e.g., 'hasChild').")
        return role is not None  # Return boolean for success

    def _parse_digits(self):
        digits_str = ""
        while True:
            token = self._peek()
            if token in DIGITS:
                digits_str += self._consume()
            else:
                break

        if not digits_str:
            self.errors.append("Error: Missing digits for cardinality. Inserting '1'.")
            if not self._add_token('1'):
                return None
            return True

        # Add the collected digits as a single token, or individual digits based on preference
        # For simplicity, we'll add them as individual tokens here, but could join to a string
        for digit in digits_str:
            if not self._add_token(digit):
                return None
        return True

    def _parse_quantifier_restriction(self):
        quantifier = self._expect(QUANTIFIERS, "Expected a quantifier ('∃' or '∀').")
        if not quantifier:
            return None

        role = self._parse_role()
        if not role:
            self.errors.append("Error: Missing role after quantifier. Inserting default.")
            if not self._add_token(random.choice(list(ROLES))):
                return None  # Insert a role

        dot = self._expect(DOT, "Expected '.' after role in quantifier restriction.")
        if not dot:
            self.errors.append("Error: Missing '.' after role. Inserting '.'.")
            if not self._add_token('.'):
                return None  # Insert dot

        concept = self._parse_concept()  # Corrected: Use _parse_concept()
        if not concept:
            self.errors.append(
                "Error: Missing concept after '.' in quantifier restriction. Inserting default.")
            if not self._add_token(random.choice(list(ATOMIC_CONCEPTS))):
                return None
        return True

    def _parse_cardinality_restriction(self):
        card_op = self._expect(CARDINALITY_OPS, "Expected a cardinality operator ('≥' or '≤').")
        if not card_op:
            return None

        digits = self._parse_digits()
        if not digits:
            return None  # _parse_digits handles its own recovery

        role = self._parse_role()
        if not role:
            self.errors.append("Error: Missing role after cardinality digits. Inserting default.")
            if not self._add_token(random.choice(list(ROLES))):
                return None

        dot = self._expect(DOT, "Expected '.' after role in cardinality restriction.")
        if not dot:
            self.errors.append("Error: Missing '.' after role. Inserting '.'.")
            if not self._add_token('.'):
                return None

        concept = self._parse_concept() # Corrected: Use _parse_concept()
        if not concept:
            self.errors.append(
                "Error: Missing concept after '.' in cardinality restriction. Inserting default.")
            if not self._add_token(random.choice(list(ATOMIC_CONCEPTS))):
                return None
        return True

    def parse(self):
        """
        Main entry point for parsing and correction.
        """
        # Initial check for starting tokens
        initial_token = self._peek()
        if not initial_token or (initial_token not in UNARY_OPS and initial_token not in QUANTIFIERS and
                                    initial_token not in CARDINALITY_OPS and initial_token not in ATOMIC_CONCEPTS and
                                    initial_token != '(' and
                                    not (isinstance(initial_token, tuple) and initial_token[0] == '¬')):
            self.errors.append(
                f"Error: Invalid start token '{initial_token}'. Inserting default 'Person'.")
            if not self._add_token('Person'):  # Add a sensible default to start
                return []

        self._parse_concept()  # Start parsing the main concept

        # Final cleanup: ensure parentheses are balanced
        open_parens = self.corrected_tokens.count('(')
        close_parens = self.corrected_tokens.count(')')

        while open_parens > close_parens and len(self.corrected_tokens) < self.max_length:
            self.corrected_tokens.append(')')
            self.errors.append("Correction: Appending missing ')' to balance parentheses.")
            close_parens += 1

        while close_parens > open_parens and len(self.corrected_tokens) < self.max_length:
            self.corrected_tokens.insert(0, '(')  # Insert at the beginning
            self.errors.append("Correction: Prepended missing '(' to balance parentheses.")
            open_parens += 1

        # Truncate to max_length if corrections exceeded it
        return self.corrected_tokens[:self.max_length]


def validate_and_correct_sequence_robust(tokens, max_length):
    parser = Parser(tokens, max_length)
    corrected_sequence = parser.parse()
    # print("\n--- Errors/Corrections Log ---")
    # for error in parser.errors:
    #     print(error)
    return corrected_sequence


# --- Test Cases ---
test_cases = [
    ['(', 'Granddaughter', '⊓', '(', 'Grandson', '⊔', '(', '¬', ')', '.', 'Daughter', ')', ')'],
    ['Person', '⊓', '(', '∃', '⊔', '.', '(', 'Granddaughter', '⊔', '(', '∃', 'hasChild', '(', '¬', ')', ')', ')', ')', ')'],
    ['Person', '⊔', '(', '∃', 'hasSibling', '.', '(', '∃', '⊓', '(', ')', ')'],
    ['¬', 'Brother', '¬', 'Animal', '⊓', '(', '(', 'Granddaughter', ')', 'Mother', 'Thing', ')'],
    ['Person', '⊓', '(', '∀', 'hasSibling', '.', '(', 'Sister', '⊓', '(', ')', ')', ')', ')'],
    ['(', 'Person', '⊓', '(', '∀', '.', '(', '(', ')', '⊔', ')', ')'],
    ['Person', '⊓', '(', 'Grandson', '⊔', '(', '∃', 'married', '.', 'Mother', ')', '⊓', ')', ')', ')'],
    ['Person', '⊔', '(', '∃', 'married', '.', '(', '∀', '⊓', '(', '¬', ')', ')', ')'],
    ['(', '⊓', '(', 'Grandson', '⊔', '(', '¬', ')', ')', '.', ')', ')', ')', ')', '(', '.', ')'],
    ['(', '⊓', '(', '∀', 'hasChild', '.', '(', 'Child', ')', '(', ')', '⊔', '(', '∀', ')', '.', ')', ')'],
    ['Person', '⊓', '(', '∀', '⊔', '(', '∀', ')', '(', '¬', ')', ')', ')'],
    ['¬', 'Person', '⊓', '(', 'Animal', '⊔', '(', '∀', 'hasChild', '.', '(', ')', ')', ')'],
    ['Person', '⊔', '(', '∀', 'hasChild', '.', '(', ')'],
    ['(', '⊔', '(', '∃', '.', '(', 'Child', ')', ')'],
    ['Person', '⊓', '(', '∃', 'married', '.', '(', 'Granddaughter', '⊔', '(', '∀', 'hasChild', '.', '(', ')', ')', ')', ')'],
    ['(', '⊓', '(', '(', '(', '∀', ')', '.', ')', ')'],
    ['∃', '⊔', '(', '∀', 'hasChild', '.', '(', 'Brother', ')'],
]

print("--- Testing Robust Parser ---")
for i, tokens in enumerate(test_cases):
    print(f"\nOriginal {i + 1}: {tokens}")
    corrected = validate_and_correct_sequence_robust(tokens, max_length=50)  # Set a reasonable max_length
    print(f"Corrected {i + 1}: {corrected}")
    # print("-" * 30)

# Add a few more specific failure cases
print("\n--- Additional Test Cases for Robustness ---")
additional_test_cases = [
    ['Person', '⊓', '(', 'Grandson', '⊔', '(', '∃', 'married', '.', 'Mother', ')', '⊓', ')', ')', ')', '⊓'], 
    ['Person', '⊓', '(', 'Sister', '⊔', '(', '∀', 'hasChild', '.', 'Son', ')', ')', '⊓', '(', '∀', ')'],
    ['(', '⊓', '⊓', '(', 'Person', '⊔', '(', '∃', 'hasSibling', '.', 'Granddaughter', ')', ')', '⊓', '(', '∀', 'hasChild', '.', '(', '¬', ')'],
    ['(', '⊓', '⊓', '(', '(', 'Granddaughter', ')', '(', ')', '.', '(', ')', '(', ')'], 
    ['(', '(', '(', 'Person', ')', ')', '⊓', 'Oxygen', '⊔'],  # Missing RHS after last op
    ['∃', '.', 'Person'],  # Missing role
    ['≥', '1', '.', 'Person'],  # Missing role
    ['Animal', 'hasChild', 'Thing'],  # Missing .
    ['(', '¬', '⊓', 'Person', ')'],  # Unary op then binary op
    ['Brother', 'Brother'],  # Two atomic concepts without operator
    ['(', '∃', 'hasChild', 'Person', ')'],  # Missing dot
    ['(', 'Person', '⊓', '('],  # Unclosed parentheses at end
    [')', 'Person'], # Unexpected ) at start
    ['Person', ')'] # Unexpected ) after concept
]

for i, tokens in enumerate(additional_test_cases):
    print(f"\nOriginal {i + 1 + len(test_cases)}: {tokens}")
    corrected = validate_and_correct_sequence_robust(tokens, max_length=50)
    print(f"Corrected {i + 1 + len(test_cases)}: {corrected}")
