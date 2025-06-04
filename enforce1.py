import random
from nces_abt import ATOMIC_CONCEPTS, BINARY_OPS, DIGITS, DOT, PARENTHESES, QUANTIFIERS, ROLES, UNARY_OPS, lookahead_grammar_strategy
from random import choice

def atomic_concepts_with_negation(replace_with_negation=False):
    if replace_with_negation:
        return ATOMIC_CONCEPTS | {("¬", atom) for atom in ATOMIC_CONCEPTS}
    return ATOMIC_CONCEPTS
NEG_ATOMIC_CONCEPTS = atomic_concepts_with_negation(replace_with_negation=False) #ATOMIC_CONCEPTS | {("¬", atom) for atom in ATOMIC_CONCEPTS}
# NEG_ATOMIC_CONCEPTS = frozenset(set(ATOMIC_CONCEPTS) | {frozenset(("¬", atom)) for atom in ATOMIC_CONCEPTS})
# if isinstance(item, str):
# print("Plain item:", item)
# elif isinstance(item, frozenset):

def is_valid_next_token(token, context_tokens):
    allowed = sorted(lookahead_grammar_strategy(context_tokens))
    return token in allowed

def validate_and_correct_sequence(tokens, max_length):
    corrected_tokens = []
    curr_valid_syn_cum = []
    choices = None
    cap = None
    cap_curr_val = {'neg': 2, 'atom': 1, 'role': 3, 'roleCard': 4}

    indx = 0
    while indx < len(tokens) and len(corrected_tokens) < max_length:
        token = tokens[indx]
        prev_token = tokens[indx-1] if indx != 0 else None 
        ahead_token = tokens[indx+1] if (indx+1) < len(tokens) else None
        next_ahead_token = tokens[indx+2] if (indx+2) < len(tokens) else None

        # --- 1. Handling nonsensical or incomplete parts early ---
        if not curr_valid_syn_cum and token in QUANTIFIERS | UNARY_OPS | BINARY_OPS | PARENTHESES | ATOMIC_CONCEPTS | DOT | ROLES:
            if indx != 0 and prev_token:
                if token in UNARY_OPS:
                    if prev_token in UNARY_OPS | ATOMIC_CONCEPTS:
                        ops_choice = random.choice(list(BINARY_OPS))
                        corrected_tokens.extend([ops_choice, token])
                        indx +=1
                        continue
                elif token in DOT and prev_token not in ROLES:
                        if ahead_token and ahead_token not in {')'}:
                            if ahead_token in ATOMIC_CONCEPTS | UNARY_OPS | {'('}:
                                ops_choice = random.choice(list(BINARY_OPS))
                                corrected_tokens.append(ops_choice)
                                indx +=1
                                continue
                elif token in ROLES and prev_token not in QUANTIFIERS:
                    if ahead_token and prev_token in {')'}:
                        ops_choice = random.choice(list(BINARY_OPS))
                        corrected_tokens.append(ops_choice)

                        if ahead_token in DOT:
                            quant_choice = random.choice(list(QUANTIFIERS))
                            corrected_tokens.extend([quant_choice, token, ahead_token])
                            curr_valid_syn_cum.extend([token, ahead_token])
                            cap = 3
                            choices = NEG_ATOMIC_CONCEPTS
                            indx += 2
                            continue
                        elif ahead_token in ATOMIC_CONCEPTS:
                            indx += 1
                            continue
                        elif ahead_token in {')'}:
                            indx +=2
                            continue
                    elif ahead_token and ahead_token in UNARY_OPS | ATOMIC_CONCEPTS:
                        if prev_token in BINARY_OPS | PARENTHESES:
                            indx += 1
                            continue
                elif token in QUANTIFIERS and prev_token not in BINARY_OPS:
                    ops_choice = random.choice(list(BINARY_OPS))

                    if prev_token in ATOMIC_CONCEPTS and (ahead_token in ROLES or (next_ahead_token and next_ahead_token in ROLES)):
                        corrected_tokens.extend([ops_choice, token])
                        choices = ROLES
                        indx += 1
                        continue
                elif token in {')'} and prev_token not in ATOMIC_CONCEPTS:
                    if prev_token in QUANTIFIERS and ahead_token in ROLES:
                        tokens.pop(indx)
                        continue
                elif token in {'('} and prev_token in {')'} | ATOMIC_CONCEPTS:
                    ops_choice = random.choice(list(BINARY_OPS))
                    corrected_tokens.extend([ops_choice, token])
                    choices = NEG_ATOMIC_CONCEPTS
                    indx += 1
                    continue
                # elif token in ATOMIC_CONCEPTS: # Has been cater for in another scope
                #     if prev_token in QUANTIFIERS | {")"}:
                #         print('===========')
                #         ops_choice = random.choice(list(BINARY_OPS))
                #         corrected_tokens.extend([ops_choice, token])
                #         indx +=1
                #         continue


            if ahead_token:
                if token in QUANTIFIERS:
                    if ahead_token in DOT:
                        role_choice = random.choice(list(ROLES))
                        corrected_tokens.extend([token, role_choice])
                        curr_valid_syn_cum.append(role_choice)
                        cap = 3
                        indx += 1
                        continue
                    elif ahead_token in PARENTHESES:
                        if prev_token and prev_token in {')'}:
                            corrected_tokens.append(random.choice(list(BINARY_OPS)))

                        role_choice = random.choice(list(ROLES))
                        corrected_tokens.extend([token, role_choice])
                        curr_valid_syn_cum.append(role_choice)
                        cap = 3
                        choices = DOT
                        indx += 1
                        continue
                    elif ahead_token in BINARY_OPS:
                        atomic_choice = random.choice(list(NEG_ATOMIC_CONCEPTS))
                        if isinstance(atomic_choice, tuple):
                            corrected_tokens.extend(atomic_choice)
                        else:
                            corrected_tokens.append(atomic_choice)
                        indx += 1
                        continue
                    elif ahead_token in ATOMIC_CONCEPTS:
                        tokens.pop(indx)
                        continue

                if token in BINARY_OPS:
                    if ahead_token in BINARY_OPS:
                        atomic_choice = random.choice(list(NEG_ATOMIC_CONCEPTS))
                        if isinstance(atomic_choice, tuple):
                            _token = [token] + list(atomic_choice)
                        else:
                            _token = [token, atomic_choice]

                        corrected_tokens.extend(_token)
                        indx += 2
                        continue
                    elif ahead_token == ')':
                        atomic_choice = random.choice(list(NEG_ATOMIC_CONCEPTS))
                        if isinstance(atomic_choice, tuple):
                            _token = [token] + list(atomic_choice)
                        else:
                            _token = [token, atomic_choice]
                        corrected_tokens.extend(_token)
                        indx += 1
                        continue
                    elif ahead_token in DOT:
                        if next_ahead_token:

                            if next_ahead_token in PARENTHESES | ATOMIC_CONCEPTS:
                                corrected_tokens.append(token)
                                indx += 2
                                continue
                            # elif next_ahead_token in ATOMIC_CONCEPTS:
                if token == ')':
                    if indx != 0:
                        if prev_token and prev_token in ATOMIC_CONCEPTS:
                            corrected_tokens.append(token)
                            indx += 1
                            continue

                        elif ahead_token in PARENTHESES | QUANTIFIERS | DOT:
                            if ahead_token in {')'}:
                                indx += 1
                                continue

                            ops_choice = random.choice(list(BINARY_OPS))
                            corrected_tokens.extend([token, ops_choice])
                            indx += 1
                            continue
                    else:
                        if ahead_token in ATOMIC_CONCEPTS:
                            corrected_tokens.append('(')
                            indx +=1
                            continue
                        elif ahead_token in BINARY_OPS:
                            atomic_choice = random.choice(list(NEG_ATOMIC_CONCEPTS))
                            if isinstance(atomic_choice, tuple):
                                _token = list(atomic_choice)
                            else:
                                _token = [atomic_choice]
                            corrected_tokens.extend(['('] + _token)
                            indx += 1
                            continue

                if token in UNARY_OPS:
                    if ahead_token in QUANTIFIERS:
                        if next_ahead_token and next_ahead_token in ROLES:
                            atomic_choice = random.choice(list(ATOMIC_CONCEPTS))
                            ops_choice = random.choice(list(BINARY_OPS))
                            corrected_tokens.extend([token, atomic_choice, ops_choice])
                            indx += 1
                            continue
                        else:
                            corrected_tokens.append(token)
                            if ahead_token not in ATOMIC_CONCEPTS | UNARY_OPS:
                                tokens[indx + 1] = random.choice(list(ATOMIC_CONCEPTS))
                            indx += 1
                            continue
                    elif ahead_token in ROLES:
                        if next_ahead_token and next_ahead_token in {'.'}:
                            quant_choice = random.choice(list(QUANTIFIERS))
                            tokens[indx] = quant_choice
                            corrected_tokens.append(quant_choice)
                            indx +=1
                            continue
                        
                        corrected_tokens.append(token)
                        atomic_choice = random.choice(list(ATOMIC_CONCEPTS))
                        tokens[indx+1] = atomic_choice
                        indx +=1
                        continue
                        
                if token in ROLES and prev_token not in QUANTIFIERS:
                    if prev_token in ATOMIC_CONCEPTS and ahead_token in ATOMIC_CONCEPTS:
                        ops_choice = random.choice(list(BINARY_OPS))
                        corrected_tokens.append(ops_choice)
                        indx +=1
                        continue



                
        # # # --- 6. Special fixes after ')' must be binop or ')' ---
        # if token == ')':
        #     if (indx + 1) < len(tokens) and tokens[indx + 1] not in BINARY_OPS | {')'}:
        #         binop_choice = random.choice(list(BINARY_OPS))
        #         corrected_tokens.append(binop_choice)

        # # --- 5. Special fixes after digits: digits must be followed by role ---
        # if token in DIGITS:
        #     if (indx + 1) < len(tokens) and tokens[indx + 1] not in DIGITS | ROLES:
        #         role_choice = random.choice(list(ROLES))
        #         corrected_tokens.append(role_choice)

        # --- 2. Special case: After '(' and next illegal ---
        if token in QUANTIFIERS:
            if curr_valid_syn_cum:
                if cap == 3 and len(curr_valid_syn_cum) == 2:
                    atomic_choice = random.choice(list(choices))

                    if isinstance(atomic_choice, tuple):
                        corrected_tokens.extend(atomic_choice)
                        tokens[indx] = atomic_choice[1]
                    else:
                        corrected_tokens.append(atomic_choice)
                        tokens[indx] = atomic_choice
                    
                    cap, choices, curr_valid_syn_cum = None, None, []
                    indx +=1
                    continue

        if token == '(':
            if ahead_token or prev_token:
                if ahead_token in UNARY_OPS | BINARY_OPS | {')'} | DOT:
                    if not curr_valid_syn_cum:
                        if indx == 0:
                            if ahead_token not in UNARY_OPS | ATOMIC_CONCEPTS:
                                token = random.choice(list(NEG_ATOMIC_CONCEPTS))
                                if isinstance(token, tuple):
                                    corrected_tokens.extend(token)
                                else:
                                    corrected_tokens.append(token)
                                indx += 1
                                continue
                            else:
                                if next_ahead_token:

                                    if ahead_token in UNARY_OPS:
                                        if next_ahead_token not in ATOMIC_CONCEPTS:
                                            atomic_choice = random.choice(list(ATOMIC_CONCEPTS))

                                            if next_ahead_token in BINARY_OPS:
                                                corrected_tokens.extend([token, atomic_choice])
                                                ahead_token = atomic_choice
                                            elif next_ahead_token in QUANTIFIERS:
                                                ops_choice = random.choice(list(BINARY_OPS))
                                                corrected_tokens.extend([token, atomic_choice, ops_choice])
                                                ahead_token = ops_choice
                                            indx +=2
                                            continue
                        else:
                            if ahead_token in BINARY_OPS:
                                if next_ahead_token and next_ahead_token in UNARY_OPS | ATOMIC_CONCEPTS:
                                    if prev_token and prev_token in BINARY_OPS:
                                        indx +=2
                                        continue
                                elif next_ahead_token in PARENTHESES and prev_token and prev_token not in ATOMIC_CONCEPTS:
                                    atomic_choice = random.choice(list(NEG_ATOMIC_CONCEPTS))

                                    if isinstance(atomic_choice, tuple):
                                        _token = list(atomic_choice)
                                        tokens[indx:indx] = _token
                                        indx += 1
                                    else:
                                        _token = [atomic_choice]
                                        tokens[indx] = _token

                                    corrected_tokens.extend(_token)
                                indx +=1
                                continue

                            elif ahead_token in {')'} | DOT :
                                atomic_choice = random.choice(list(NEG_ATOMIC_CONCEPTS))
                                ops_choice = random.choice(list(BINARY_OPS))

                                if isinstance(atomic_choice, tuple):
                                    _token = list(atomic_choice)
                                else:
                                    _token = [atomic_choice]

                                if prev_token and prev_token not in BINARY_OPS:
                                    _token = [ops_choice] + _token

                                _token = [token] + _token 
                                
                                corrected_tokens.extend(_token if ahead_token in DOT else _token + [')'])
                                indx += 2
                                continue          
                    else:
                        if ahead_token in {')'}:
                            if all(choice in NEG_ATOMIC_CONCEPTS for choice in choices):
                                atomic_choice = random.choice(list(choices))
                            else:
                                atomic_choice = random.choice(list(NEG_ATOMIC_CONCEPTS))
                            
                            if isinstance(atomic_choice, tuple):
                                _token = [token] + list(atomic_choice)
                            else:
                                _token = [token, atomic_choice]

                            if prev_token and prev_token in ROLES:
                                _token = ['.'] + _token

                            corrected_tokens.extend(_token)
                            cap, choices, curr_valid_syn_cum = None, None, []

                            indx += 1
                            continue
                        elif ahead_token in UNARY_OPS:
                            if prev_token and prev_token in ROLES:
                                if next_ahead_token and next_ahead_token in ATOMIC_CONCEPTS:
                                    corrected_tokens.extend([token, '.', ahead_token, next_ahead_token])
                                    cap, choices, curr_valid_syn_cum = None, None, []
                                    indx += 3
                                    continue
                            elif (indx - 2 != 0 and tokens[indx-2] in ROLES) and prev_token and prev_token in DOT:
                                corrected_tokens.extend([token, ahead_token])

                                if next_ahead_token:
                                    if next_ahead_token in ATOMIC_CONCEPTS and (indx + 3) < len(tokens) and tokens[indx + 3] in {')'}:
                                        corrected_tokens.extend([next_ahead_token, tokens[indx + 3]])
                                        indx += 2
                                indx += 1
                                cap, choices, curr_valid_syn_cum = None, None, []
                                continue
                            elif cap == 3 and len(curr_valid_syn_cum) == 1:
                                dot_choice = list(choices) if choices in DOT else '.'
                                corrected_tokens.append(dot_choice)

                            atomic_choice = random.choice(list(ATOMIC_CONCEPTS))
                            corrected_tokens.extend([token, ahead_token, atomic_choice])
                            cap, choices, curr_valid_syn_cum = None, None, []

                            indx +=2
                            continue
                elif ahead_token in QUANTIFIERS:
                    if curr_valid_syn_cum: # atomic_choices are available
                        if all(choice in NEG_ATOMIC_CONCEPTS for choice in choices):
                            atomic_choice = random.choice(list(choices))
                        else:
                            atomic_choice = random.choice(list(NEG_ATOMIC_CONCEPTS)) # NEG_ATOMIC_CONCEPTS


                        if isinstance(atomic_choice, tuple):
                            _token = [token] + list(atomic_choice)
                        else:
                            _token = [token, atomic_choice]    
                        
                        if next_ahead_token and next_ahead_token not in {')'} or next_ahead_token not in BINARY_OPS:
                            _token += [')']
                        corrected_tokens.extend(_token)
                        cap, choices, curr_valid_syn_cum = None, None, []

                        if (next_ahead_token and next_ahead_token in ROLES):
                            ops_choice = random.choice(list(BINARY_OPS))
                            corrected_tokens.append(ops_choice)
                            indx +=1
                            continue

                        indx +=2
                        continue
                    else:
                        if not next_ahead_token:
                            atomic_choice = random.choice(list(ATOMIC_CONCEPTS))
                            corrected_tokens.append(token)
                            tokens[indx+1] = atomic_choice
                            indx +=1
                            continue
                elif prev_token:
                    if prev_token in ATOMIC_CONCEPTS:
                        ops_choice = random.choice(list(BINARY_OPS))
                        corrected_tokens.extend([ops_choice, token])
                        indx +=1
                        continue
        
        if token == ')':
            if curr_valid_syn_cum:
                if cap == 3 and len(curr_valid_syn_cum) == 2:
                    atomic_choice = random.choice(list(ATOMIC_CONCEPTS))
                    corrected_tokens.append(atomic_choice)
                    cap, choices, curr_valid_syn_cum = None, None, []
                    indx +=1
                    continue

        if not curr_valid_syn_cum and token in ATOMIC_CONCEPTS | ROLES and ahead_token:
            if token in ATOMIC_CONCEPTS:
                ops_choice = random.choice(list(BINARY_OPS))

                if ahead_token in ATOMIC_CONCEPTS and prev_token not in {')'}:
                    corrected_tokens.extend([token, ops_choice])
                    indx +=1
                    continue
                elif ahead_token in ROLES:
                    quant_choice = random.choice(list(QUANTIFIERS))
                    corrected_tokens.extend([token, ops_choice, quant_choice])
                    indx +=1
                    continue
            else:
                _token = [token, '.']
                if ahead_token and ahead_token in ATOMIC_CONCEPTS:
                    corrected_tokens.extend(_token + [ahead_token])
                    indx +=2
                    continue
                elif ahead_token and ahead_token in BINARY_OPS:
                    atomic_choice = random.choice(list(NEG_ATOMIC_CONCEPTS))

                    if isinstance(atomic_choice, tuple):
                        choice = list(atomic_choice)
                    else:
                        choice = [atomic_choice]

                    corrected_tokens.extend(_token + choice)
                    indx +=1
                    continue

        # --- 3. Progressive construction ---
        if token not in BINARY_OPS | PARENTHESES | QUANTIFIERS:
            _token = token

            if _token in UNARY_OPS | ATOMIC_CONCEPTS:
                if _token in UNARY_OPS:
                    if not cap:
                        cap = cap_curr_val['neg']
                    choices = ATOMIC_CONCEPTS
                else:
                    if not cap:
                        cap = cap_curr_val['atom']
                    choices = BINARY_OPS
            elif _token in DIGITS | DOT | ROLES:
                if _token in DIGITS: #TODO: Work on this later
                    if not cap:
                        cap = cap_curr_val['roleCard']
                    if curr_valid_syn_cum and (ahead_token and ahead_token in DIGITS):
                        _token += ahead_token
                        indx += 1  # skip next
                    choices = ROLES
                elif _token in ROLES:
                    if not cap:
                        cap = cap_curr_val['role']
                    choices = DOT
                elif _token in DOT:
                    choices = NEG_ATOMIC_CONCEPTS
            curr_valid_syn_cum.append(_token)

        # --- 4. Token validation ---
        if not is_valid_next_token(token, corrected_tokens):
            token = random.choice(list(choices))
            curr_valid_syn_cum.append(token)

        if isinstance(token, tuple):
            corrected_tokens.extend(token)
        else:
            corrected_tokens.append(token)

        if len(curr_valid_syn_cum) == cap or (curr_valid_syn_cum and curr_valid_syn_cum[0] == '.' and len(curr_valid_syn_cum) == 2):
            cap, choices, curr_valid_syn_cum = None, None, []

        indx +=1
    return corrected_tokens

def fix_mid_tokens_errors(tokens: list[str]) -> list[str]:
    container = []
    i = 0

    while i < len(tokens):
        prev_token = tokens[i - 1] if i - 1 >= 0 else None
        token = tokens[i]
        next_token = tokens[i + 1] if i + 1 < len(tokens) else None
        next_next_token = tokens[i + 2] if i + 2 < len(tokens) else None

        if (
            prev_token == '(' and
            token in BINARY_OPS and
            next_token == '.' and
            next_next_token):
            if next_next_token in ATOMIC_CONCEPTS | UNARY_OPS:
                i += 2
            else:
                container.append(choice(list(ATOMIC_CONCEPTS)))
                container.append(token)
                i += 1
            i += 1
            continue

        if prev_token in ROLES and token == '(':
            if next_token:
                if next_token in ATOMIC_CONCEPTS | UNARY_OPS:
                    container.append('.')
                    container.append(token)
                elif next_token in DOT and next_next_token and next_next_token in ATOMIC_CONCEPTS | UNARY_OPS:
                    i += 1
                    container.append('.')
                    container.append(token)
                i += 1
                continue

        if prev_token == '(' and token == '.' and next_token == ')':
            container.append(choice(list(ATOMIC_CONCEPTS)))
            i += 1
            continue
        
        if token == prev_token and token in BINARY_OPS | QUANTIFIERS | {'.'}:
            i += 1
            continue

        if prev_token in BINARY_OPS and token in BINARY_OPS:
            i += 1
            continue

        if token == '.' and prev_token in {'(', ')'}:
            i += 1
            continue

        if prev_token == '(' and token in BINARY_OPS:
            i += 1
            continue

        if prev_token in BINARY_OPS and token == ')':
            container.pop() 
            container.append(token)
            i += 1
            continue

        if prev_token == ')' and token == '(':
            container.append(choice(list(BINARY_OPS)))
            container.append(token)
            i += 1
            continue

        if (prev_token == ')' and token not in BINARY_OPS and next_token == '('):
            container.append(choice(list(BINARY_OPS)))
            i += 1  
            continue

        if prev_token in ATOMIC_CONCEPTS | {')'} and token in ATOMIC_CONCEPTS | UNARY_OPS | {'('} | DOT:
            container.append(choice(list(BINARY_OPS)))
            if token in DOT:
                i += 1
            else:
                container.append(token)
                i += 1
            continue

        if prev_token in BINARY_OPS and token in QUANTIFIERS and next_token in BINARY_OPS:
            container.append(choice(list(ATOMIC_CONCEPTS)))
            i += 1
            continue

        container.append(token)
        i += 1

    return container

def postprocess_tail_fix(tokens: list[str], max_length: int) -> list[str]:
    def is_incomplete_tail(toks):
        if not toks:
            return True
        return toks[-1] in BINARY_OPS | QUANTIFIERS | UNARY_OPS | {'.', '(', *ROLES}

    def minimal_completion_after(toks):
        last = toks[-1] if toks else None
        remaining = max_length - len(toks)

        if last is None:
            return [choice(list(ATOMIC_CONCEPTS))]

        if last in BINARY_OPS:
            return [choice(list(ATOMIC_CONCEPTS))] if remaining >= 1 else []

        if last in QUANTIFIERS:
            return [choice(list(ROLES)), '.', choice(list(ATOMIC_CONCEPTS))] if remaining >= 3 else []

        if last in DIGITS:
            return [choice(list(ROLES)), '.', choice(list(ATOMIC_CONCEPTS))] if remaining >= 3 else []

        if last in ROLES:
            if len(toks) >= 2 and toks[-2] in QUANTIFIERS:
                return ['.', choice(list(ATOMIC_CONCEPTS))] if remaining >= 2 else []
            return []

        if last == '.':
            if len(toks) >= 2 and toks[-2] in ROLES:
                return [choice(list(ATOMIC_CONCEPTS))] if remaining >= 1 else []
            return []

        if last in UNARY_OPS:
            return [choice(list(ATOMIC_CONCEPTS))] if remaining >= 1 else []

        if last == '(':
            return [choice(list(ATOMIC_CONCEPTS)), ')'] if remaining >= 2 else []

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

def balance_flatten_parentheses(tokens: list[str], max_length: int = None) -> list[str]:
    stack, result = [], []

    # First pass: balance parentheses
    for token in tokens:
        if token == '(':
            stack.append(len(result))
            result.append(token)
        elif token == ')':
            if stack:
                stack.pop()
                result.append(token)
            # else skip unmatched ')'
        else:
            result.append(token)

    # Handle unmatched '('
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

    # Second pass: flatten redundant ((...))
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

            # If we had exactly one nested pair: remove outer
            if j < len(result) and result[j] == ')':
                result = result[:i+1] + result[i+2:j] + result[j+1:]
                continue  # recheck from i after collapse
        i += 1

    return result


invalid_sequences = [

    # Not feasible cases
    # ['(', '⊓', '⊓', '(', 'Person', '⊔', '(', '∃', 'hasSibling', '.', 'Sister', ')', ')', '⊓', '(', '∀', 'hasChild', '.', '(', '¬', ')'],
    # ['(', '⊓', '⊓', '(', 'Person', '⊔', '(', '∃', 'hasSibling', '.', 'Brother', ')', ')', '⊓', '(', '∀', 'hasChild', '.', '(', ')'],
    # ['(', '⊓', '⊓', '(', '(', 'Granddaughter', ')', '(', ')', '.', '(', ')', '(', ')'], 
    
    # ['Person', '⊓', '(', '∀', 'married', '.', '(', 'Grandmother', ')', '(', '¬', ')', ')'],
    # ['Person', '⊓', '(', '(', '⊔', '(', '∃', 'married', '.', 'Granddaughter', ')', ')', '⊓', '(', '∀', 'hasParent', '.', '(', ')', ')']


    # ##### working cases
    ['(', '⊓', '(', 'Granddaughter', '⊔', '(', '∀', ')', ')', ')', ')', ')', ')', '.', ')'],
    # ['Mother', '⊔', '(', '∃', 'married', '.', '(', '¬', 'Grandparent', ')', '(', ')', ')', ')'],
    # ['Person', '⊓', '(', 'Brother', '⊔', '(', '∀', 'married', '.', '(', ')', '⊓', '(', '⊓', '(', 'Son', ')', ')'],
    # ['Person',  '⊓',  '(', 'Mother',  '⊔',  '(', 'Daughter',  ')',  '(', '¬', '(', ')', ')', ')'],
    # ['Person', '⊓', '(', 'Sister', '.', '(', 'Brother', ')', ')', ')', ')', ')', ')'],
    # ['Person', '⊓', '(', '∃', '⊔', '.', '(', '(', '⊓', '(', ')', ')', ')'],
    # ['Person', '⊓', '(', 'Mother', '⊔', '(', 'Sister', ')', '(', '¬', ')', ')', ')'],
    # ['Person', '⊔', '(', '∃', 'hasSibling', '.', '(', 'Sister', ')', '(', '∀', 'hasChild', '.', '(', '¬', 'Grandson', ')', ')', ')', ')'],
    # ['∃', '⊓', '(', 'Sister', ')', '∀', ')', '.', ')', ')', ')'],
    # ['∃', '⊓', '(', 'Sister', ')', '(', '∀', ')', '.', ')', ')', ')'],
    # ['(', '⊓', '(', '∀', 'married', '.', '(', '(', ')', ')', '(', '¬', ')', '(', ')', ')', ')'],
    # ['Person', '⊓', '(', 'Mother', '⊔', '(', '∀', 'hasSibling', '.', '(', '¬', 'Mother', ')', ')', ')', '(', ')', ')', ')'],
    # ['Person', '⊓', '(', 'Grandmother', '⊔', '(', '∃', 'married', '.', '(', '¬', ')', ')', ')', ')', '(', '∀'],
    # ['Brother', '⊔', '(', '¬', 'married'],
    # ['Sister', '⊔', '(', '∃', 'married', '.', ')'],
    # ['Father', '⊔', '(', '∃', 'hasSibling'],
    # ['∀', '⊔', '(', '¬', 'married'],
    # ['Father', '⊔', '(', '¬', 'married', '.'],
    # ['∀', '⊔', '(', '¬', 'married', '.'],
    # ['(', 'Parent', '⊓', '(', '∀', 'hasSibling', '.', '∀', ')', ')', '⊔', '(', ')', ')', ')'],
    # ['(', 'Daughter', '⊓', '(', '(', '¬', 'married', '.', '.', ')', ')', ')', ')', ' '],
    # ['Brother', '⊔', '(', '¬', 'hasSibling', '.', 'Sister', ')'],
    # ['Brother', '⊔', '(', 'married', 'Sister', ')'],
    # ['Person', '⊔', '(', '⊔', 'Sister'],
    # ['Person', '⊓', '(', '∀', 'hasSibling', '.', '(', '∃', 'married', '(', ')', ')', ')', '(', ')', ')'],
    # ['Person', '⊓', '(', '(', '⊔', '(', '∀', 'hasChild', '.', 'Daughter', ')', ')'],
    # ['Person', '⊔', '(', '∃', 'hasSibling', '.', '(', '∃', '⊓', '(', ')', 'hasChild', ')', 'Father', ')', ')', ')'],
    # ['∀', 'hasChild', '(', '¬', 'Father', ')'],
    # ['(', '⊓', '∀', 'hasSibling', '(', '(', ')', '.', 'Daughter', ')', ')', '⊔', '(', '∃', 'hasParent', '.', ')'],
    # ['Person', '⊓', '(', '∀', 'married', '(', '∃', 'hasSibling', '.', 'Parent', ')', ')'],
    # ['Person', '⊓', '(', '∃', 'married', '.', '(', '∀', 'hasChild', '.', '(', ')', ')', ')'],
    # ['Person', '⊓', '(', '∀', '(', '¬', ')', ')', ')', ')'],
    # ['Brother', '⊔', '(', '∃', '(', '∀', '.', ')', ')'],
    # ['Person', '⊓', '(', 'Sister', '⊔', '(', '∀', 'hasChild', '.', '(', '¬', 'Mother', ')', ')', ')', '⊓', '(', '∀', 'hasParent'],
    # ['Person', '⊓', '(', '∀', 'married', '.', '(', 'Granddaughter', '⊔', '(', ')', 'hasChild', '.', '(', ')', ')'],
    # ['Person', '⊓', '(', '∀', '⊔', '(', 'Grandmother', '⊔', 'Grandchild', '∃', ')', 'married', '.', ')', ')', ')'],
    # ['Person', '⊓', '(', '∀', 'hasSibling', '.', '(', 'Grandmother', '⊓', '(', ')', 'hasSibling', '.', '(', ')', ')', ')'],
    # ['Person', '⊓', '(', '∀', 'hasChild', '.', '(', 'Grandchild', '⊔', '(', ')', 'hasChild', 'Child', ')', ')', ')'],
    # ['Person', '⊓', '(', '∀', 'hasChild', '.', '(', 'Grandchild', '⊔', '(', ')', 'hasChild', '.','Child', ')', ')', ')'],
    # ['Person', '⊓', '(', '∀', '.', '(', 'Granddaughter', '⊔', '(', '∀', 'hasSibling', '.', '(', ')', ')', ')'],
    # ['Person', '⊓', '(', 'Granddaughter', '⊔', '(', '∃', 'hasSibling', '.', '(', '¬', 'Brother', ')', ')', ')', '⊓', '(', '∀', 'hasParent', '.'],
    # ['Person',  '⊓',  '(', '∀',  'married', '.', '(', 'Brother',  'hasChild',  'Sister', ')', ')'],
    # ['Person', '⊔', '(', '∃', 'hasChild', '.', '(', ')', ')', '(', ')', ')'],
    # ['∀', 'married', '.', '(', '(', 'Brother', '⊔', '¬', 'Sister', ')', '⊓', 'Grandson'],
    # ['Person', '⊓', '(', '∀', '⊔', '(', '∀', 'hasChild', '.', '(', ')', '⊓', '¬', '∀', ')', ')', ')'],
    # ['(', '⊓', '(', 'Brother', '⊔', '(', '(', ')', '.', '(', ')', ')', ')', ')', ')'],
    # ['(',  '⊓',  '(', 'Brother',  '⊔',  '(', '(',  ')', ')', 'Sister', ')', ')'],
    # ['Person', '⊓', '(', 'Brother', '⊔', '(', '∀', 'hasParent', '.', 'Female', '(', 'Brother', ')'],
    # ['Father', '⊔', '(', '∃', 'married', '.', '(', '¬', 'Grandson', ')'],
    # ['PersonWithASibling', '⊔', '(', '∀', 'hasChild', '.', '(', '¬', 'Brother',')'],
    # ['Person', '⊓', '(', '∀', '⊔', '.', '(', '∃', 'hasChild', '.', '(', '¬', 'Sister', ')', ')', ')', ')'],
    # ['Person', '⊓', '(', 'Grandson', '⊔', '(', '∃', 'married', '.', '¬','Mother', ')', '⊓', ')', ')', ')', '⊓'], 
    # ['Person', '⊓', '(', 'Grandson', '⊔', '(', '∃', 'married', '.','Mother', ')', '⊓', ')', ')', ')', '⊓'], 
    # ['Person', '⊓', '(', 'Sister', '⊔', '(', '∀', 'hasChild', '.', 'Son', ')', ')', '⊓', '(', '∀', ')'],
    # ['Brother', '⊔', 'Grandson', '⊓', '∀'],
    # ['Person', '⊔', '∀', 'hasSibling', '.', '¬'],
    # ['Mother', '⊓', '¬'],
    # ['Female', '⊔', '(', '∀', 'hasChild', '.', '('],
    # ['Person', '⊓', '∃', 'married', '.', 'Granddaughter', '⊔', '.', 'Grandson'],
    # ['Granddaughter', '⊔', '(', '∃', 'married', '.', '(', '(', ')', ')'],
    # ['Person', '⊓', '(', '∀', 'married', '.', '(', 'Granddaughter', '⊔', '(', '∀', 'hasChild', '.', 'PersonWithASibling', ')', ')', ')'],
    # ['(', '⊓', '(', ')', '⊔', '(', '∃', '.', '.', ')', ')', ')', ')', ')'],
    # ['Person', '⊓', '(', '∀', '⊔', '.', '(', '∃', 'hasChild', '.', '(', '¬', ')', ')', ')', ')'],
    # ['Person', '⊓', '(', 'Daughter', '⊔', '(', '∀', 'married', '.', '(', ')', ')', ')', ')', ')', '(', '⊓', '(', '.', ')', ')'],
    # ['PersonWithASibling', '⊔', '(', '∃', 'hasChild', '.', '(', '∀', ')', ')', ')'],
    # ['PersonWithASibling', '⊔', '(', '∀', 'hasChild', '.', '(', '¬', ')'],
    # ['Person', '⊔', '(', '∃', 'married', '.', '(', ')'],
    # ['Person', '⊔', '(', '∃', 'married'],
    # [')', '⊓', 'Person'],
    # ['Animal', 'hasChild', 'Mother'],
    # ['∃', 'Person', ')'],
    # ['(', '⊓', '∃', 'Person', ')'],
    # ['Person', 'Mother', 'Thing'],
    # ['(', 'Granddaughter','⊓','(', 'Grandson','⊔','(', '¬',')', '.', 'Daughter', ')', ')'],
    # ['Person', '⊓', '(', '∃', '⊔', '.', '(', 'Granddaughter', '⊔', '(', '∃', 'hasChild', '(', '¬', ')', ')', ')', ')', ')'],
    # ['Person', '⊔', '(', '∃', 'hasSibling', '.', '(', '∃', '⊓', '(', ')', ')'],
    # ['¬', 'Brother', '¬', 'Animal', '⊓', '(', '(', 'Granddaughter', ')', 'Mother', 'Thing', ')'],
    # ['Person','⊓','(', '∀','hasSibling', '.', '(', 'Sister','⊓','(',')', ')', ')'], 
    # ['(','Person', '⊓', '(', '∀', '.', '(', '(', ')', '⊔', ')', ')'], 
    # ['Person', '⊓', '(', 'Grandson', '⊔', '(', '∃', 'married', '.', 'Mother', ')', '⊓', ')', ')', ')'], 
    # ['Person', '⊔', '(', '∃', 'married', '.', '(', '∀', '⊓', '(', '¬', ')', ')', ')'],
    # ['(', '⊓', '(', 'Grandson', '⊔', '(', '¬', ')', ')', '.', ')', ')', ')', ')', '(', '.', ')'],
    # ['(', '⊓', '(', '∀', 'hasChild', '.', '(', 'Child', ')', '(', ')', '⊔', '(', '∀', ')', '.', ')', ')'],
    # ['Person', '⊓', '(', '∀', '⊔', '(', '∀', ')', '(', '¬', ')', ')', ')'],
    # ['¬','Person', '⊓', '(', 'Animal', '⊔', '(', '∀', 'hasChild', '.', '(', ')', ')', ')'],
    # ['Person', '⊔', '(', '∀', 'hasChild', '.', '(', ')'],
    # ['(', '⊔', '(', '∃', '.', '(', 'Child', ')', ')'],
    # ['Person', '⊓', '(', '∃', 'married', '.', '(', 'Granddaughter', '⊔', '(', '∀', 'hasChild', '.', '(', ')', ')', ')', ')'],
    # ['(', '⊓', '(', '(', '(', '∀', ')', '.', ')', ')'],
    # ['∃', '⊔', '(', '∀', 'hasChild', '.', '(', 'Brother', ')'],
    ]

for seq in invalid_sequences:
    print(seq)
    valid_sequence = validate_and_correct_sequence(seq, 20)
    # print(valid_sequence)
    # valid_sequence = fix_mid_tokens_errors(seq)
    tokens = fix_mid_tokens_errors(valid_sequence)
    # print(tokens)
    post_fix  = postprocess_tail_fix(tokens, 50)
    post_fix_paren_balance = balance_flatten_parentheses(post_fix)

    print(post_fix_paren_balance)
    print()


