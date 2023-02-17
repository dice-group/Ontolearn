import numpy as np, pandas as pd
from scipy.stats import wilcoxon


def wilcoxon_statistical_test(data1, data2):
    stat, p = wilcoxon(data1, data2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    return stat, p
        

def get_Manchester_Syntax(concept_name: str) -> str:
    result = ''
    is_existential = True
    skip_space = False
    for c in concept_name:
        if c == '∀':
            is_existential = False
            skip_space = True
        if c == '∃':
            is_existential = True
            skip_space = True
        if not c in ['∃', '∀', '.', '⊔', '⊓', ' ', '¬', '⊤', '⊥']:
            result += c
        if c == '⊔':
            result += 'or'
        if c == '⊓':
            result += 'and'
        if c == '.' and is_existential:
            result += ' some '
        if c == '.' and not is_existential:
            result += ' only '
        if c == '¬':
            result += 'not '
        if c == '⊤':
            result += 'Thing' # DLFoil config may not support Thing, # hint: manually replace all Thing by some class name
        if c == '⊥':
            result += 'Nothing' # Should be Nothing, but Nothing is mostly not supported by DLFoil config on some knowledge bases, tips: manually replace Nothing or Thing by some class name
        if c == ' ' and skip_space:
            skip_space = False
        elif c == ' ' and not skip_space:
            result += c
    return result