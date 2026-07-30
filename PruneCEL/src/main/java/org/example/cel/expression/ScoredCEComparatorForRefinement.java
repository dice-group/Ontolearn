package org.example.cel.expression;

import java.util.Comparator;

public class ScoredCEComparatorForRefinement implements Comparator<ScoredClassExpression> {

    @Override
    public int compare(ScoredClassExpression ce1, ScoredClassExpression ce2) {
        int diff = Double.compare(ce1.getRefinementScore(), ce2.getRefinementScore());
        if (diff == 0) {
            return 0;
        } else {
            return -diff;
        }
    }

}
