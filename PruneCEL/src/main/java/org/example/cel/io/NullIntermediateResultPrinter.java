package org.example.cel.io;

import org.dice_research.topicmodeling.commons.collections.TopDoubleObjectCollection;
import org.example.cel.expression.ScoredClassExpression;

public class NullIntermediateResultPrinter implements IntermediateResultPrinter {

    @Override
    public void printIntermediateResults(TopDoubleObjectCollection<ScoredClassExpression> topExpressions) {
    }

    @Override
    public void setStartTime(long startTime) {
    }

    @Override
    public void recursionStarts() {
    }

    @Override
    public void recursionEnds() {
    }

}
