package org.example.cel.io;

import org.dice_research.topicmodeling.commons.collections.TopDoubleObjectCollection;
import org.example.cel.expression.ScoredClassExpression;

public interface IntermediateResultPrinter {
    
    void setStartTime(long startTime);

    void printIntermediateResults(TopDoubleObjectCollection<ScoredClassExpression> topExpressions);

    void recursionStarts();

    void recursionEnds();
}
