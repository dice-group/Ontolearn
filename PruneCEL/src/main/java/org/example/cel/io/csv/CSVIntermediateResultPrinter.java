package org.example.cel.io.csv;

import java.io.PrintStream;
import java.util.Deque;
import java.util.LinkedList;

import org.dice_research.topicmodeling.commons.collections.TopDoubleObjectCollection;
import org.example.cel.expression.ScoredClassExpression;
import org.example.cel.io.IntermediateResultPrinter;

public class CSVIntermediateResultPrinter implements IntermediateResultPrinter, AutoCloseable {

    protected PrintStream pout;
    protected long startTime = 0L;
    protected Deque<Double> previousPerformances = new LinkedList<Double>();

    public CSVIntermediateResultPrinter(PrintStream pout) {
        this.pout = pout;
        pout.print("time,rel.time,recursion,positives,negatives,cscore,rscore,expression");
        previousPerformances.add(Double.NEGATIVE_INFINITY);
    }

    @Override
    public void printIntermediateResults(TopDoubleObjectCollection<ScoredClassExpression> topExpressions) {
        ScoredClassExpression exp = (ScoredClassExpression) topExpressions.objects[0];
        if (exp.getClassificationScore() > previousPerformances.peekLast()) {
            pout.print(System.currentTimeMillis());
            pout.print(',');
            pout.print(System.currentTimeMillis() - startTime);
            pout.print(',');
            pout.print(previousPerformances.size());
            pout.print(',');
            pout.print(exp.getPosCount());
            pout.print(',');
            pout.print(exp.getNegCount());
            pout.print(',');
            pout.print(exp.getClassificationScore());
            pout.print(',');
            pout.print(exp.getRefinementScore());
            pout.print(',');
            pout.println(exp.getClassExpression().toString());
            previousPerformances.pollLast();
            previousPerformances.add(exp.getClassificationScore());
        }
    }

    @Override
    public void close() throws Exception {
        pout.close();
    }

    @Override
    public void setStartTime(long startTime) {
        this.startTime = startTime;
    }

    @Override
    public void recursionStarts() {
        previousPerformances.add(Double.NEGATIVE_INFINITY);
    }

    @Override
    public void recursionEnds() {
        previousPerformances.pollLast();
    }

}
