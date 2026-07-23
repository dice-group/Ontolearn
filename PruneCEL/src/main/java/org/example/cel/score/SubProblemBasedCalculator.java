package org.example.cel.score;

import org.example.cel.expression.ClassExpression;

@Deprecated
public class SubProblemBasedCalculator extends AbstractScoreCalculatorDecorator {

    protected int parentPosCount = 0;
    protected int parentNegCount = 0;

    public SubProblemBasedCalculator(ScoreCalculator decorated) {
        super(decorated);
    }

    @Override
    public double calculateClassificationScore(int posCount, int negCount) {
        return super.calculateClassificationScore(posCount + parentPosCount, negCount + parentNegCount);
    }

    @Override
    public double calculateRefinementScore(int posCount, int negCount, double classificationScore, ClassExpression ce) {
        return super.calculateRefinementScore(posCount + parentPosCount, negCount + parentNegCount, classificationScore,
                ce);
    }
}
