package org.example.cel.score;

import org.example.cel.expression.ClassExpression;

public abstract class AbstractScoreCalculatorDecorator implements ScoreCalculator {

    private ScoreCalculator decorated;

    public AbstractScoreCalculatorDecorator(ScoreCalculator decorated) {
        super();
        this.decorated = decorated;
    }

    @Override
    public double calculateClassificationScore(int posCount, int negCount) {
        return decorated.calculateClassificationScore(posCount, negCount);
    }

    @Override
    public double calculateRefinementScore(int posCount, int negCount, double classificationScore, ClassExpression ce) {
        return decorated.calculateRefinementScore(posCount, negCount, classificationScore, ce);
    }

    /**
     * @return the decorated
     */
    public ScoreCalculator getDecorated() {
        return decorated;
    }

    /**
     * @param decorated the decorated to set
     */
    public void setDecorated(ScoreCalculator decorated) {
        this.decorated = decorated;
    }

    @Override
    public double getPerfectScore() {
        return decorated.getPerfectScore();
    }
}
