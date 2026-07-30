package org.example.cel.score;

/**
 * Sets the refinement score of solutions that only select a single positive
 * example to 0.
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 *
 */
public class AvoidingPickySolutionsDecorator extends AbstractScoreCalculatorDecorator {

    public AvoidingPickySolutionsDecorator(ScoreCalculator decorated) {
        super(decorated);
    }

    public double calculateRefinementScore(int posCount, int negCount, double classificationScore,
            org.example.cel.expression.ClassExpression ce) {
        if (posCount <= 1) {
            return 0;
        } else {
            return super.calculateRefinementScore(posCount, negCount, classificationScore, ce);
        }
    };

    public static class Factory implements ScoreCalculatorFactory {
        protected ScoreCalculatorFactory decoratedFactory;

        public Factory(ScoreCalculatorFactory decoratedFactory) {
            super();
            this.decoratedFactory = decoratedFactory;
        }

        @Override
        public ScoreCalculator create(int numOfPositives, int numOfNegatives) {
            return new AvoidingPickySolutionsDecorator(decoratedFactory.create(numOfPositives, numOfNegatives));
        }
    }

}
