package org.example.cel.score;

public class BalancedAccuracyCalculator implements ScoreCalculator {

    protected double numOfPositives;
    protected double numOfNegatives;

    public BalancedAccuracyCalculator(int numOfPositives, int numOfNegatives) {
        super();
        this.numOfPositives = numOfPositives;
        this.numOfNegatives = numOfNegatives;
    }

    @Override
    public double calculateClassificationScore(int posCount, int negCount) {
        return ((posCount / numOfPositives) + ((numOfNegatives - negCount) / numOfNegatives)) / 2;
    }

    @Override
    public double getPerfectScore() {
        return 1.0;
    }

    public static class Factory implements ScoreCalculatorFactory {
        @Override
        public ScoreCalculator create(int numOfPositives, int numOfNegatives) {
            return new BalancedAccuracyCalculator(numOfPositives, numOfNegatives);
        }
    }
}
