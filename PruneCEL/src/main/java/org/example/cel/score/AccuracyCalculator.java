package org.example.cel.score;

public class AccuracyCalculator implements ScoreCalculator {

    // protected int numOfPositives;
    protected int numOfNegatives;
    protected double exampleSum;

    public AccuracyCalculator(int numOfPositives, int numOfNegatives) {
        super();
        // this.numOfPositives = numOfPositives;
        this.numOfNegatives = numOfNegatives;
        this.exampleSum = numOfPositives + numOfNegatives;
    }

    @Override
    public double calculateClassificationScore(int posCount, int negCount) {
        return (posCount + (numOfNegatives - negCount)) / exampleSum;
    }

    @Override
    public double getPerfectScore() {
        return 1.0;
    }

    public static class Factory implements ScoreCalculatorFactory {
        @Override
        public ScoreCalculator create(int numOfPositives, int numOfNegatives) {
            return new AccuracyCalculator(numOfPositives, numOfNegatives);
        }
    }
}
