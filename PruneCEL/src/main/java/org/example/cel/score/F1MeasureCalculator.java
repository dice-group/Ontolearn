package org.example.cel.score;

public class F1MeasureCalculator implements ScoreCalculator {

    protected double numOfPositives;

    public F1MeasureCalculator(int numOfPositives) {
        super();
        this.numOfPositives = numOfPositives;
    }

    @Override
    public double calculateClassificationScore(int posCount, int negCount) {
        double precision = 0.0;
        double sum = posCount + negCount;
        if (sum > 0) {
            precision = posCount / sum;
        }
        double recall = 0.0;
        if (numOfPositives > 0) {
            recall = posCount / (double) numOfPositives;
        }
        sum = precision + recall;
        if (sum > 0) {
            return 2 * precision * recall / sum;
        } else {
            return 0;
        }
    }

    @Override
    public double getPerfectScore() {
        return 1.0;
    }

    public static class Factory implements ScoreCalculatorFactory {
        @Override
        public ScoreCalculator create(int numOfPositives, int numOfNegatives) {
            return new F1MeasureCalculator(numOfPositives);
        }
    }
}