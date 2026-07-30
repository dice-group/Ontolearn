package org.example.cel.score;

public interface ScoreCalculatorFactory {

    ScoreCalculator create(int numOfPositives, int numOfNegatives);

}
