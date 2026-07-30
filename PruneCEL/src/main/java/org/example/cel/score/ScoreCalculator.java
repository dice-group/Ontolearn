package org.example.cel.score;

import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.ScoredClassExpression;

public interface ScoreCalculator {

    double calculateClassificationScore(int posCount, int negCount);

    default double calculateRefinementScore(int posCount, int negCount, double classificationScore,
            ClassExpression ce) {
        return classificationScore;
    }

    default ScoredClassExpression score(ClassExpression ce, int posCount, int negCount, boolean addedEdge) {
        double cScore = calculateClassificationScore(posCount, negCount);
        return new ScoredClassExpression(ce, cScore, calculateRefinementScore(posCount, negCount, cScore, ce), posCount,
                negCount, addedEdge);
    }

    double getPerfectScore();
}
