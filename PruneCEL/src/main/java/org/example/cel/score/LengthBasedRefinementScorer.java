package org.example.cel.score;

import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.ClassExpressionVisitor;
import org.example.cel.expression.Junction;
import org.example.cel.expression.NamedClass;
import org.example.cel.expression.SimpleQuantifiedRole;

public class LengthBasedRefinementScorer extends AbstractScoreCalculatorDecorator {

    protected double lengthPenalty = 0.01;

    public LengthBasedRefinementScorer(ScoreCalculator decorated) {
        super(decorated);
    }

    public LengthBasedRefinementScorer(ScoreCalculator decorated, double lengthPenalty) {
        super(decorated);
        this.lengthPenalty = lengthPenalty;
    }

    @Override
    public double calculateRefinementScore(int posCount, int negCount, double classificationScore, ClassExpression ce) {
        LengthDetectingVisitor visitor = new LengthDetectingVisitor();
        ce.accept(visitor);
        return classificationScore - (visitor.length * lengthPenalty);
    }

    public static class Factory implements ScoreCalculatorFactory {
        protected ScoreCalculatorFactory decoratedFactory;

        public Factory(ScoreCalculatorFactory decoratedFactory) {
            super();
            this.decoratedFactory = decoratedFactory;
        }

        @Override
        public ScoreCalculator create(int numOfPositives, int numOfNegatives) {
            return new LengthBasedRefinementScorer(decoratedFactory.create(numOfPositives, numOfNegatives));
        }
    }

    /**
     * Length as defined in "Learning Concept Lengths Accelerates Concept Learning
     * in ALC"
     * 
     * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
     *
     */
    public static class LengthDetectingVisitor implements ClassExpressionVisitor {

        protected int length = 0;

        public LengthDetectingVisitor() {
            super();
        }

        @Override
        public void visitNamedClass(NamedClass node) {
            if (node.isNegated()) {
                length += 2;
            } else {
                length += 1;
            }
        }

        @Override
        public void visitJunction(Junction node) {
            length += 1;
            for (ClassExpression child : node.getChildren()) {
                child.accept(this);
            }
        }

        @Override
        public void visitSimpleQuantificationRole(SimpleQuantifiedRole node) {
            length += 2;
            node.getTailExpression().accept(this);
        }

    }

}
