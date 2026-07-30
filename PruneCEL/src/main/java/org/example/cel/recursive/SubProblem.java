package org.example.cel.recursive;

import java.util.Collection;

import org.example.cel.expression.ScoredClassExpression;
import org.example.cel.refine.RefinementOperator;
import org.example.cel.score.ScoreCalculator;

public class SubProblem {

    protected ScoredClassExpression parentExpression;
    protected Collection<String> positive;
    protected Collection<String> negative;
    protected ScoreCalculator scoreCalculator;
    protected RefinementOperator rho;

    public SubProblem(ScoredClassExpression parentExpression, Collection<String> positive,
            Collection<String> negative, ScoreCalculator scoreCalculator, RefinementOperator rho) {
        super();
        this.parentExpression = parentExpression;
        this.positive = positive;
        this.negative = negative;
        this.scoreCalculator = scoreCalculator;
        this.rho = rho;
    }

    /**
     * @return the parentExpression
     */
    public ScoredClassExpression getParentExpression() {
        return parentExpression;
    }

    /**
     * @param parentExpression the parentExpression to set
     */
    public void setParentExpression(ScoredClassExpression parentExpression) {
        this.parentExpression = parentExpression;
    }

    /**
     * @return the positive
     */
    public Collection<String> getPositive() {
        return positive;
    }

    /**
     * @param positive the positive to set
     */
    public void setPositive(Collection<String> positive) {
        this.positive = positive;
    }

    /**
     * @return the negative
     */
    public Collection<String> getNegative() {
        return negative;
    }

    /**
     * @param negative the negative to set
     */
    public void setNegative(Collection<String> negative) {
        this.negative = negative;
    }

    /**
     * @return the scoreCalculator
     */
    public ScoreCalculator getScoreCalculator() {
        return scoreCalculator;
    }

    /**
     * @param scoreCalculator the scoreCalculator to set
     */
    public void setScoreCalculator(ScoreCalculator scoreCalculator) {
        this.scoreCalculator = scoreCalculator;
    }

    /**
     * @return the RefinementOperator
     */
    public RefinementOperator getRefinementOperator() {
        return rho;
    }

    /**
     * @param rho the RefinementOperator to set
     */
    public void setRefinementOperator(RefinementOperator rho) {
        this.rho = rho;
    }

}