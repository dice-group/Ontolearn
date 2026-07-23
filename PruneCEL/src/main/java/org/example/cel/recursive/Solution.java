package org.example.cel.recursive;

import java.util.Collection;

import org.example.cel.expression.ScoredClassExpression;

public class Solution {
    protected SubProblem subProblem;
    protected ScoredClassExpression partialSolution;
    protected ScoredClassExpression mergedSolution;

    public Solution(SubProblem subProblem, ScoredClassExpression partialSolution) {
        this(subProblem, partialSolution, partialSolution);
    }

    public Solution(SubProblem subProblem, ScoredClassExpression partialSolution,
            ScoredClassExpression mergedSolution) {
        this.subProblem = subProblem;
        this.partialSolution = partialSolution;
        this.mergedSolution = mergedSolution;
    }

    /**
     * @return the subProblem
     */
    public SubProblem getSubProblem() {
        return subProblem;
    }

    /**
     * @param subProblem the subProblem to set
     */
    public void setSubProblem(SubProblem subProblem) {
        this.subProblem = subProblem;
    }

    /**
     * @return the partialSolution
     */
    public ScoredClassExpression getPartialSolution() {
        return partialSolution;
    }

    /**
     * @param partialSolution the partialSolution to set
     */
    public void setPartialSolution(ScoredClassExpression partialSolution) {
        this.partialSolution = partialSolution;
    }

    /**
     * @return the mergedSolution
     */
    public ScoredClassExpression getMergedSolution() {
        return mergedSolution;
    }

    /**
     * @param mergedSolution the mergedSolution to set
     */
    public void setMergedSolution(ScoredClassExpression mergedSolution) {
        this.mergedSolution = mergedSolution;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((mergedSolution == null) ? 0 : mergedSolution.hashCode());
        result = prime * result + ((partialSolution == null) ? 0 : partialSolution.hashCode());
        result = prime * result + ((subProblem == null) ? 0 : subProblem.hashCode());
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Solution other = (Solution) obj;
        if (mergedSolution == null) {
            if (other.mergedSolution != null)
                return false;
        } else if (!mergedSolution.equals(other.mergedSolution))
            return false;
        if (partialSolution == null) {
            if (other.partialSolution != null)
                return false;
        } else if (!partialSolution.equals(other.partialSolution))
            return false;
        if (subProblem == null) {
            if (other.subProblem != null)
                return false;
        } else if (!subProblem.equals(other.subProblem))
            return false;
        return true;
    }

    public Collection<ScoredClassExpression> refine() {
        return subProblem.getRefinementOperator().refine(partialSolution.getClassExpression(), 0);
    }

}