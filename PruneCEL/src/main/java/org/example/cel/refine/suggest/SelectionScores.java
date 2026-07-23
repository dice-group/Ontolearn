package org.example.cel.refine.suggest;

/**
 * A simple data class that stores the number of positive and negative examples
 * that an expression selected (without having an explicit relation to the
 * expression).
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 *
 */
public class SelectionScores {

    /**
     * The number of positive examples that have been selected.
     */
    public int posCount;
    /**
     * The number of negative examples that have been selected.
     */
    public int negCount;

    /**
     * Constructor.
     * 
     * @param posCount The number of positive examples that have been selected
     * @param negCount The number of negative examples that have been selected
     */
    public SelectionScores(int posCount, int negCount) {
        super();
        this.posCount = posCount;
        this.negCount = negCount;
    }

    /**
     * @return the posCount
     */
    public int getPosCount() {
        return posCount;
    }

    /**
     * @param posCount the posCount to set
     */
    public void setPosCount(int posCount) {
        this.posCount = posCount;
    }

    /**
     * @return the negCount
     */
    public int getNegCount() {
        return negCount;
    }

    /**
     * @param negCount the negCount to set
     */
    public void setNegCount(int negCount) {
        this.negCount = negCount;
    }

    /**
     * Adds the given scores to the internal scores.
     * 
     * @param posCount
     * @param negCount
     */
    public void add(int posCount, int negCount) {
        this.posCount += posCount;
        this.negCount += negCount;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + negCount;
        result = prime * result + posCount;
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
        SelectionScores other = (SelectionScores) obj;
        if (negCount != other.negCount)
            return false;
        if (posCount != other.posCount)
            return false;
        return true;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("SelectionScores [posCount=");
        builder.append(posCount);
        builder.append(", negCount=");
        builder.append(negCount);
        builder.append("]");
        return builder.toString();
    }
}
