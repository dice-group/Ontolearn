package org.example.cel.refine.suggest;

import java.util.Objects;

/**
 * This data class is an extension of the {@link SelectionScores} class and
 * stores an IRI together with its selection scores. It also stores whether the
 * IRI has been inverted (which only makes sense if the IRI is used as a role).
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 *
 */
public class ScoredIRI extends SelectionScores {

    /**
     * The IRI that has been scored.
     */
    public String iri;
    /**
     * A flag that indicates whether the {@link #iri} has to be used as an inverted
     * role to achieve the score.
     */
    public boolean inverted;

    /**
     * Constructor.
     * 
     * @param iri      The IRI that has been scored
     * @param posCount The number of positive examples that have been selected
     * @param negCount The number of negative examples that have been selected
     */
    public ScoredIRI(String iri, int posCount, int negCount) {
        this(iri, posCount, negCount, false);
    }

    /**
     * Constructor.
     * 
     * @param iri      The IRI that has been scored
     * @param posCount The number of positive examples that have been selected
     * @param negCount The number of negative examples that have been selected
     * @param inverted A flag that indicates whether the given IRI has to be used as
     *                 an inverted role to achieve the score
     */
    public ScoredIRI(String iri, int posCount, int negCount, boolean inverted) {
        super(posCount, negCount);
        this.iri = iri;
        this.inverted = inverted;
    }

    /**
     * @return the iri
     */
    public String getIri() {
        return iri;
    }

    /**
     * @param iri the iri to set
     */
    public void setIri(String iri) {
        this.iri = iri;
    }

    /**
     * @return the inverted
     */
    public boolean isInverted() {
        return inverted;
    }

    /**
     * @param inverted the inverted to set
     */
    public void setInverted(boolean inverted) {
        this.inverted = inverted;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("ScoredIRI [iri=");
        builder.append(iri);
        builder.append(", posCount=");
        builder.append(posCount);
        builder.append(", negCount=");
        builder.append(negCount);
        builder.append(", inverted=");
        builder.append(inverted);
        builder.append("]");
        return builder.toString();
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = super.hashCode();
        result = prime * result + Objects.hash(inverted, iri);
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (!super.equals(obj))
            return false;
        if (getClass() != obj.getClass())
            return false;
        ScoredIRI other = (ScoredIRI) obj;
        return inverted == other.inverted && Objects.equals(iri, other.iri);
    }


}
