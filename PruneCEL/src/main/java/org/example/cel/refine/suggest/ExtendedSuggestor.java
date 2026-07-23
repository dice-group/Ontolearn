package org.example.cel.refine.suggest;

import java.util.Collection;

import org.example.cel.expression.ClassExpression;

/**
 * An extension of the {@link Suggestor} interface that is able to score a given
 * expression.
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 *
 */
public interface ExtendedSuggestor extends Suggestor {

    /**
     * This method scores a single expression based on the given positive and
     * negative examples. Please note that this method can be quite <b>expensive</b>
     * compared to the suggestion methods.
     * 
     * @param expression the expression that should be scored
     * @param positive   the positive examples
     * @param negative   the negative examples
     * @return the scored expression based on the number of positive and negative
     *         examples that the given expression selects
     */
    SelectionScores scoreExpression(ClassExpression expression, Collection<String> positive,
            Collection<String> negative);
}
