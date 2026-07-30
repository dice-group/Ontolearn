package org.example.cel.sparql;

import java.util.Collection;
import java.util.Set;

import org.example.cel.expression.ClassExpression;

/**
 * Interface to access the single instances from the list of positive and
 * negative examples that are selected by a given class expression.
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 *
 */
public interface InstanceRetriever {

    /**
     * This method retrieves the single instances from the list of positive and
     * negative examples that are selected by the given class expression. Note that
     * this operation is expensive.
     * 
     * @param expression
     * @param positive
     * @param negative
     * @return
     */
    public Set<String> retrieveInstances(ClassExpression expression, Collection<String> positive,
            Collection<String> negative);
}
