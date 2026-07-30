package org.example.cel.refine;

import java.util.Set;

import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.ScoredClassExpression;

/**
 * Basic interface of a refinement operator.
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 *
 */
public interface RefinementOperator {

    /**
     * Given a class expression, the refinement operator creates a set of new class
     * expressions that are refinements of the given expression.
     * 
     * @param nextBestExpression the class expression that should be refined further
     * @param timeToStop         the point in time at which the refinement has to
     *                           stop. If the value is below or equal to 0, it will
     *                           be ignored.
     * @return a set of new class expressions that are refinements of the given
     *         expression.
     */
    Set<ScoredClassExpression> refine(ClassExpression nextBestExpression, long timeToStop);
}
