package org.example.cel.refine.suggest.sparql;

import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.ClassExpressionVisitingCreator;
import org.example.cel.expression.Junction;
import org.example.cel.expression.NamedClass;
import org.example.cel.expression.SimpleQuantifiedRole;

/**
 * This class is a visitor that returns {@code true} if the given class
 * expression should not be put into a {@code FILTER NOT EXISTS} (FNE) statement
 * since the statement itself will consist of one (or several) FNE statements.
 * Instead, such an expression should be simply negated.
 * 
 * For example, trying to add an FNE statement with the expression ∀r.⊥ leads to
 * an FNE statement within an FNE, without any additional triple pattern in the
 * outer filter. That means that the variables within the filter and outside of
 * the filter are not connected, anymore. Hence, it is easier to add ∃r.⊤
 * instead. The same holds for negated, named classes (¬A) or a conjunction of
 * these two types of statements.
 * 
 * Note that the given statement should already have been preprocessed for being
 * transformed into a SPARQL query.
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 *
 */
public class FilterNotExistsChecker implements ClassExpressionVisitingCreator<Boolean> {

    @Override
    public Boolean visitNamedClass(NamedClass node) {
        return node.isNegated();
    }

    @Override
    public Boolean visitJunction(Junction node) {
        for (ClassExpression child : node.getChildren()) {
            if (!child.accept(this)) {
                return Boolean.FALSE;
            }
        }
        return Boolean.TRUE;
    }

    @Override
    public Boolean visitSimpleQuantificationRole(SimpleQuantifiedRole node) {
        return !node.isExists();
    }

}
