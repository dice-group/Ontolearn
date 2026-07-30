package org.example.cel.refine.suggest.sparql;

import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.ClassExpressionVisitingCreator;
import org.example.cel.expression.Junction;
import org.example.cel.expression.NamedClass;
import org.example.cel.expression.SimpleQuantifiedRole;

/**
 * A simple visitor that checks whether the expression contains a disjunction.
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 *
 */
public class DisjunctionCheckingVisitor implements ClassExpressionVisitingCreator<Boolean> {

    public boolean containsDisjunction(ClassExpression ce) {
        return ce.accept(this);
    }

    @Override
    public Boolean visitNamedClass(NamedClass node) {
        return Boolean.FALSE;
    }

    @Override
    public Boolean visitJunction(Junction node) {
        if (node.isConjunction()) {
            for (ClassExpression child : node.getChildren()) {
                if (child.accept(this)) {
                    return Boolean.TRUE;
                }
            }
            return Boolean.FALSE;
        } else {
            return Boolean.TRUE;
        }
    }

    @Override
    public Boolean visitSimpleQuantificationRole(SimpleQuantifiedRole node) {
        return node.getTailExpression().accept(this);
    }

}