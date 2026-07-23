package org.example.cel.refine.suggest.sparql;

import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.ClassExpressionVisitingCreator;
import org.example.cel.expression.Junction;
import org.example.cel.expression.NamedClass;
import org.example.cel.expression.SimpleQuantifiedRole;
import org.example.cel.refine.suggest.Suggestor;

/**
 * This visitor deletes the sub tree of a class expression that contains the
 * {@link Suggestor#CONTEXT_POSITION_MARKER}. The sub expression is deleted up
 * to the first disjunction. If the expression does not contain any
 * disjunctions, {@code null} is returned.
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 *
 */
public class SubExpressionDeleter implements ClassExpressionVisitingCreator<ClassExpression> {

    @Override
    public ClassExpression visitNamedClass(NamedClass node) {
        if (Suggestor.CONTEXT_POSITION_MARKER.equals(node)) {
            return null;
        } else {
            return node;
        }
    }

    @Override
    public ClassExpression visitJunction(Junction node) {
        ClassExpression[] newChildren = node.getChildren().stream().map(child -> child.accept(this))
                .filter(child -> child != null).toArray(ClassExpression[]::new);
        if ((newChildren.length != node.getChildren().size()) && (node.isConjunction())) {
            return null;
        } else {
            return new Junction(node.isConjunction(), newChildren);
        }
    }

    @Override
    public ClassExpression visitSimpleQuantificationRole(SimpleQuantifiedRole node) {
        ClassExpression newChild = node.getTailExpression().accept(this);
        if (newChild == null) {
            return null;
        } else {
            return new SimpleQuantifiedRole(node.isExists(), node.getRole(), node.isInverted(), newChild);
        }
    }

}