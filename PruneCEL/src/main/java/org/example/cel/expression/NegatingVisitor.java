package org.example.cel.expression;

import java.util.Set;

public class NegatingVisitor implements ClassExpressionVisitingCreator<ClassExpression> {

    @Override
    public ClassExpression visitNamedClass(NamedClass node) {
        if (NamedClass.TOP.equals(node)) {
            return NamedClass.BOTTOM;
        }
        if (NamedClass.BOTTOM.equals(node)) {
            return NamedClass.TOP;
        }
        NamedClass negated = (NamedClass) node.deepCopy();
        negated.setNegated(!negated.isNegated());
        return negated;
    }

    @Override
    public ClassExpression visitJunction(Junction node) {
        Set<ClassExpression> children = node.getChildren();
        ClassExpression[] negated = new ClassExpression[children.size()];
        int pos = 0;
        for (ClassExpression child : children) {
            negated[pos] = child.accept(this);
            ++pos;
        }
        return new Junction(!node.isConjunction(), negated);
    }

    @Override
    public ClassExpression visitSimpleQuantificationRole(SimpleQuantifiedRole node) {
        ClassExpression negated = node.getTailExpression().accept(this);
        return new SimpleQuantifiedRole(!node.isExists(), node.getRole(), node.isInverted(), negated);
    }

    public ClassExpression negateExpression(ClassExpression ce) {
        return ce.accept(this);
    }
}