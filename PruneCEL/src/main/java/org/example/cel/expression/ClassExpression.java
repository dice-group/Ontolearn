package org.example.cel.expression;

public interface ClassExpression {

    void toString(StringBuilder builder);

    ClassExpression deepCopy();

    void accept(ClassExpressionVisitor visitor);

    <T> T accept(ClassExpressionVisitingCreator<T> visitor);
}
