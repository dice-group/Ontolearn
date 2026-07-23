package org.example.cel.expression;

/**
 * Interface for implementing the visitor pattern.
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 *
 */
public interface ClassExpressionVisitor {

    void visitNamedClass(NamedClass node);

    void visitJunction(Junction node);

    void visitSimpleQuantificationRole(SimpleQuantifiedRole node);
}
