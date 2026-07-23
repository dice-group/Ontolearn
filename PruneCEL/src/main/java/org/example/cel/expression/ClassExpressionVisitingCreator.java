package org.example.cel.expression;

/**
 * Interface for implementing the visitor pattern with a return type.
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 *
 */
public interface ClassExpressionVisitingCreator<T> {

    T visitNamedClass(NamedClass node);

    T visitJunction(Junction node);

    T visitSimpleQuantificationRole(SimpleQuantifiedRole node);
}
