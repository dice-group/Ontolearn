package org.example.cel.refine.suggest;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.ClassExpressionVisitingCreator;
import org.example.cel.expression.Junction;
import org.example.cel.expression.NamedClass;
import org.example.cel.expression.SimpleQuantifiedRole;

public class ClassExpressionUpdater {

    /**
     * This method creates a copy of the given class expression in which the given
     * old sub expression is replaced by the new given sub expression.
     * 
     * @param original
     * @param oldSubExpression
     * @param newSubExpression
     * @return
     */
    public static ClassExpression update(ClassExpression original, ClassExpression oldSubExpression,
            ClassExpression newSubExpression) {
        return update(original, ce -> ((ClassExpression) ce).equals(oldSubExpression), newSubExpression);
    }

    /**
     * This method creates a copy of the given class expression in which each sub
     * expression that fulfills the given check is replaced by the new given sub
     * expression.
     * 
     * @param original
     * @param check
     * @param newSubExpression
     * @return
     */
    public static ClassExpression update(ClassExpression original, Predicate<ClassExpression> check,
            ClassExpression newSubExpression) {
        ReplacingClassExpressionVisitor visitor = new ReplacingClassExpressionVisitor(check, newSubExpression);
        return visitor.createNewExpression(original);
    }

    /**
     * An implementation of the visitor pattern, that creates a copy of a given
     * class expression while at the same time searching and replacing a given sub
     * expression with another given sub expression.
     * 
     * <p>
     * Note that this implementation is <b>not thread-safe</b>!
     * </p>
     * 
     * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
     *
     */
    public static class ReplacingClassExpressionVisitor implements ClassExpressionVisitingCreator<ClassExpression> {

        protected Predicate<ClassExpression> check;
        protected ClassExpression replacement;

        public ReplacingClassExpressionVisitor(Predicate<ClassExpression> check, ClassExpression replacement) {
            super();
            this.check = check;
            this.replacement = replacement;
        }

        public ClassExpression createNewExpression(ClassExpression original) {
            if (check.test(original)) {
                return replacement.deepCopy();
            }
            return original.accept(this);
        }

        @Override
        public ClassExpression visitNamedClass(NamedClass node) {
            // there is nothing more to do. We simply create a copy of the named class and
            // return it.
            return node.deepCopy();
        }

        @Override
        public ClassExpression visitJunction(Junction node) {
            List<ClassExpression> newChildren = new ArrayList<>();
            for (ClassExpression child : node.getChildren()) {
                if (check.test(child)) {
                    newChildren.add(replacement.deepCopy());
                } else {
                    newChildren.add(child.accept(this));
                }
            }
            return new Junction(node.isConjunction(), newChildren);
        }

        @Override
        public ClassExpression visitSimpleQuantificationRole(SimpleQuantifiedRole node) {
            return new SimpleQuantifiedRole(node.isExists(), node.getRole(), node.isInverted(),
                    check.test(node.getTailExpression()) ? replacement.deepCopy()
                            : node.getTailExpression().accept(this));
        }

    }

}
