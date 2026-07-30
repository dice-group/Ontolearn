package org.example.cel.expression;

import java.util.HashSet;
import java.util.Set;
import java.util.function.Predicate;

/**
 * A simple visitor that checks whether a class expression contains a statement
 * that makes it being equal to bottom like (A⊓¬A) or (∃r.⊥). In such a case, it
 * returns {@code false}. In other cases, {@code true} is returned. Note that it
 * only checks simple named concepts and not more complex expressions.
 * 
 * The implementation checks expressions recursively. If the
 * {@code #checkRecursively} flag is set to {@code false}, only the given
 * expression is checked but not it's children.
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 *
 */
public class BasicSanityCheckingVisitor implements ClassExpressionVisitingCreator<Boolean>, Predicate<ClassExpression> {

    protected boolean checkRecursively = true;

    public BasicSanityCheckingVisitor() {
        super();
    }

    public BasicSanityCheckingVisitor(boolean checkRecursively) {
        super();
        this.checkRecursively = checkRecursively;
    }

    @Override
    public boolean test(ClassExpression t) {
        return t.accept(this);
    }

    @Override
    public Boolean visitNamedClass(NamedClass node) {
        return true;
    }

    @Override
    public Boolean visitJunction(Junction node) {
        if (node.isConjunction()) {
            // This is a conjunction. check whether there are cases in which the single
            // parts of the conjunction are complements of each other
            Set<String> namedClasses = new HashSet<String>();
            Set<String> negatedNamedClasses = new HashSet<String>();
            NamedClass nc;
            for (ClassExpression child : node.getChildren()) {
                if (child instanceof NamedClass) {
                    nc = (NamedClass) child;
                    if (nc.isNegated()) {
                        // The given class is negated. Check whether we saw it before in it's normal
                        // form
                        if (namedClasses.contains(nc.getName())) {
                            // We found (A⊓¬A)
                            return false;
                        }
                        negatedNamedClasses.add(nc.getName());
                    } else {
                        // The given class is not negated. Check whether we saw it before in it's
                        // negated form
                        if (negatedNamedClasses.contains(nc.getName())) {
                            // We found (A⊓¬A)
                            return false;
                        }
                        namedClasses.add(nc.getName());
                    }
                }
            }
            if (checkRecursively) {
                // Whatever junction, check the children
                for (ClassExpression child : node.getChildren()) {
                    if (!child.accept(this)) {
                        return false;
                    }
                }
            }
        } else {
            if (checkRecursively) {
                boolean result = false;
                // Whatever junction, check the children
                for (ClassExpression child : node.getChildren()) {
                    if (child.accept(this)) {
                        result = true;
                    }
                }
                return result;
            }
        }
        return true;
    }

    @Override
    public Boolean visitSimpleQuantificationRole(SimpleQuantifiedRole node) {
        // ∃r.⊥
        if (node.isExists() && (NamedClass.BOTTOM.equals(node.getTailExpression()))) {
            return false;
        }
        if (checkRecursively) {
            return node.getTailExpression().accept(this);
        } else {
            return true;
        }
    }

}
