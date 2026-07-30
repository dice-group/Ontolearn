package org.example.cel.expression;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.collections4.CollectionUtils;

public class Junction implements ClassExpression, Iterable<ClassExpression> {

    private boolean isConjunction;
    private Set<ClassExpression> children;

    public Junction(boolean isConjunction, Collection<ClassExpression> children) {
        super();
        this.isConjunction = isConjunction;
        this.children = new HashSet<>(children);
    }

    public Junction(boolean isConjunction, ClassExpression... children) {
        super();
        this.isConjunction = isConjunction;
        this.children = new HashSet<>(Arrays.asList(children));
    }

    /**
     * @return the isConjunction
     */
    public boolean isConjunction() {
        return isConjunction;
    }

    /**
     * @param isConjunction the isConjunction to set
     */
    public void setConjunction(boolean isConjunction) {
        this.isConjunction = isConjunction;
    }

    /**
     * @return the children
     */
    public Set<ClassExpression> getChildren() {
        return children;
    }

    /**
     * @param children the children to set
     */
    public void setChildren(Set<ClassExpression> children) {
        this.children = children;
    }

    @Override
    public void toString(StringBuilder builder) {
        char junction;
        if (isConjunction) {
            junction = '⊓';
        } else {
            junction = '⊔';
        }
        boolean first = true;
        builder.append('(');
        for (ClassExpression child : children) {
            if (first) {
                first = false;
            } else {
                builder.append(junction);
            }
            child.toString(builder);
        }
        builder.append(')');
    }

    @Override
    public Iterator<ClassExpression> iterator() {
        return children.iterator();
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((children == null) ? 0 : children.hashCode());
        result = prime * result + (isConjunction ? 1231 : 1237);
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Junction other = (Junction) obj;
        if (children == null) {
            if (other.children != null)
                return false;
        } else if (!CollectionUtils.isEqualCollection(children, other.children))
            return false;
        if (isConjunction != other.isConjunction)
            return false;
        return true;
    }

    @Override
    public ClassExpression deepCopy() {
        return new Junction(this.isConjunction,
                this.children.stream().map(c -> c.deepCopy()).collect(Collectors.toSet()));
    }

    @Override
    public void accept(ClassExpressionVisitor visitor) {
        visitor.visitJunction(this);
    }

    @Override
    public <T> T accept(ClassExpressionVisitingCreator<T> visitor) {
        return visitor.visitJunction(this);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        this.toString(builder);
        return builder.toString();
    }
}
