package org.example.cel.expression;

import java.util.Objects;

public class NamedClass implements ClassExpression {

    public static final NamedClass TOP = new NamedClass("⊤");
    public static final NamedClass BOTTOM = new NamedClass("⊥");

    protected String name;
    protected boolean negated;

    public NamedClass(String name) {
        this(name, false);
    }

    public NamedClass(String name, boolean negated) {
        super();
        Objects.nonNull(name);
        if (name.isEmpty()) {
            throw new IllegalArgumentException("Got an empty name for a named class.");
        }
        this.name = name;
        this.negated = negated;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((name == null) ? 0 : name.hashCode());
        result = prime * result + (negated ? 1231 : 1237);
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
        NamedClass other = (NamedClass) obj;
        if (name == null) {
            if (other.name != null)
                return false;
        } else if (!name.equals(other.name))
            return false;
        if (negated != other.negated)
            return false;
        return true;
    }

    @Override
    public String toString() {
        if (isNegated()) {
            return '¬' + name;
        } else {
            return name;
        }
    }

    @Override
    public void toString(StringBuilder builder) {
        if (isNegated()) {
            builder.append('¬');
        }
        builder.append(name);
    }

    @Override
    public ClassExpression deepCopy() {
        return new NamedClass(this.name, this.negated);
    }

    @Override
    public void accept(ClassExpressionVisitor visitor) {
        visitor.visitNamedClass(this);
    }

    @Override
    public <T> T accept(ClassExpressionVisitingCreator<T> visitor) {
        return visitor.visitNamedClass(this);
    }

    /**
     * @return the name
     */
    public String getName() {
        return name;
    }

    /**
     * @param name the name to set
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * @return the negated
     */
    public boolean isNegated() {
        return negated;
    }

    /**
     * @param negated the negated to set
     */
    public void setNegated(boolean negated) {
        this.negated = negated;
    }
}
