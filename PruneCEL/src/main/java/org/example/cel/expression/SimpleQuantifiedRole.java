package org.example.cel.expression;

/**
 * Role with a "simple" quantifier (i.e., ∃ or ∀).
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 *
 */
public class SimpleQuantifiedRole implements ClassExpression {

    private boolean isExists;
    private String role;
    private boolean isInverted;
    private ClassExpression tailExpression;

    public SimpleQuantifiedRole(boolean isExists, String role, boolean isInverted, ClassExpression tailExpression) {
        super();
        this.isExists = isExists;
        this.role = role;
        this.isInverted = isInverted;
        this.tailExpression = tailExpression;
    }

    /**
     * @return the isExists
     */
    public boolean isExists() {
        return isExists;
    }

    /**
     * @param isExists the isExists to set
     */
    public void setExists(boolean isExists) {
        this.isExists = isExists;
    }

    /**
     * @return the tailExpression
     */
    public ClassExpression getTailExpression() {
        return tailExpression;
    }

    /**
     * @param tailExpression the tailExpression to set
     */
    public void setTailExpression(ClassExpression tailExpression) {
        this.tailExpression = tailExpression;
    }

    /**
     * @return the role
     */
    public String getRole() {
        return role;
    }

    /**
     * @param role the role to set
     */
    public void setRole(String role) {
        this.role = role;
    }

    /**
     * @return the isInverted
     */
    public boolean isInverted() {
        return isInverted;
    }

    /**
     * @param isInverted the isInverted to set
     */
    public void setInverted(boolean isInverted) {
        this.isInverted = isInverted;
    }

    @Override
    public void toString(StringBuilder builder) {
        if (isExists) {
            builder.append('∃');
        } else {
            builder.append('∀');
        }
        builder.append(role);
        if (isInverted) {
            builder.append('-');
        }
        builder.append('.');
        if (tailExpression != null) {
            tailExpression.toString(builder);
        } else {
            builder.append("null");
        }
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + (isExists ? 1231 : 1237);
        result = prime * result + (isInverted ? 1231 : 1237);
        result = prime * result + ((role == null) ? 0 : role.hashCode());
        result = prime * result + ((tailExpression == null) ? 0 : tailExpression.hashCode());
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
        SimpleQuantifiedRole other = (SimpleQuantifiedRole) obj;
        if (isExists != other.isExists)
            return false;
        if (isInverted != other.isInverted)
            return false;
        if (role == null) {
            if (other.role != null)
                return false;
        } else if (!role.equals(other.role))
            return false;
        if (tailExpression == null) {
            if (other.tailExpression != null)
                return false;
        } else if (!tailExpression.equals(other.tailExpression))
            return false;
        return true;
    }

    @Override
    public ClassExpression deepCopy() {
        return new SimpleQuantifiedRole(isExists, role, isInverted, tailExpression.deepCopy());
    }

    @Override
    public void accept(ClassExpressionVisitor visitor) {
        visitor.visitSimpleQuantificationRole(this);
    }

    @Override
    public <T> T accept(ClassExpressionVisitingCreator<T> visitor) {
        return visitor.visitSimpleQuantificationRole(this);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        this.toString(builder);
        return builder.toString();
    }

}
