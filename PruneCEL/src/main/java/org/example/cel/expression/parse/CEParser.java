package org.example.cel.expression.parse;

import java.util.Iterator;

import org.apache.commons.lang3.NotImplementedException;
import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.ClassExpressionVisitingCreator;
import org.example.cel.expression.ClassExpressionVisitor;
import org.example.cel.expression.Junction;
import org.example.cel.expression.NamedClass;
import org.example.cel.expression.SimpleQuantifiedRole;

/**
 * 
 * <b>WARNING</b>: This parser cannot parse {@code ∃r.A} correctly since the
 * tokenizer has no chance to identify the dot as separator of the names (which
 * could also be IRIs). Hence, it will be parser as a single role name
 * {@code r.A}.
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 *
 */
public class CEParser {

    public ClassExpression parse(String expressionString) throws CEParserException {
        ParserState state = new ParserState(expressionString);
        Tokenizer tokenizer = new Tokenizer(this);
        tokenizer.parseString(state);
        if (state.stack.isEmpty()) {
            return null;
        }
        if (state.stack.size() > 1) {
            throw new CEParserException("Didn't consume all content from the stack: " + state.stack.toString());
        }
        return state.stack.pollLast();
    }

    protected void foundOpeningParanthesis(ParserState state) {
        state.stack.add(new Parenthesis());
    }

    protected void foundClosingParanthesis(ParserState state) throws CEParserException {
        // We found a closing parenthesis... search for it and delete it...
        Iterator<ClassExpression> iter = state.stack.descendingIterator();
        while (iter.hasNext()) {
            if (iter.next() instanceof Parenthesis) {
                iter.remove();
                reduceStackIfPossible(state);
                return;
            }
        }
        throw new CEParserException(
                "Found closing parenthesis for which there were no open parenthesis in the expression: \""
                        + state.expressionString.substring(0, state.nextChar) + "^"
                        + state.expressionString.substring(state.nextChar) + "\".");
    }

    public void foundQuantifiedRole(ParserState state, String quantifier, String role) {
        if (role.endsWith("-")) {
            state.stack.add(
                    new SimpleQuantifiedRole("∃".equals(quantifier), role.substring(0, role.length() - 1), true, null));
        } else {
            state.stack.add(new SimpleQuantifiedRole("∃".equals(quantifier), role, false, null));
        }
    }

    public void foundNamedClass(ParserState state, String className) {
        if (className.startsWith("¬")) {
            state.stack.add(new NamedClass(className.substring(1), true));
        } else {
            state.stack.add(new NamedClass(className));
        }
        reduceStackIfPossible(state);
    }

    public void foundJunction(ParserState state, boolean isConjunction) {
        // reduceStackIfPossible(state);
        Junction junction = new Junction(isConjunction);
        junction.getChildren().add(state.stack.pollLast());
        state.stack.add(junction);
    }

    private void reduceStackIfPossible(ParserState state) {
        ClassExpression last = state.stack.pollLast();
        ClassExpression before = state.stack.pollLast();
        boolean consumed = true;
        while (consumed && (before != null) && (last != null) && (!(before instanceof Parenthesis))) {
            if (before instanceof Junction) {
                ((Junction) before).getChildren().add(last);
                consumed = true;
            }
            if (before instanceof SimpleQuantifiedRole) {
                ((SimpleQuantifiedRole) before).setTailExpression(last);
                consumed = true;
            }
            if (consumed) {
                last = before;
                before = state.stack.pollLast();
            }
        }
        if (before != null) {
            state.stack.add(before);
        }
        if (last != null) {
            state.stack.add(last);
        }
    }

    protected static class Parenthesis implements ClassExpression {

        @Override
        public void toString(StringBuilder builder) {
            builder.append("TEMPORARY_PARENTHESIS_OPEN");
        }

        @Override
        public ClassExpression deepCopy() {
            return this;
        }

        @Override
        public void accept(ClassExpressionVisitor visitor) {
            throw new NotImplementedException("This node should never have been visited!");
        }

        @Override
        public <T> T accept(ClassExpressionVisitingCreator<T> visitor) {
            throw new NotImplementedException("This node should never have been visited!");
        }

    }
}
