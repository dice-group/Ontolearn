package org.example.cel.expression.parse;

public class Tokenizer {

    protected CEParser parser;

    public Tokenizer(CEParser parser) {
        super();
        this.parser = parser;
    }

    public void parseString(ParserState state) throws CEParserException {
        int startIdentifier = 0;
        String quantifier = null;
        while (state.nextChar < state.length) {
            char currentChar = state.expressionString.charAt(state.nextChar);
            switch (currentChar) {
            case '(':
                if (startIdentifier < state.nextChar) {
                    parser.foundNamedClass(state, state.expressionString.substring(startIdentifier, state.nextChar));
                }
                parser.foundOpeningParanthesis(state);
                startIdentifier = state.nextChar + 1;
                break;
            case ')':
                if (startIdentifier < state.nextChar) {
                    parser.foundNamedClass(state, state.expressionString.substring(startIdentifier, state.nextChar));
                }
                parser.foundClosingParanthesis(state);
                startIdentifier = state.nextChar + 1;
                break;
            case '∃': // falls through
            case '∀':
                quantifier = Character.toString(currentChar);
                startIdentifier = state.nextChar + 1;
                break;
            case '⊔':
                if (startIdentifier < state.nextChar) {
                    parser.foundNamedClass(state, state.expressionString.substring(startIdentifier, state.nextChar));
                }
                parser.foundJunction(state, false);
                startIdentifier = state.nextChar + 1;
                break;
            case '⊓':
                if (startIdentifier < state.nextChar) {
                    parser.foundNamedClass(state, state.expressionString.substring(startIdentifier, state.nextChar));
                }
                parser.foundJunction(state, true);
                startIdentifier = state.nextChar + 1;
                break;
            case '.': {
                // look ahead to decide whether this is a separator between a role and the next
                // expression part
                if ((quantifier != null) && newStmtFollows(state.expressionString.substring(state.nextChar + 1))) {
                    parser.foundQuantifiedRole(state, quantifier,
                            state.expressionString.substring(startIdentifier, state.nextChar));
                    quantifier = null;
                    startIdentifier = state.nextChar + 1;
                }
                break;
            }
            default: {
                // Nothing to do
                break;
            }
            }
            ++state.nextChar;
        }

        if (startIdentifier < state.nextChar) {
            parser.foundNamedClass(state, state.expressionString.substring(startIdentifier, state.nextChar));
        }
    }

    protected boolean newStmtFollows(String substring) {
        if (substring.isEmpty()) {
            return true;
        }
        switch (substring.charAt(0)) {
        case '(': // falls through
        case ')':
        case '¬':
        case '∃':
        case '∀':
        case '⊔':
        case '⊓':
        case '⊤':
        case '⊥': {
            return true;
        }
        }
        if (substring.startsWith("http:") || substring.startsWith("https:")) {
            return true;
        }
        return false;
    }
}
