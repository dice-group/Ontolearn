package org.example.cel.expression.parse;

public class CEParserException extends Exception {

    private static final long serialVersionUID = 1L;

    public CEParserException() {
        super();
    }

    public CEParserException(String message) {
        super(message);
    }

    public CEParserException(Throwable cause) {
        super(cause);
    }

    public CEParserException(String message, Throwable cause) {
        super(message, cause);
    }
}
