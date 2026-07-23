package org.example.cel.expression.parse;

import java.util.ArrayList;
import java.util.List;

import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.Junction;
import org.example.cel.expression.NamedClass;
import org.example.cel.expression.SimpleQuantifiedRole;
import org.example.cel.expression.parse.CEParser;
import org.example.cel.expression.parse.CEParserException;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(Parameterized.class)
public class CEParserTest {

    @Parameters
    public static List<Object[]> parameters() {
        List<Object[]> testCases = new ArrayList<>();

        // A
        testCases.add(new Object[] { new NamedClass("A", false) });
        testCases.add(new Object[] { new NamedClass("http://example.org/A", false) });

        // not A
        testCases.add(new Object[] { new NamedClass("A", true) });
        testCases.add(new Object[] { new NamedClass("http://example.org/A", true) });

        // Exists r TOP
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", false, NamedClass.TOP) });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", true, NamedClass.TOP) });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "http://example.org/r", false, NamedClass.TOP) });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "http://example.org/r", true, NamedClass.TOP) });

        // Exists r A
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", false, new NamedClass("http://example.org/A", true)) });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", true, new NamedClass("http://example.org/A", true)) });
        testCases.add(new Object[] {
                new SimpleQuantifiedRole(true, "http://example.org/r", false, new NamedClass("http://example.org/A", true)) });
        testCases.add(new Object[] {
                new SimpleQuantifiedRole(true, "http://example.org/r", true, new NamedClass("http://example.org/A", true)) });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", false, new NamedClass("http://example.org/A", false)) });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", true, new NamedClass("http://example.org/A", false)) });
        testCases.add(new Object[] {
                new SimpleQuantifiedRole(true, "http://example.org/r", false, new NamedClass("http://example.org/A", false)) });
        testCases.add(new Object[] {
                new SimpleQuantifiedRole(true, "http://example.org/r", true, new NamedClass("http://example.org/A", false)) });

        // Forall r BOTTOM
        testCases.add(new Object[] { new SimpleQuantifiedRole(false, "r", false, NamedClass.BOTTOM) });
        testCases.add(new Object[] { new SimpleQuantifiedRole(false, "r", true, NamedClass.BOTTOM) });
        testCases.add(
                new Object[] { new SimpleQuantifiedRole(false, "http://example.org/r", false, NamedClass.BOTTOM) });
        testCases
                .add(new Object[] { new SimpleQuantifiedRole(false, "http://example.org/r", true, NamedClass.BOTTOM) });

        // Conjunction of A and B with changing negations
        testCases.add(new Object[] { new Junction(true, new NamedClass("A", false), new NamedClass("B", false)) });
        testCases.add(new Object[] { new Junction(true, new NamedClass("A", true), new NamedClass("B", false)) });
        testCases.add(new Object[] { new Junction(true, new NamedClass("A", false), new NamedClass("B", true)) });
        testCases.add(new Object[] { new Junction(true, new NamedClass("A", true), new NamedClass("B", true)) });
        testCases.add(new Object[] { new Junction(true, new NamedClass("http://example.org/A", false),
                new NamedClass("http://example.org/B", false)) });
        testCases.add(new Object[] { new Junction(true, new NamedClass("http://example.org/A", true),
                new NamedClass("http://example.org/B", false)) });
        testCases.add(new Object[] { new Junction(true, new NamedClass("http://example.org/A", false),
                new NamedClass("http://example.org/B", true)) });
        testCases.add(new Object[] { new Junction(true, new NamedClass("http://example.org/A", true),
                new NamedClass("http://example.org/B", true)) });

        // Disjunction of A and B with changing negations
        testCases.add(new Object[] { new Junction(false, new NamedClass("A", false), new NamedClass("B", false)) });
        testCases.add(new Object[] { new Junction(false, new NamedClass("A", true), new NamedClass("B", false)) });
        testCases.add(new Object[] { new Junction(false, new NamedClass("A", false), new NamedClass("B", true)) });
        testCases.add(new Object[] { new Junction(false, new NamedClass("A", true), new NamedClass("B", true)) });
        testCases.add(new Object[] { new Junction(false, new NamedClass("http://example.org/A", false),
                new NamedClass("http://example.org/B", false)) });
        testCases.add(new Object[] { new Junction(false, new NamedClass("http://example.org/A", true),
                new NamedClass("http://example.org/B", false)) });
        testCases.add(new Object[] { new Junction(false, new NamedClass("http://example.org/A", false),
                new NamedClass("http://example.org/B", true)) });
        testCases.add(new Object[] { new Junction(false, new NamedClass("http://example.org/A", true),
                new NamedClass("http://example.org/B", true)) });

        // (∃r1.∃r2.⊤⊔∃r1.(∃r3.⊤⊓A)
        testCases.add(new Object[] { new Junction(false,
                new SimpleQuantifiedRole(true, "r1", false,
                        new SimpleQuantifiedRole(true, "r2", false, NamedClass.TOP)),
                new SimpleQuantifiedRole(true, "r1", false, new Junction(true,
                        new SimpleQuantifiedRole(true, "r3", false, NamedClass.TOP), new NamedClass("B", false)))) });

        return testCases;
    }

    private ClassExpression ce;

    public CEParserTest(ClassExpression ce) {
        this.ce = ce;
    }

    @Test
    public void test() throws CEParserException {
        String serialized = ce.toString();

        CEParser parser = new CEParser();
        ClassExpression parsedCE = parser.parse(serialized);

        Assert.assertEquals(ce, parsedCE);
    }
}
