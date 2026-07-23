package org.example.cel.expression;

import java.util.ArrayList;
import java.util.List;

import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.Junction;
import org.example.cel.expression.NamedClass;
import org.example.cel.expression.SimpleQuantifiedRole;
import org.example.cel.expression.parse.CEParserException;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(Parameterized.class)
public class ClassExpressionComparisonTest {

    protected ClassExpression ce1;
    protected ClassExpression ce2;
    protected boolean expectedResult;

    public ClassExpressionComparisonTest(ClassExpression ce1, ClassExpression ce2, boolean expectedResult) {
        super();
        this.ce1 = ce1;
        this.ce2 = ce2;
        this.expectedResult = expectedResult;
    }

    @Test
    public void test() {
        Assert.assertEquals(expectedResult, ce1.equals(ce2));
        Assert.assertEquals(expectedResult, ce2.equals(ce1));
    }

    @Parameters
    public static List<Object[]> parameters() throws CEParserException {
        List<Object[]> testCases = new ArrayList<>();

        testCases.add(new Object[] { new NamedClass("A"), new NamedClass("A"), true });
        testCases.add(new Object[] { new NamedClass("A", true), new NamedClass("A", true), true });
        testCases.add(new Object[] { new NamedClass("A", true), new NamedClass("A", false), false });
        testCases.add(new Object[] { new NamedClass("A"), new NamedClass("B"), false });
        testCases.add(new Object[] { new NamedClass("A", true), new NamedClass("B", true), false });

        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", false, NamedClass.TOP),
                new SimpleQuantifiedRole(true, "r", false, NamedClass.TOP), true });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", false, NamedClass.TOP),
                new SimpleQuantifiedRole(false, "r", false, NamedClass.TOP), false });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r1", false, NamedClass.TOP),
                new SimpleQuantifiedRole(true, "r2", false, NamedClass.TOP), false });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", false, NamedClass.TOP),
                new SimpleQuantifiedRole(true, "r", true, NamedClass.TOP), false });

        testCases.add(new Object[] { new SimpleQuantifiedRole(false, "r", false, NamedClass.BOTTOM),
                new SimpleQuantifiedRole(false, "r", false, NamedClass.BOTTOM), true });
        testCases.add(new Object[] { new SimpleQuantifiedRole(false, "r1", false, NamedClass.BOTTOM),
                new SimpleQuantifiedRole(false, "r2", false, NamedClass.BOTTOM), false });
        testCases.add(new Object[] { new SimpleQuantifiedRole(false, "r", false, NamedClass.BOTTOM),
                new SimpleQuantifiedRole(false, "r", true, NamedClass.BOTTOM), false });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", true, NamedClass.TOP),
                new SimpleQuantifiedRole(true, "r", true, NamedClass.TOP), true });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r1", true, NamedClass.TOP),
                new SimpleQuantifiedRole(true, "r2", true, NamedClass.TOP), false });
        testCases.add(new Object[] { new SimpleQuantifiedRole(false, "r", true, NamedClass.BOTTOM),
                new SimpleQuantifiedRole(false, "r", true, NamedClass.BOTTOM), true });
        testCases.add(new Object[] { new SimpleQuantifiedRole(false, "r1", true, NamedClass.BOTTOM),
                new SimpleQuantifiedRole(false, "r2", true, NamedClass.BOTTOM), false });

        testCases.add(new Object[] { new Junction(true, new NamedClass("A"), new NamedClass("B")),
                new Junction(true, new NamedClass("A"), new NamedClass("B")), true });
        testCases.add(new Object[] { new Junction(true, new NamedClass("A"), new NamedClass("B")),
                new Junction(false, new NamedClass("A"), new NamedClass("B")), false });
        testCases.add(new Object[] { new Junction(true, new NamedClass("A"), new NamedClass("B")),
                new Junction(true, new NamedClass("A")), false });
        testCases.add(new Object[] { new Junction(true, new NamedClass("A"), new NamedClass("B")),
                new Junction(true, new NamedClass("A"), new NamedClass("B"), new NamedClass("B"), new NamedClass("B")),
                true });
        testCases.add(new Object[] { new Junction(false, new NamedClass("A"), new NamedClass("B")),
                new Junction(false, new NamedClass("B"), new NamedClass("A")), true });

        return testCases;
    }
}
