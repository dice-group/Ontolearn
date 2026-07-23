package org.example.cel.expression;

import java.util.ArrayList;
import java.util.List;

import org.example.cel.expression.BasicSanityCheckingVisitor;
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
public class BasicSanityCheckingVisitorTest {

    protected ClassExpression ce;
    protected boolean checkRecursively;
    protected boolean expectedResult;

    public BasicSanityCheckingVisitorTest(ClassExpression ce, boolean checkRecursively, boolean expectedResult) {
        super();
        this.ce = ce;
        this.checkRecursively = checkRecursively;
        this.expectedResult = expectedResult;
    }

    @Test
    public void test() {
        if (expectedResult) {
            Assert.assertTrue("Expected the check of " + ce.toString() + " to pass.",
                    ce.accept(new BasicSanityCheckingVisitor(checkRecursively)));
        } else {
            Assert.assertFalse("Expected the check of " + ce.toString() + " to fail.",
                    ce.accept(new BasicSanityCheckingVisitor(checkRecursively)));
        }
    }

    @Parameters
    public static List<Object[]> parameters() throws CEParserException {
        List<Object[]> testCases = new ArrayList<>();

        testCases.add(new Object[] { new NamedClass("A"), true, true });
        testCases.add(new Object[] { new NamedClass("A"), false, true });
        testCases.add(new Object[] { new NamedClass("A", true), true, true });
        testCases.add(new Object[] { new NamedClass("A", true), false, true });

        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", false, NamedClass.TOP), true, true });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", false, NamedClass.TOP), false, true });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", false, NamedClass.BOTTOM), true, false });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", false, NamedClass.BOTTOM), false, false });
        testCases.add(new Object[] { new SimpleQuantifiedRole(false, "r", false, NamedClass.TOP), true, true });
        testCases.add(new Object[] { new SimpleQuantifiedRole(false, "r", false, NamedClass.TOP), false, true });
        testCases.add(new Object[] { new SimpleQuantifiedRole(false, "r", false, NamedClass.BOTTOM), true, true });
        testCases.add(new Object[] { new SimpleQuantifiedRole(false, "r", false, NamedClass.BOTTOM), false, true });

        testCases.add(new Object[] { new Junction(true, new NamedClass("A"), new NamedClass("B")), true, true });
        testCases.add(new Object[] { new Junction(true, new NamedClass("A"), new NamedClass("B")), false, true });
        testCases.add(new Object[] { new Junction(true, new NamedClass("A"), new NamedClass("B", true)), true, true });
        testCases.add(new Object[] { new Junction(true, new NamedClass("A"), new NamedClass("B", true)), false, true });
        testCases.add(new Object[] { new Junction(true, new NamedClass("A", true), new NamedClass("B")), true, true });
        testCases.add(new Object[] { new Junction(true, new NamedClass("A", true), new NamedClass("B")), false, true });
        testCases.add(new Object[] { new Junction(true, new NamedClass("A"), new NamedClass("A", true)), true, false });
        testCases
                .add(new Object[] { new Junction(true, new NamedClass("A"), new NamedClass("A", true)), false, false });
        testCases.add(new Object[] { new Junction(true, new NamedClass("A", true), new NamedClass("A")), true, false });
        testCases
                .add(new Object[] { new Junction(true, new NamedClass("A", true), new NamedClass("A")), false, false });

        testCases.add(new Object[] { new Junction(false, new NamedClass("A"), new NamedClass("B")), true, true });
        testCases.add(new Object[] { new Junction(false, new NamedClass("A"), new NamedClass("B")), false, true });
        testCases.add(new Object[] { new Junction(false, new NamedClass("A"), new NamedClass("B", true)), true, true });
        testCases
                .add(new Object[] { new Junction(false, new NamedClass("A"), new NamedClass("B", true)), false, true });
        testCases.add(new Object[] { new Junction(false, new NamedClass("A", true), new NamedClass("B")), true, true });
        testCases
                .add(new Object[] { new Junction(false, new NamedClass("A", true), new NamedClass("B")), false, true });

        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", false,
                new Junction(true, new NamedClass("A", true), new NamedClass("A"))), true, false });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", false,
                new Junction(true, new NamedClass("A", true), new NamedClass("A"))), false, true });

        return testCases;
    }
}
