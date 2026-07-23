package org.example.cel.refine.suggest.sparql;

import java.util.ArrayList;
import java.util.List;

import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.Junction;
import org.example.cel.expression.NamedClass;
import org.example.cel.expression.SimpleQuantifiedRole;
import org.example.cel.refine.suggest.sparql.FilterNotExistsChecker;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(Parameterized.class)
public class FilterNotExistsCheckerTest {

    private ClassExpression givenExpression;
    private Boolean expectedResult;

    public FilterNotExistsCheckerTest(ClassExpression givenExpression, Boolean expectedResult) {
        super();
        this.givenExpression = givenExpression;
        this.expectedResult = expectedResult;
    }

    @Test
    public void test() {
        FilterNotExistsChecker checker = new FilterNotExistsChecker();
        if (expectedResult) {
            Assert.assertTrue(givenExpression.accept(checker));
        } else {
            Assert.assertFalse(givenExpression.accept(checker));
        }
    }

    @Parameters
    public static List<Object[]> parameters() {
        List<Object[]> testCases = new ArrayList<>();

        testCases.add(new Object[] { new NamedClass("A"), Boolean.FALSE });
        testCases.add(new Object[] { new NamedClass("A", true), Boolean.TRUE });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", false, new NamedClass("A")), Boolean.FALSE });
        testCases.add(new Object[] { new SimpleQuantifiedRole(false, "r", false, new NamedClass("A")), Boolean.TRUE });

        testCases.add(new Object[] { new Junction(true, new NamedClass("A"), new NamedClass("B")), Boolean.FALSE });
        testCases.add(
                new Object[] { new Junction(true, new NamedClass("A", true), new NamedClass("B")), Boolean.FALSE });
        testCases.add(
                new Object[] { new Junction(true, new NamedClass("A"), new NamedClass("B", true)), Boolean.FALSE });
        testCases.add(new Object[] { new Junction(true, new NamedClass("A", true), new NamedClass("B", true)),
                Boolean.TRUE });
        testCases
                .add(new Object[] {
                        new Junction(true, new NamedClass("A", true),
                                new Junction(true, new NamedClass("B", true),
                                        new SimpleQuantifiedRole(false, "r", false, new NamedClass("A")))),
                        Boolean.TRUE });

        return testCases;
    }

}
