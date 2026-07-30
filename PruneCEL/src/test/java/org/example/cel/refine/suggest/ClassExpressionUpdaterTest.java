package org.example.cel.refine.suggest;

import java.util.ArrayList;
import java.util.List;

import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.NamedClass;
import org.example.cel.expression.SimpleQuantifiedRole;
import org.example.cel.refine.suggest.ClassExpressionUpdater;
import org.example.cel.refine.suggest.Suggestor;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(Parameterized.class)
public class ClassExpressionUpdaterTest {

    @Parameters
    public static List<Object[]> parameters() {
        List<Object[]> testCases = new ArrayList<>();

        testCases.add(new Object[] { new NamedClass("A"), Suggestor.CONTEXT_POSITION_MARKER, new NamedClass("B"),
                new NamedClass("A") });
        testCases.add(new Object[] { new SimpleQuantifiedRole(true, "r", false, Suggestor.CONTEXT_POSITION_MARKER),
                Suggestor.CONTEXT_POSITION_MARKER, new NamedClass("B"),
                new SimpleQuantifiedRole(true, "r", false, new NamedClass("B")) });
        testCases.add(new Object[] {
                new SimpleQuantifiedRole(true, "r1", false,
                        new SimpleQuantifiedRole(true, "r2", false, Suggestor.CONTEXT_POSITION_MARKER)),
                Suggestor.CONTEXT_POSITION_MARKER, new NamedClass("B"), new SimpleQuantifiedRole(true, "r1", false,
                        new SimpleQuantifiedRole(true, "r2", false, new NamedClass("B"))) });

        return testCases;
    }

    private ClassExpression givenExpression;
    private ClassExpression subExpression;
    private ClassExpression replacement;
    private ClassExpression expectedExpression;

    public ClassExpressionUpdaterTest(ClassExpression givenExpression, ClassExpression subExpression,
            ClassExpression replacement, ClassExpression expectedExpression) {
        super();
        this.givenExpression = givenExpression;
        this.subExpression = subExpression;
        this.replacement = replacement;
        this.expectedExpression = expectedExpression;
    }

    @Test
    public void test() {
        Assert.assertEquals(expectedExpression,
                ClassExpressionUpdater.update(givenExpression, subExpression, replacement));
    }
}
