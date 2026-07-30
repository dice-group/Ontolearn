package org.example.cel.score;

import java.util.ArrayList;
import java.util.List;

import org.example.cel.score.AccuracyCalculator;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(Parameterized.class)
public class AccuracyCalculatorTest {

    @Parameters
    public static List<Object[]> parameters() {
        List<Object[]> testCases = new ArrayList<>();

        // Basic examples
        testCases.add(new Object[] { 1, 1, 1, 0, 1.0 });
        testCases.add(new Object[] { 1, 1, 1, 1, 0.5 });
        testCases.add(new Object[] { 1, 1, 0, 1, 0.0 });
        testCases.add(new Object[] { 1, 1, 0, 0, 0.5 });

        testCases.add(new Object[] { 10, 20, 10, 20, 1.0 / 3.0 });
        testCases.add(new Object[] { 10, 20, 5, 20, 1.0 / 6.0 });
        testCases.add(new Object[] { 10, 20, 5, 10, 0.5 });

        return testCases;
    }

    private int numPosExamples;
    private int numNegExamples;
    private int numPosSelected;
    private int numNegSelected;
    private double expected;

    public AccuracyCalculatorTest(int numPosExamples, int numNegExamples, int numPosSelected, int numNegSelected,
            double expected) {
        super();
        this.numPosExamples = numPosExamples;
        this.numNegExamples = numNegExamples;
        this.numPosSelected = numPosSelected;
        this.numNegSelected = numNegSelected;
        this.expected = expected;
    }

    @Test
    public void test() {
        Assert.assertTrue("Faulty test case. There are more positive selected than positive examples exist.",
                numPosSelected <= numPosExamples);
        Assert.assertTrue("Faulty test case. There are more negative selected than negative examples exist.",
                numNegSelected <= numNegExamples);
        AccuracyCalculator calculator = new AccuracyCalculator(numPosExamples, numNegExamples);
        Assert.assertEquals(expected, calculator.calculateClassificationScore(numPosSelected, numNegSelected), 0.00001);
    }
}
