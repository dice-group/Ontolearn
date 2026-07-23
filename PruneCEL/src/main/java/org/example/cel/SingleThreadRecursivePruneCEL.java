package org.example.cel;

import java.io.OutputStream;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.stream.Stream;

import org.apache.commons.collections.SetUtils;
import org.dice_research.topicmodeling.commons.collections.TopDoubleObjectCollection;
import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.Junction;
import org.example.cel.expression.NamedClass;
import org.example.cel.expression.ScoredClassExpression;
import org.example.cel.io.IntermediateResultPrinter;
import org.example.cel.io.NullIntermediateResultPrinter;
import org.example.cel.recursive.Solution;
import org.example.cel.recursive.SolutionComparatorForRefinement;
import org.example.cel.recursive.SubProblem;
import org.example.cel.refine.RefinementOperator;
import org.example.cel.refine.SuggestorBasedRefinementOperator;
import org.example.cel.refine.suggest.ExtendedSuggestor;
import org.example.cel.score.ScoreCalculator;
import org.example.cel.score.ScoreCalculatorFactory;
import org.example.cel.sparql.InstanceRetriever;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 * 
 * @deprecated This attempt is way too complicated. Use {@link SimpleRecursivePruneCEL} instead.
 *
 */
@Deprecated
public class SingleThreadRecursivePruneCEL extends PruneCEL {

    private static final Logger LOGGER = LoggerFactory.getLogger(MultithreadRecursivePruneCEL.class);

    protected static final int EXCLUSIVE_ITERATIONS_FOR_NEW_QUEUE = 25;

    public SingleThreadRecursivePruneCEL(ExtendedSuggestor suggestor, DescriptionLogic logic,
            ScoreCalculatorFactory calculatorFactory, InstanceRetriever retriever) {
        super(suggestor, logic, calculatorFactory);
        this.retriever = retriever;
    }

    public List<ScoredClassExpression> findClassExpression(Collection<String> positive, Collection<String> negative,
            OutputStream logStream, IntermediateResultPrinter iResultPrinter, long startTime, long timeToStop) {
        if (iResultPrinter == null) {
            iResultPrinter = new NullIntermediateResultPrinter();
        }
        TopDoubleObjectCollection<ScoredClassExpression> topExpressions = new TopDoubleObjectCollection<>(10, false);
        Set<ScoredClassExpression> seenExpressions = new HashSet<>();
        StackedQueue mainQueue = new StackedQueue(new PriorityQueue<Solution>(new SolutionComparatorForRefinement()),
                0);
        StackedQueue currentQueue = mainQueue;
        ScoreCalculator mainScoreCalculator = calculatorFactory.create(positive.size(), negative.size());
        double perfectScore = mainScoreCalculator.getPerfectScore();

        RefinementOperator rho = new SuggestorBasedRefinementOperator(suggestor, logic, mainScoreCalculator, positive,
                negative);
        ((SuggestorBasedRefinementOperator) rho).setLogStream(logStream);

        Set<String> subProblemKeys = new HashSet<>();
        Deque<StackedQueue> queueStack = new ArrayDeque<>();
        queueStack.add(mainQueue);

        SubProblem mainProblem = new SubProblem(null, positive, negative, mainScoreCalculator, rho);
        Collection<ScoredClassExpression> newExpressions;
        ScoredClassExpression nextBestExpression;
        ScoredClassExpression mergedExpression;
        // Create Top expression
        nextBestExpression = mainScoreCalculator.score(NamedClass.TOP, positive.size(), negative.size(), false);
        Solution currentSolution = new Solution(mainProblem, nextBestExpression);
        currentQueue.queue.add(currentSolution);
        topExpressions.add(nextBestExpression.getClassificationScore(), nextBestExpression);
        iResultPrinter.printIntermediateResults(topExpressions);
        int iterationCount = 0;
        boolean foundPerfectScore = false;
        // Iterate over the queue while
        while (// 1. the queue is not empty (and there are no other queues left)
        (!currentQueue.queue.isEmpty() || (queueStack.size() > 1)) &&
        // 2. the best expression has not the perfect score
                (!foundPerfectScore) &&
                // 3. We haven't reached the maximum number of iterations (if defined)
                (maxIterations == 0 || iterationCount < maxIterations) &&
                // 4. We haven't reached the maximum amount of time that we are allowed to use
                (maxTime == 0 || (System.currentTimeMillis() < timeToStop))) {
            // Ensure that we are working on the right queue
            currentQueue = getCurrentQueue(mainQueue, queueStack);
            // If the queue is empty at this point, we have to stop
            if (currentQueue.queue.isEmpty()) {
                // Jump out of the loop
                continue;
            }
            // Get next solution from queue (and decrease exclusive queue counter)
            currentSolution = currentQueue.queue.poll();
            --currentQueue.remainingExclusiveIterations;
            mergedExpression = currentSolution.getMergedSolution();
            LOGGER.debug("Refining rScore={}, cScore={}, ce={}, sce={}", mergedExpression.getRefinementScore(),
                    mergedExpression.getClassificationScore(), mergedExpression.getClassExpression(),
                    currentSolution.getPartialSolution().getClassExpression());
            // Refine this expression
            newExpressions = currentSolution.refine();
            // Check the expressions
            for (ScoredClassExpression newExpression : newExpressions) {
                // Get the merged expression (in case we have looked into a sub problem
                currentSolution = getSolution(newExpression, currentSolution.getSubProblem(), mainScoreCalculator);
                mergedExpression = currentSolution.getMergedSolution();
                // If 1) we haven't seen this before AND 2a) we are configured to not further
                // look into this OR 2b) the classification score of this
                // new expression is better than the expression that we refined OR 2c) we
                // extended the previous expression by adding an edge
                if (seenExpressions.add(mergedExpression) && (!skipNonImprovingStmts
                        || (nextBestExpression.getClassificationScore() < mergedExpression.getClassificationScore())
                        || (mergedExpression.isAddedEdge()))) {
                    currentQueue.queue.add(currentSolution);
                    if (topExpressions.add(mergedExpression.getClassificationScore(), mergedExpression)) {
                        iResultPrinter.printIntermediateResults(topExpressions);
                        foundPerfectScore = foundPerfectScore
                                || mergedExpression.getClassificationScore() >= perfectScore;
                    }
                    // If the new expression is very precise (and there is no perfect solution yet)
                    if ((getPrecision(newExpression) >= precisionThreshold) && (newExpression.getPosCount() > 1)
                            && (newExpression.getPosCount() < (positive.size() - 1)) && !foundPerfectScore) {
                        // Start sub task if possible
                        // 1. Figure out the remaining positives
                        Set<String> remainingPositives = new HashSet<String>(positive);
                        @SuppressWarnings("unchecked")
                        Set<String> selectedInstances = retriever
                                .retrieveInstances(mergedExpression.getClassExpression(), positive, SetUtils.EMPTY_SET);
                        remainingPositives.removeAll(selectedInstances);
                        // 2. Generate a key from it
                        String remainingPosKey = generateKey(remainingPositives);
                        // If we do not already know this key, we have identified a new sub problem
                        if (subProblemKeys.add(remainingPosKey)) {
                            LOGGER.debug("Removing positives selected by {}", newExpression.getClassExpression());
                            LOGGER.debug(
                                    "Trying to solve the remaining sub problem recursively for the remaining {} positives ...",
                                    remainingPositives.size());
                            // Create a new sub problem
                            ScoreCalculator subScoreCalculator = calculatorFactory.create(remainingPositives.size(),
                                    negative.size());
                            RefinementOperator subRho = new SuggestorBasedRefinementOperator(suggestor, logic,
                                    subScoreCalculator, remainingPositives, negative);
                            SubProblem subProblem = new SubProblem(mergedExpression, remainingPositives, negative,
                                    subScoreCalculator, subRho);
                            Solution subSolution = new Solution(subProblem,
                                    subScoreCalculator.score(NamedClass.TOP, remainingPositives.size(), negative.size(),
                                            false),
                                    mainScoreCalculator.score(
                                            merge(mergedExpression.getClassExpression(), NamedClass.TOP),
                                            positive.size(), negative.size(), false));
                            // Create an exclusive queue for this sub problem
                            StackedQueue newQueue = new StackedQueue(
                                    new PriorityQueue<Solution>(new SolutionComparatorForRefinement()),
                                    EXCLUSIVE_ITERATIONS_FOR_NEW_QUEUE);
                            newQueue.queue.add(subSolution);
                            queueStack.add(newQueue);
                        }
                    } // If very precise
                }
            }
            iterationCount++;
        }
        LOGGER.info("Stopping search. Saw {} expressions in {} iterations.", seenExpressions.size(), iterationCount);
        return Stream.of(topExpressions.getObjects()).map(o -> (ScoredClassExpression) o).toList();
    }

    protected StackedQueue getCurrentQueue(StackedQueue mainQueue, Deque<StackedQueue> queueStack) {
        StackedQueue currentQueue = queueStack.getLast();
        // Ensure that we are working on the right queue
        while (currentQueue.queue.isEmpty()
                || ((queueStack.size() > 1) && currentQueue.remainingExclusiveIterations <= 0)) {
            LOGGER.info("Removing a queue");
            // Get the last queue that has been added to the stack
            queueStack.removeLast();
            mainQueue.queue.addAll(currentQueue.queue);
            currentQueue = queueStack.getLast();
        }
        return currentQueue;
    }

    protected Solution getSolution(ScoredClassExpression newExpression, SubProblem subProblem,
            ScoreCalculator mainScoreCalculator) {
        // if we work on the main problem, return the give expression
        if (subProblem.getParentExpression() == null) {
            return new Solution(subProblem, newExpression, newExpression);
        } else {
            ScoredClassExpression parentExpression = subProblem.getParentExpression();
            // We have to merge the parent expression and the new expression
            ClassExpression mergedExpression = merge(parentExpression.getClassExpression(),
                    newExpression.getClassExpression());
            ScoredClassExpression scoredMergedExp = mainScoreCalculator.score(mergedExpression,
                    newExpression.getPosCount() + parentExpression.getPosCount(),
                    newExpression.getNegCount() + parentExpression.getNegCount(), newExpression.isAddedEdge());
            // If the refinement score is set to 0, we should keep it like that
            if (newExpression.getRefinementScore() == 0.0) {
                scoredMergedExp.setRefinementScore(0.0);
            }
            return new Solution(subProblem, newExpression, scoredMergedExp);
        }
    }

    protected static ClassExpression merge(ClassExpression ex1, ClassExpression ex2) {
        List<ClassExpression> children = new ArrayList<>();
        // If ex1 is a disjunction
        if ((ex1 instanceof Junction) && !((Junction) ex1).isConjunction()) {
            children.addAll(((Junction) ex1).getChildren());
        } else {
            children.add(ex1);
        }
        // If ex2 is a disjunction
        if ((ex2 instanceof Junction) && !((Junction) ex2).isConjunction()) {
            children.addAll(((Junction) ex2).getChildren());
        } else {
            children.add(ex2);
        }
        return new Junction(false, children);
    }

    public static String generateKey(Set<String> remainingPositives) {
        return remainingPositives.stream().sorted()
                .collect(StringBuilder::new, StringBuilder::append, StringBuilder::append).toString();
    }

    public static class StackedQueue {
        public Queue<Solution> queue;
        public int remainingExclusiveIterations;

        public StackedQueue(Queue<Solution> queue, int remainingExclusiveIterations) {
            super();
            this.queue = queue;
            this.remainingExclusiveIterations = remainingExclusiveIterations;
        }

    }

}
