package org.example.cel.refine;

import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.collections.SetUtils;
import org.apache.commons.io.IOUtils;
import org.example.cel.DescriptionLogic;
import org.example.cel.expression.ClassExpression;
import org.example.cel.expression.ClassExpressionVisitor;
import org.example.cel.expression.Junction;
import org.example.cel.expression.NamedClass;
import org.example.cel.expression.NegatingVisitor;
import org.example.cel.expression.ScoredClassExpression;
import org.example.cel.expression.SimpleQuantifiedRole;
import org.example.cel.refine.suggest.ClassExpressionUpdater;
import org.example.cel.refine.suggest.ExtendedSuggestor;
import org.example.cel.refine.suggest.ScoredIRI;
import org.example.cel.refine.suggest.SelectionScores;
import org.example.cel.refine.suggest.Suggestor;
import org.example.cel.score.ScoreCalculator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This is an implementation of a refinement operator that is based on an
 * {@link ExtendedSuggestor} to retrieve candidates for refinements.
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 *
 */
public class SuggestorBasedRefinementOperator implements RefinementOperator {

    private static final Logger LOGGER = LoggerFactory.getLogger(SuggestorBasedRefinementOperator.class);

    /**
     * The suggestor used to get suggestions for a further extension of a class
     * expression.
     */
    protected ExtendedSuggestor suggestor;
    /**
     * The class to score the single class expressions.
     */
    protected ScoreCalculator scoreCalculator;
    /**
     * The set of positive examples.
     */
    protected Collection<String> positive;
    /**
     * The set of negative examples.
     */
    protected Collection<String> negative;
    /**
     * The description logic that should be used.
     */
    protected DescriptionLogic logic;
    /**
     * An output stream to which this operator can log details about the refinement
     * process. It is mainly used for debugging.
     */
    protected OutputStream logStream;

    protected boolean debugMode = false;

    /**
     * Constructor.
     * 
     * @param suggestor       The suggestor used to get suggestions for a further
     *                        extension of a class expression
     * @param logic           The description logic that should be used
     * @param scoreCalculator The class to score the single class expressions
     * @param positive        The set of positive examples
     * @param negative        The set of negative examples
     */
    public SuggestorBasedRefinementOperator(ExtendedSuggestor suggestor, DescriptionLogic logic,
            ScoreCalculator scoreCalculator, Collection<String> positive, Collection<String> negative) {
        super();
        this.suggestor = suggestor;
        this.logic = logic;
        this.scoreCalculator = scoreCalculator;
        this.positive = positive;
        this.negative = negative;
    }

    @Override
    public Set<ScoredClassExpression> refine(ClassExpression nextBestExpression, long timeToStop) {
        RecursivlyRefiningVisitor visitor = new RecursivlyRefiningVisitor(this, positive.size(), negative.size(),
                logic, timeToStop);
        nextBestExpression.accept(visitor);
        Set<ScoredClassExpression> results = visitor.getResults();
        logRefinementResults(nextBestExpression, results);
        if (debugMode) {
            for (ScoredClassExpression result : results) {
                SelectionScores scores = suggestor.scoreExpression(result.getClassExpression(), positive, negative);
                if ((result.getPosCount() != scores.posCount) || (result.getNegCount() != scores.negCount)) {
                    LOGGER.error(
                            "Error while checking results! The expression's counts {} differ from the counts when checking it again {}.",
                            result, scores);
                }
            }
        }
        return results;
    }

    /**
     * Logs the given refinement results, i.e., the given base expression and the
     * set of refinements are written to the {@link #logStream}, if the stream is
     * not {@code null}. Otherwise, nothing happens.
     * 
     * @param baseExpression
     * @param results
     */
    protected void logRefinementResults(ClassExpression baseExpression, Set<ScoredClassExpression> results) {
        if (logStream != null) {
            try {
                logStream.write("\nRefining ".getBytes(StandardCharsets.UTF_8));
                logStream.write(baseExpression.toString().getBytes(StandardCharsets.UTF_8));
                logStream.write(" led to ".getBytes(StandardCharsets.UTF_8));
                if (results.size() > 0) {
                    logStream.write(Integer.toString(results.size()).getBytes(StandardCharsets.UTF_8));
                    logStream.write(" new expressions:\n".getBytes(StandardCharsets.UTF_8));
                    IOUtils.writeLines(results, "\n", logStream, StandardCharsets.UTF_8);
                } else {
                    logStream.write("no new expressions.\n".getBytes(StandardCharsets.UTF_8));
                }
            } catch (Exception e) {
                LOGGER.warn("Exception while writing to refinement logging stream.", e);
            }
        }
    }

    /**
     * Sets an output stream to which this operator can log details about the
     * refinement process. It is mainly used for debugging.
     * 
     * @param logStream
     */
    public void setLogStream(OutputStream logStream) {
        this.logStream = logStream;
    }

    public void setDebugMode(boolean debugMode) {
        this.debugMode = debugMode;
    }

    /**
     * An implementation of the visitor pattern, that creates a copy of a given
     * class expression while at the same time searching and replacing a given sub
     * expression with another given sub expression.
     * 
     * <p>
     * Note that this implementation is <b>not thread-safe</b>!
     * </p>
     * 
     * <p>
     * Warning! This class works with side-effects. This method acts based on the
     * context that the visitor has when the method is called. It also stores the
     * results in the visitor's result set.
     * </p>
     * 
     * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
     *
     */
    public static class RecursivlyRefiningVisitor implements ClassExpressionVisitor {

        protected SuggestorBasedRefinementOperator parentOperator;
        protected Set<ScoredClassExpression> results = new HashSet<>();
        protected ClassExpression context = Suggestor.CONTEXT_POSITION_MARKER;
        protected NegatingVisitor negator = new NegatingVisitor();
        protected int numberOfPositives;
        protected int numberOfNegatives;
        protected ClassExpression parentNode = null;
        protected DescriptionLogic logic;
        protected long timeToStop;

        public RecursivlyRefiningVisitor(SuggestorBasedRefinementOperator parentOperator, int numberOfPositives,
                int numberOfNegatives, DescriptionLogic logic, long timeToStop) {
            super();
            this.parentOperator = parentOperator;
            this.numberOfPositives = numberOfPositives;
            this.numberOfNegatives = numberOfNegatives;
            this.logic = logic;
            this.timeToStop = timeToStop;
        }

        protected void addResult(ScoredIRI suggestion, ClassExpression newNode, boolean addedEdge) {
            // Add the suggestion
            ClassExpression newExpression = ClassExpressionUpdater.update(context, Suggestor.CONTEXT_POSITION_MARKER,
                    newNode);
            addResult(newExpression, suggestion, addedEdge);
        }

        protected void addResult(ClassExpression newExpression, SelectionScores scores, boolean addedEdge) {
            // Check results for sanity
            if ((scores.getPosCount() < 0) || (scores.getPosCount() > numberOfPositives) || (scores.getNegCount() < 0)
                    || (scores.getNegCount() > numberOfNegatives)) {
                LOGGER.error("Got wrong counts: #positives={}, #negatives={}, expression={}, scores={}",
                        numberOfPositives, numberOfNegatives, newExpression, scores);
            }
            results.add(parentOperator.scoreCalculator.score(newExpression, scores.getPosCount(), scores.getNegCount(),
                    addedEdge));
            if (logic.supportsComplexConceptNegation()) {
                // Add its negation
                results.add(parentOperator.scoreCalculator.score(negator.negateExpression(newExpression),
                        numberOfPositives - scores.getPosCount(), numberOfNegatives - scores.getNegCount(), addedEdge));
            }
        }

        protected void addContextBasedClassSuggestions(final Set<String> blacklist) {
            Collection<ScoredIRI> suggestions = parentOperator.suggestor.suggestClass(parentOperator.positive,
                    parentOperator.negative, context);
            suggestions.stream().filter(s -> !blacklist.contains(s.getIri()))
                    .forEach(s -> addResult(s, new NamedClass(s.getIri()), false));
            // If the logic supports atomic negation, we should ask for negated classes.
            // However, we only do that in cases in which the context is not simply the
            // position marker since the complex negation already covers these cases OR if
            // the logic does not allow the complex negation.
            if (logic.supportsAtomicNegation() && (!logic.supportsComplexConceptNegation()
                    || !context.equals(Suggestor.CONTEXT_POSITION_MARKER))) {
                suggestions = parentOperator.suggestor.suggestNegatedClass(parentOperator.positive,
                        parentOperator.negative, context);
                suggestions.stream().filter(s -> !blacklist.contains(s.getIri()))
                        .forEach(s -> addResult(s, new NamedClass(s.getIri(), true), false));
            }
        }

        protected void addContextBasedRoleSuggestions(Set<String> blacklist) {
            Collection<ScoredIRI> suggestions = parentOperator.suggestor.suggestProperty(parentOperator.positive,
                    parentOperator.negative, context);
//            ClassExpression newExpression;
            for (ScoredIRI suggestion : suggestions) {
                if (!blacklist.contains(suggestion.getIri())) {
                    // Add the suggestion
                    addResult(suggestion, new SimpleQuantifiedRole(true, suggestion.getIri(), suggestion.isInverted(),
                            NamedClass.TOP), true);
                }
            }
        }

        public void visitAnyNode(ClassExpression node, Set<String> classBlacklist, Set<String> roleBlacklist) {
            if ((timeToStop > 0) && (System.currentTimeMillis() >= timeToStop)) {
                // We have to stop ...
                return;
            }
            if (logic.supportsConceptIntersection() || logic.supportsConceptUnion()) {
                // Can we extend the node with a conjunction?
                boolean conjunction = logic.supportsConceptIntersection();
                // Whatever junction we take, can we switch to the other junction type?
                boolean switchFlag = conjunction && logic.supportsConceptUnion();
                if (parentNode instanceof Junction) {
                    if (switchFlag) {
                        switchFlag = false;
                        conjunction = !((Junction) parentNode).isConjunction();
                    } else {
                        // The logic that we use does not support the other Junction. Hence, we can
                        // leave.
                        return;
                    }
                }
                extendJunction(new Junction(conjunction, Collections.singleton(node)), classBlacklist, roleBlacklist,
                        switchFlag);
            }
        }

        public void extendJunction(Junction junction, Set<String> classBlacklist, Set<String> roleBlacklist,
                boolean switchFlag) {
            // Update context by adding a new junction
            ClassExpression oldContext = context;
            junction.getChildren().add(Suggestor.CONTEXT_POSITION_MARKER);
            context = ClassExpressionUpdater.update(oldContext, Suggestor.CONTEXT_POSITION_MARKER, junction);
            addContextBasedClassSuggestions(classBlacklist);
            addContextBasedRoleSuggestions(roleBlacklist);
            // We are allowed to change the given conjunction into a disjunction
            if (switchFlag) {
                junction.setConjunction(!junction.isConjunction());
                context = ClassExpressionUpdater.update(oldContext, Suggestor.CONTEXT_POSITION_MARKER, junction);
                addContextBasedClassSuggestions(classBlacklist);
                addContextBasedRoleSuggestions(roleBlacklist);
            }
            junction.getChildren().remove(Suggestor.CONTEXT_POSITION_MARKER);
            context = oldContext;
        }

        @SuppressWarnings("unchecked")
        @Override
        public void visitNamedClass(NamedClass node) {
            if ((timeToStop > 0) && (System.currentTimeMillis() >= timeToStop)) {
                // We have to stop ...
                return;
            }
            // Check if this is TOP
            if (NamedClass.TOP.equals(node)) {
                // replace TOP with classes
                addContextBasedClassSuggestions(SetUtils.EMPTY_SET);
                // replace TOP with roles
                addContextBasedRoleSuggestions(SetUtils.EMPTY_SET);
            } else if (!NamedClass.BOTTOM.equals(node)) {
                // Ensure that we do not try to extend BOTTOM
                // We have a blacklist with a single element, so a tree set should be better
                // than a hash set
                visitAnyNode(node, Collections.singleton(node.getName()), SetUtils.EMPTY_SET);
            }
        }

        @Override
        public void visitJunction(Junction node) {
            if ((timeToStop > 0) && (System.currentTimeMillis() >= timeToStop)) {
                // We have to stop ...
                return;
            }
            ClassExpression oldContext = context;
            ClassExpression oldparentNode = parentNode;
            parentNode = node;
            Set<ClassExpression> children = node.getChildren();
            Set<ClassExpression> originalChildren = new HashSet<>(children);
            // 1. Visit children
            children.add(Suggestor.CONTEXT_POSITION_MARKER);
            for (ClassExpression child : originalChildren) {
                children.remove(child);
                context = ClassExpressionUpdater.update(oldContext, Suggestor.CONTEXT_POSITION_MARKER, node);
                child.accept(this);
                children.add(child);
            }
            context = oldContext;
            node.setChildren(originalChildren);
            // 2. Extend this junction
            Set<String> classBlacklist = new HashSet<>();
            Set<String> roleBlacklist = new HashSet<>();
            for (ClassExpression child : originalChildren) {
                if (child instanceof NamedClass) {
                    classBlacklist.add(((NamedClass) child).getName());
                } else if (child instanceof SimpleQuantifiedRole) {
                    roleBlacklist.add(((SimpleQuantifiedRole) child).getRole());
                }
            }
            extendJunction(node, classBlacklist, roleBlacklist, false);
            parentNode = oldparentNode;
            context = oldContext;
        }

        @SuppressWarnings("unchecked")
        @Override
        public void visitSimpleQuantificationRole(SimpleQuantifiedRole node) {
            if ((timeToStop > 0) && (System.currentTimeMillis() >= timeToStop)) {
                // We have to stop ...
                return;
            }
            ClassExpression newExpression = new SimpleQuantifiedRole(node.isExists(), node.getRole(), node.isInverted(),
                    Suggestor.CONTEXT_POSITION_MARKER);
            ClassExpression oldContext = context;
            ClassExpression oldparentNode = parentNode;
            parentNode = node;
            context = ClassExpressionUpdater.update(oldContext, Suggestor.CONTEXT_POSITION_MARKER, newExpression);
            node.getTailExpression().accept(this);
            parentNode = oldparentNode;
            context = oldContext; // set the context back;
            // If 1) this node has the ∃ quantifier, 2) the logic that we use supports the ∀
            // quantifier and 3) this node has something else than TOP as child, we could
            // try a for all
            // quantifier
            if (node.isExists() && logic.supportsUniversalRestrictions()
                    && !NamedClass.TOP.equals(node.getTailExpression())) {
                newExpression = new SimpleQuantifiedRole(false, node.getRole(), node.isInverted(),
                        node.getTailExpression().deepCopy());
                newExpression = ClassExpressionUpdater.update(context, Suggestor.CONTEXT_POSITION_MARKER,
                        newExpression);
                SelectionScores scores = parentOperator.suggestor.scoreExpression(newExpression,
                        parentOperator.positive, parentOperator.negative);
                addResult(newExpression, scores, false);
            }

            visitAnyNode(node, SetUtils.EMPTY_SET, Collections.singleton(node.getRole()));
        }

        private Set<ScoredClassExpression> getResults() {
            return results;
        }
    }

}
