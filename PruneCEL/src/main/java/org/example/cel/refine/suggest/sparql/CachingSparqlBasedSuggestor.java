package org.example.cel.refine.suggest.sparql;

import java.util.Collection;
import java.util.Objects;

import org.aksw.jenax.arq.connection.core.QueryExecutionFactory;
import org.example.cel.DescriptionLogic;
import org.example.cel.expression.ClassExpression;
import org.example.cel.refine.suggest.SelectionScores;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;

public class CachingSparqlBasedSuggestor extends SparqlBasedSuggestor {

    private static final Logger LOGGER = LoggerFactory.getLogger(CachingSparqlBasedSuggestor.class);

    protected LoadingCache<CacheItem, SelectionScores> cache;
    protected InternalLoader loader;

    public CachingSparqlBasedSuggestor(QueryExecutionFactory queryExecFactory, DescriptionLogic logic) {
        super(queryExecFactory, logic);
        loader = new InternalLoader(this);
        cache = CacheBuilder.newBuilder().maximumSize(100).build(loader);
    }

    protected SelectionScores scorePreparedExpression(ClassExpression prepared, Collection<String> positive,
            Collection<String> negative) {
        try {
            return cache.get(new CacheItem(prepared, positive, negative));
        } catch (Exception e) {
            LOGGER.error("Error while accessing the cache. Calculating the result without the cache.", e);
            return super.scorePreparedExpression(prepared, positive, negative);
        }
    }

    protected SelectionScores scorePreparedExpressionWithoutCache(ClassExpression prepared, Collection<String> positive,
            Collection<String> negative) {
        return super.scorePreparedExpression(prepared, positive, negative);
    }

    protected static class InternalLoader extends CacheLoader<CacheItem, SelectionScores> {

        protected CachingSparqlBasedSuggestor suggestor;

        public InternalLoader(CachingSparqlBasedSuggestor suggestor) {
            this.suggestor = suggestor;
        }

        @Override
        public SelectionScores load(CacheItem key) throws Exception {
            return suggestor.scorePreparedExpressionWithoutCache(key.prepared, key.positive, key.negative);
        }

    }

    protected static class CacheItem {
        protected ClassExpression prepared;
        protected Collection<String> positive;
        protected Collection<String> negative;

        public CacheItem(ClassExpression prepared, Collection<String> positive, Collection<String> negative) {
            super();
            this.prepared = prepared;
            this.positive = positive;
            this.negative = negative;
        }

        @Override
        public int hashCode() {
            return Objects.hash(negative, positive, prepared);
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj)
                return true;
            if (obj == null)
                return false;
            if (getClass() != obj.getClass())
                return false;
            CacheItem other = (CacheItem) obj;
            return Objects.equals(negative, other.negative) && Objects.equals(positive, other.positive)
                    && Objects.equals(prepared, other.prepared);
        }
    }
}
