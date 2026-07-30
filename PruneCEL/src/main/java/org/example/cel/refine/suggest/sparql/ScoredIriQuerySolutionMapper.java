package org.example.cel.refine.suggest.sparql;

import java.util.Set;
import java.util.function.Function;

import org.apache.jena.query.QuerySolution;
import org.example.cel.refine.suggest.ScoredIRI;

public class ScoredIriQuerySolutionMapper implements Function<QuerySolution, ScoredIRI> {
    protected String iriVariable;
    protected Set<String> blacklist;

    public ScoredIriQuerySolutionMapper(String iriVariable, Set<String> blacklist) {
        super();
        this.iriVariable = iriVariable;
        this.blacklist = blacklist;
    }

    @Override
    public ScoredIRI apply(QuerySolution s) {
        if (s.contains(iriVariable)) {
            String iri = s.getResource(iriVariable).getURI();
            if (iri != null) {
                if (!blacklist.contains(iri)) {
                    return new ScoredIRI(s.getResource(iriVariable).getURI(), s.getLiteral("posHits").getInt(),
                            s.getLiteral("negHits").getInt());
                }
            } else {
                // FIXME We found a blank node. Let's ignore it.
                return null;
            }
        } else {
            return new ScoredIRI(null, s.getLiteral("posHits").getInt(), s.getLiteral("negHits").getInt());
        }
        return null;
    }
}