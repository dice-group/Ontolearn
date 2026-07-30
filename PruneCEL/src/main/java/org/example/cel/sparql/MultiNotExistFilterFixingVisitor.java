package org.example.cel.sparql;

import java.util.List;

import org.apache.jena.query.Query;
import org.apache.jena.sparql.core.TriplePath;
import org.apache.jena.sparql.expr.E_NotExists;
import org.apache.jena.sparql.expr.Expr;
import org.apache.jena.sparql.syntax.Element;
import org.apache.jena.sparql.syntax.ElementAssign;
import org.apache.jena.sparql.syntax.ElementBind;
import org.apache.jena.sparql.syntax.ElementData;
import org.apache.jena.sparql.syntax.ElementDataset;
import org.apache.jena.sparql.syntax.ElementExists;
import org.apache.jena.sparql.syntax.ElementFilter;
import org.apache.jena.sparql.syntax.ElementGroup;
import org.apache.jena.sparql.syntax.ElementLateral;
import org.apache.jena.sparql.syntax.ElementMinus;
import org.apache.jena.sparql.syntax.ElementNamedGraph;
import org.apache.jena.sparql.syntax.ElementNotExists;
import org.apache.jena.sparql.syntax.ElementOptional;
import org.apache.jena.sparql.syntax.ElementPathBlock;
import org.apache.jena.sparql.syntax.ElementService;
import org.apache.jena.sparql.syntax.ElementSubQuery;
import org.apache.jena.sparql.syntax.ElementTriplesBlock;
import org.apache.jena.sparql.syntax.ElementUnion;
import org.apache.jena.sparql.syntax.ElementVisitor;
import org.apache.jena.vocabulary.RDF;

/**
 * Transforms problematic statements like
 * 
 * <pre>
 * FILTER NOT EXISTS {
     ?pos &lt;http://w3id.org/dice-research/qa-bench#hasQuery&gt; ?x2 .
     FILTER NOT EXISTS { 
       ?class a &lt;http://www.w3.org/2002/07/owl#Class&gt; .
       FILTER NOT EXISTS { ?x2 a ?class . } 
     }
     ?class a &lt;http://www.w3.org/2002/07/owl#Class&gt;.
   }
 * </pre>
 * 
 * into
 * 
 * <pre>
 * FILTER NOT EXISTS {
     ?pos &lt;http://w3id.org/dice-research/qa-bench#hasQuery&gt; ?x2 .
     ?x2 a ?class .
     ?class a &lt;http://www.w3.org/2002/07/owl#Class&gt; .
   }
 * </pre>
 * 
 * Note: this class has an inner state and is not thread-safe!
 * 
 * @author Michael R&ouml;der (michael.roeder@uni-paderborn.de)
 *
 */
public class MultiNotExistFilterFixingVisitor implements ElementVisitor {

    protected String variableName;
    protected String typeUri;
    protected Element innerElement = null;

    public MultiNotExistFilterFixingVisitor(String variableName, String typeUri) {
        super();
        if (variableName.startsWith("?")) {
            this.variableName = variableName.substring(1);
        } else {
            this.variableName = variableName;
        }
        this.typeUri = typeUri;
    }

    public void fixQuery(Query query) {
        if (query != null) {
            // We are within a new (sub)-query, so we shouldn't have any inner element (this
            // shouldn't be necessary though)
            innerElement = null;
            Element e = query.getQueryPattern();
            if (e != null) {
                e.visit(this);
                // If something has been found
                if (innerElement != null) {
                    query.setQueryPattern(innerElement);
                    innerElement = null;
                    // We have to check the new element
                    fixQuery(query);
                }
            }
        }
    }

    public void visit(ElementNotExists el) {
        Element ene = el.getElement();
        if (ene == null) {
            return;
        }
        if (checkFilter(ene)) {
            return;
        } else {
            // This doesn't seem to be something that is interesting for our case... just
            // visit the element
            ene.visit(this);
            // If something has been found
            if (innerElement != null) {
                // We cannot set the element, so let's simply propagate this up...
                innerElement = new ElementNotExists(innerElement);
            }
        }
    }

    protected boolean checkFilter(Element ene) {
        // If we have a group of elements
        if ((ene != null) && (ene instanceof ElementGroup)) {
            ElementGroup eg = (ElementGroup) ene;
            List<Element> elements = eg.getElements();
            if (elements.size() == 1) {
                // Why this is a group? Anyways, let the next case decide whether it is
                // interesting for us...
                ene = elements.get(0);
            } else if (elements.size() == 2) {
                int tripleId = findVariableTypeTriple(elements);
                if (tripleId < 0) {
                    // This element contains more information than the simple one we are looking
                    // for...
                    visitChildren(elements);
                }
                Element child = elements.get(tripleId == 0 ? 1 : 0);
                // Check for a not exists filter in the form of ElementNotExists
                if (child instanceof ElementNotExists) {
                    // We found what we were looking for!
                    innerElement = reduceIfSingleElementGroup(((ElementNotExists) child).getElement());
                    return true;
                }
                // Check for a not exists filter in the form of ElementFilter
                if (child instanceof ElementFilter) {
                    Expr exp = ((ElementFilter) child).getExpr();
                    if (exp instanceof E_NotExists) {
                        // We found what we were looking for!
                        innerElement = reduceIfSingleElementGroup(((E_NotExists) exp).getElement());
                        return true;
                    }
                }
            }
        }
        // If we found a single ElementNotExists as child
        if (ene instanceof ElementNotExists) {
            // We found what we were looking for!
            innerElement = ene;
            return true;
        }
        return false;
    }

    private Element reduceIfSingleElementGroup(Element element) {
        if (element instanceof ElementGroup) {
            ElementGroup eg = (ElementGroup) element;
            // This group has a single element and is not needed anymore. We can simply
            // return this single element.
            if (eg.getElements().size() == 1) {
                return eg.get(0);
            }
        }
        return element;
    }

    protected int findVariableTypeTriple(List<Element> elements) {
        Element e;
        for (int i = 0; i < elements.size(); ++i) {
            e = elements.get(i);
            if (e instanceof ElementPathBlock) {
                ElementPathBlock epb = (ElementPathBlock) e;
                TriplePath tp = epb.patternElts().next();
                // If we have ?variable a typeUri
                if (tp.getSubject().isVariable() && tp.getSubject().getName().equals(variableName)
                        && tp.getPredicate().getURI().equals(RDF.type.getURI()) && tp.getObject().isURI()
                        && tp.getObject().getURI().equals(typeUri)) {
                    return i;
                }
            }
        }
        return -1;
    }

    protected void visitChildren(List<Element> elements) {
        for (int i = 0; i < elements.size(); ++i) {
            elements.get(i).visit(this);
            // If something has been found
            if (innerElement != null) {
                elements.set(i, innerElement);
                innerElement = null;
                // we have to visit the new element
                --i;
            }
        }
    }

    @Override
    public void visit(ElementTriplesBlock el) {
        // nothing to do
    }

    @Override
    public void visit(ElementPathBlock el) {
        // nothing to do
    }

    @Override
    public void visit(ElementFilter el) {
        if (el.getExpr() instanceof E_NotExists) {
            E_NotExists ene = (E_NotExists) el.getExpr();
            Element e = ene.getElement();
            if (checkFilter(e)) {
                return;
            } else {
                // This doesn't seem to be something that is interesting for our case... just
                // visit the element
                e.visit(this);
                // If something has been found
                if (innerElement != null) {
                    // We cannot set the element, so let's simply propagate this up...
                    innerElement = new ElementFilter(new E_NotExists(innerElement));
                }
            }
        }
    }

    @Override
    public void visit(ElementAssign el) {
        // nothing to do
    }

    @Override
    public void visit(ElementBind el) {
        // nothing to do
    }

    @Override
    public void visit(ElementData el) {
        // nothing to do
    }

    @Override
    public void visit(ElementUnion el) {
        visitChildren(el.getElements());
    }

    @Override
    public void visit(ElementOptional el) {
        if (el.getOptionalElement() != null) {
            el.getOptionalElement().visit(this);
            // If something has been found
            if (innerElement != null) {
                // We cannot set the optional element, so let's simply propagate this up...
                innerElement = new ElementOptional(innerElement);
            }
        }
    }

    @Override
    public void visit(ElementLateral el) {
        if (el.getLateralElement() != null) {
            el.getLateralElement().visit(this);
            // If something has been found
            if (innerElement != null) {
                // We cannot set the lateral element, so let's simply propagate this up...
                innerElement = new ElementLateral(innerElement);
            }
        }
    }

    @Override
    public void visit(ElementGroup el) {
        visitChildren(el.getElements());
    }

    @Override
    public void visit(ElementDataset el) {
        if (el.getElement() != null) {
            el.getElement().visit(this);
            // If something has been found
            if (innerElement != null) {
                // We cannot set the inner element, so let's simply propagate this up...
                innerElement = new ElementDataset(el.getDataset(), innerElement);
            }
        }
    }

    @Override
    public void visit(ElementNamedGraph el) {
        if (el.getElement() != null) {
            el.getElement().visit(this);
            // If something has been found
            if (innerElement != null) {
                // We cannot set the inner element, so let's simply propagate this up...
                innerElement = new ElementNamedGraph(el.getGraphNameNode(), innerElement);
            }
        }
    }

    @Override
    public void visit(ElementExists el) {
        if (el.getElement() != null) {
            el.getElement().visit(this);
            // If something has been found
            if (innerElement != null) {
                // We cannot set the inner element, so let's simply propagate this up...
                innerElement = new ElementExists(innerElement);
            }
        }
    }

    @Override
    public void visit(ElementMinus el) {
        if (el.getMinusElement() != null) {
            el.getMinusElement().visit(this);
            // If something has been found
            if (innerElement != null) {
                // We cannot set the inner element, so let's simply propagate this up...
                innerElement = new ElementMinus(innerElement);
            }
        }
    }

    @Override
    public void visit(ElementService el) {
        if (el.getElement() != null) {
            el.getElement().visit(this);
            // If something has been found
            if (innerElement != null) {
                // We cannot set the inner element, so let's simply propagate this up...
                innerElement = new ElementService(el.getServiceNode(), innerElement, el.getSilent());
            }
        }
    }

    @Override
    public void visit(ElementSubQuery el) {
        fixQuery(el.getQuery());
    }
}
