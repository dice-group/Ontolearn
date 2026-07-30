package org.example.cel;

import org.apache.jena.riot.RDFParser;
import org.apache.jena.riot.system.StreamRDF;
import org.apache.jena.riot.system.StreamRDFBase;
import org.apache.jena.riot.system.StreamRDFReject;
import org.apache.jena.sys.JenaSystem;

public class TempCheck {

    public static void main(String[] args) {
        JenaSystem.init();

        String file = "file:///home/quannian/Tentris_Graph/QALD10/QALD10_ALL.nt";

        RDFParser.source(file).parse(new StreamRDFBase());
    }
}
