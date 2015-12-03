package ChineseSegPOS;

import java.io.*;
import java.util.*;
import HOSemiCRF.*;
import ChineseSegPOS.Features.*;

public class FeatureTypeGen {
    String templateFilename;

    public FeatureTypeGen(String filename) {
        templateFilename = filename;
    }

    public ArrayList<FeatureType> generateFeatureTypes() throws Exception {
        ArrayList<FeatureType> fts = new ArrayList<FeatureType>();
        BufferedReader in = new BufferedReader(new FileReader(templateFilename));
        String line;
        while ((line = in.readLine()) != null) {
            line = line.trim();
            if (line.length() > 0 && line.charAt(0) != '#') {
                String[] toks = line.split("[ :/]");
                if (toks[0].charAt(0)=='U') {
                    fts.add(new UnigramFeature(toks));
                } else if (toks[0].charAt(0)=='B') {
                    fts.add(new Edge());
                    fts.add(new EdgeBag());
                }
            }
        }

        // fts.add(new FirstOrderTransition());
        // fts.add(new SecondOrderTransition());
        // fts.add(new ThirdOrderTransition());

        in.close();
        return fts;
    }

    public static String getObsID(DataSequence seq, int pos, String[] args) {
        String obsID = new String(args[0] + ":");
        for (int i=1; i<args.length; i++) {
            obsID += get1Arg(seq, pos, args[i]);
            if (i < args.length-1) obsID += "/";
        }
        return obsID;
    }

    public static String get1Arg(DataSequence seq, int pos, String arg) {
        if (arg.charAt(0) == '%' && arg.charAt(1) == 'x') {
            String row_str = arg.substring(arg.indexOf('[')+1,arg.indexOf(','));
            String col_str = arg.substring(arg.indexOf(',')+1,arg.indexOf(']'));
            int row = Integer.parseInt(row_str);
            int col = Integer.parseInt(col_str);
            if (pos+row >= 0 && pos+row<seq.length()) {
                WordDetails w = (WordDetails) seq.x(pos+row);
                return w.getElement(col);
            }
        }
        return "";
    }
}

