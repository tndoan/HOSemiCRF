package ChineseSegPOS.Features;

import java.util.*;
import HOSemiCRF.*;
import ChineseSegPOS.*;

/**
 * Unigram features
 * @author Nguyen Viet Cuong
 */
public class UnigramFeature extends FeatureType {

    String[] args;

    public UnigramFeature(String[] args) {
        this.args = args;
    }
	
    public ArrayList<String> generateObsAt(DataSequence seq, int segStart, int segEnd) {
        ArrayList<String> obs = new ArrayList<String>();
        for (int i = segStart; i <= segEnd; i++) {
            obs.add(FeatureTypeGen.getObsID(seq,i,args));
        }
        return obs;
    }

    public int order() {
        return 0;
    }
}
