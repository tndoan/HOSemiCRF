package ChineseSegPOS;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;

import HOSemiCRF.DataSequence;
import HOSemiCRF.DataSet;
import HOSemiCRF.FeatureGenerator;
import HOSemiCRF.FeatureType;
import HOSemiCRF.HighOrderSemiCRF;
import HOSemiCRF.LabelMap;
import HOSemiCRF.Params;
import HOSemiCRF.Scorer;

public class ChineseSegPOS {
	
    static String trainFilename = "/home/thanhnam/workspace/ctb5/ctb5/CRF-combinedLabels-small/ctb5.train";
    static String testFilename = "/home/thanhnam/workspace/ctb5/ctb5/CRF-combinedLabels-small/ctb5.test";
    static String predictFilename = "/home/thanhnam/workspace/ctb5/ctb5/CRF-combinedLabels-small/ctb5.predict";
    static String templateFilename = "/home/thanhnam/workspace/ctb5/ctb5/CRF-combinedLabels-small/ctb5.template";
   
    HighOrderSemiCRF crfModel;
    FeatureGenerator featureGen;
    LabelMap labelmap;
    String configFile;

    public ChineseSegPOS(String filename) {
        labelmap = new LabelMap();
        configFile = filename;
    }

    public DataSet readTagged(String filename) throws Exception {
        BufferedReader in = new BufferedReader(new FileReader(filename));

        ArrayList<DataSequence> td = new ArrayList<>();
        ArrayList<WordDetails> inps = new ArrayList<WordDetails>();
        ArrayList<String> labels = new ArrayList<String>();
        String line;

        while ((line = in.readLine()) != null) {
            if (line.length() > 0) {
                String[] toks = line.split("[ \t]");
                inps.add(new WordDetails(toks));
                labels.add(toks[toks.length-1]);
            } else if (labels.size() > 0) {
            	DataSequence ds = new DataSequence(labelmap.mapArrayList(labels), inps.toArray(), labelmap);
                //td.add(new DataSequence(labelmap.mapArrayList(labels), inps.toArray(), labelmap));
            	td.add(ds);
                inps = new ArrayList<WordDetails>();
                labels = new ArrayList<String>();
            }
        }
        if (labels.size() > 0) {
            td.add(new DataSequence(labelmap.mapArrayList(labels), inps.toArray(), labelmap));
        }

        in.close();
        return new DataSet(td);
    }

    public void createFeatureGenerator() throws Exception {
        FeatureTypeGen ftsGen = new FeatureTypeGen(ChineseSegPOS.templateFilename);
        ArrayList<FeatureType> fts = ftsGen.generateFeatureTypes();
        Params params = new Params(configFile, labelmap.size());
        featureGen = new FeatureGenerator(fts, params);
    }

    public void train() throws Exception {
        File modeldir = new File("learntModels/");
        modeldir.mkdirs();
        
        DataSet trainData = readTagged(ChineseSegPOS.trainFilename);
        labelmap.write("learntModels/labelmap");
        
        createFeatureGenerator();
        featureGen.initialize(trainData.getSeqList());
        featureGen.write("learntModels/features");

        crfModel = new HighOrderSemiCRF(featureGen);
        crfModel.train(trainData.getSeqList(), 5);
        crfModel.write("learntModels/crf");
    }

    public void test() throws Exception {
        labelmap.read("learntModels/labelmap");
        createFeatureGenerator();
        featureGen.read("learntModels/features");
        crfModel = new HighOrderSemiCRF(featureGen);
        crfModel.read("learntModels/crf");
        
        System.out.print("Running Viterbi...");
        DataSet testData = readTagged(ChineseSegPOS.testFilename);
        long startTime = System.currentTimeMillis();
        crfModel.runViterbi(testData.getSeqList());
        System.out.println("done in " + (System.currentTimeMillis() - startTime) + " ms");
        
        testData.writeToFile(ChineseSegPOS.predictFilename);
        score(testFilename, ChineseSegPOS.predictFilename);
    }
    
    public void score(String trueFilename, String predictedFilename) throws Exception {
        System.out.println("Scoring results...");
        long startTime = System.currentTimeMillis();
        DataSet trueTestData = readTagged(trueFilename);
        DataSet predictedTestData = readTagged(predictedFilename);
        Scorer scr = new Scorer(trueTestData.getSeqList(), predictedTestData.getSeqList(), labelmap, false);
        scr.tokenScore();
        System.out.println("done in " + (System.currentTimeMillis() - startTime) + " ms");
    }

    public static void main(String argv[]) throws Exception {
        ChineseSegPOS chSegPOS = new ChineseSegPOS(argv[1]);
        if (argv[0].toLowerCase().equals("all")) {
            chSegPOS.train();
            chSegPOS.test();
        } else if (argv[0].toLowerCase().equals("train")) {
            chSegPOS.train();
        } else if (argv[0].toLowerCase().equals("test")) {
            chSegPOS.test();
        } else if (argv[0].toLowerCase().equals("score")) {
            chSegPOS.score(argv[3], argv[4]);
        }
    }
}

