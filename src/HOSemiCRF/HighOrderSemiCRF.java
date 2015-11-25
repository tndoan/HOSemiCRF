/*
Copyright (C) 2012 Nguyen Viet Cuong, Ye Nan, Sumit Bhagwani

This file is part of HOSemiCRF.

HOSemiCRF is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

HOSemiCRF is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with HOSemiCRF. If not, see <http://www.gnu.org/licenses/>.
*/

package HOSemiCRF;

import java.io.*;
import java.util.*;

import edu.stanford.nlp.optimization.QNMinimizer;
import optimization.FirstOrderDiffFunction;
import optimization.SGDMinimizer;
import optimization.SVRGMinimizer;
import Parallel.*;

/**
 * High-order semi-CRF class
 * @author Nguyen Viet Cuong
 */
public class HighOrderSemiCRF {

    FeatureGenerator featureGen; // Feature generator
    double[] lambda; // Feature weight vector
	
    /**
     * Construct and initialize a high-order semi-CRF from feature generator.
     * @param fgen Feature generator
     */
    public HighOrderSemiCRF(FeatureGenerator fgen) {
        featureGen = fgen;
        lambda = new double[featureGen.featureMap.size()];
        Arrays.fill(lambda, 0.0);
    }

    /**
     * Train a high-order semi-CRF from data.
     * @param data Training data
     * @param method	1 for quasi-Newton; 2 for SVRG; 3 for batch SVRG(Experimental); 4 SGD with backtracking; 
     * 					otherwise, mini-batch SGD without backtracking
     */
    public void train(List<DataSequence> data, int method) {
    	// use library to do minimization
    	int maxIters = featureGen.params.maxIters;
    	double epsForConvergence = featureGen.params.epsForConvergence;
    	double learningRate = featureGen.params.getLearningRate();
    	
    	if (method == 1) {
	        QNMinimizer qn = new QNMinimizer();
	        Function df = new Function(featureGen, data);
	        lambda = qn.minimize(df, epsForConvergence, lambda, maxIters);
	        System.out.println(df.valueAt(lambda));
    	} else if (method == 2)	{
	        FirstOrderDiffFunction func = new FirstOrderDiffFunction(featureGen, data);
	        SVRGMinimizer svrg = new SVRGMinimizer();
	        lambda = svrg.minimize(func, lambda, learningRate, maxIters, epsForConvergence);
    	} else if (method == 3) {
    		FirstOrderDiffFunction func = new FirstOrderDiffFunction(featureGen, data);
	        SVRGMinimizer svrg = new SVRGMinimizer();
	        lambda = svrg.minimize(func, lambda, learningRate, featureGen.params.getNumRan(), maxIters, featureGen.params.getUpFreq(), epsForConvergence);
	        System.out.println(func.valueAt(lambda));
    	} else if (method == 4) {
    		FirstOrderDiffFunction func = new FirstOrderDiffFunction(featureGen, data);
    		SGDMinimizer sgd = new SGDMinimizer();
    		lambda = sgd.minimize(func, lambda, learningRate, maxIters, epsForConvergence);
    		System.out.println(func.valueAt(lambda));
    	} else {
    		FirstOrderDiffFunction func = new FirstOrderDiffFunction(featureGen, data);
    		SGDMinimizer sgd = new SGDMinimizer();
    		lambda = sgd.minimizeNoBackTracking(func, lambda, maxIters, epsForConvergence); // TODO: numThread must be flexible
    		System.out.println(func.valueAt(lambda));
    	}
    }

    /**
     * Run Viterbi algorithm on testing data.
     * @param data Testing data
     */
    public void runViterbi(List<DataSequence> data) throws Exception {
        Viterbi tester = new Viterbi(featureGen, lambda, data);
        Scheduler sch = new Scheduler(tester, featureGen.params.numthreads, Scheduler.DYNAMIC_NEXT_AVAILABLE);
        sch.run();
    }
	
    /**
     * Write the high-order semi-CRF to a file.
     * @param filename Name of the output file
     */
    public void write(String filename) throws Exception {
        PrintWriter out = new PrintWriter(new FileOutputStream(filename));
        out.println(lambda.length);
        for (int i = 0; i < lambda.length; i++) {
            out.println(lambda[i]);
        }
        out.close();
    }

    /**
     * Read the high-order semi-CRF from a file.
     * @param filename Name of the input file
     */
    public void read(String filename) throws Exception {
        BufferedReader in = new BufferedReader(new FileReader(filename));
        int featureNum = Integer.parseInt(in.readLine());
        lambda = new double[featureNum];
        for (int i = 0; i < featureNum; i++) {
            String line = in.readLine();
            lambda[i] = Double.parseDouble(line);
        }
        in.close();
    }
}
