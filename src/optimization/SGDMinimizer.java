package optimization;

import HOSemiCRF.Params;

/**
 * implementation of Stochastic Gradient Descent
 * @author tndoan
 *
 */
public class SGDMinimizer extends Minimizer {

	@Override
	public double[] minimize(AbstractSVRGFunction f, double[] init,
			double learningRate, int maxPasses, int upFreq, double funcTol) {
		// Note: upFreq is not used, 
		// we also dont use learningRate parameter because learning rate will be learned by backtracking
		
		double[] result = new double[init.length];
		System.arraycopy(init, 0, result, 0, init.length);
		double pre_obj = Double.MAX_VALUE;
		
		double alpha = 0.5; // TODO: it will be moved to Params
		double beta = 0.8;  // it will be moved to Params
		double[] temp = new double[init.length]; // should choose a good name
		
		for (int i = 0; i < maxPasses; i++){
			for (int j = 0; j < f.getNumberOfDataPoints(); j++) {
				// this loop get data in order so we need to shuffle data in the pre-processing step
				
				double[] dev = f.takeDerivative(result, j);
				
				double sq_norm_dev = 0.0; // square norm-2 of dev vector
				for (double d : dev) 
					sq_norm_dev += d * d;
				
				// backtracking
				double t = 1.0; // learning rate
				double lhs = Double.MAX_VALUE;
				double f_x =  f.valueAt(result, j); // f(x) does not change in backtracking
				
				while (lhs > f_x - alpha * t * sq_norm_dev) {
					t = beta * t;
					for (int k = 0; k < dev.length; k++){
						temp[k] = result[k] - t * dev[k];
					}
					lhs = f.valueAt(temp, j);
				}
				// end of backtracking
				
				for (int k = 0; k < dev.length; k++){
					result[k] -= t * dev[k];
				}
//				System.out.println(f.valueAt(result)); // this line is for checking
			}
			
			// check convergence
			double cur_obj = f.valueAt(result);
			System.out.println("objective function:" + cur_obj);

			if (i > 1 && Math.abs(pre_obj - cur_obj) < funcTol) {
				this.isConv = true;
				break;
			}
			pre_obj = cur_obj;
		}
		
		return result;
	}

	/**
	 *
	 * @param id
	 * @param f
	 * @param init
	 * @param maxPasses
	 * @param funcTol
	 * @param start
	 * @param end
	 * @return
	 */
	public double[] minimize(int id, AbstractSVRGFunction f, double[] init, int maxPasses, double funcTol, int start, int end) {		
		double[] result = new double[init.length];
		System.arraycopy(init, 0, result, 0, init.length);
		double pre_obj = Double.MAX_VALUE;

		double alpha = 0.5; // TODO: it will be moved to Params
		double beta = 0.8;  // it will be moved to Params
		double[] temp = new double[init.length]; // should choose a good name

		for (int i = 0; i < maxPasses; i++){
			double t = 1.0; // learning rate

			for (int j = start; j < end; j++) {
				// this loop get data in order so we need to shuffle data in the pre-processing step

				double[] dev = f.takeDerivative(result, j);

				double sq_norm_dev = 0.0; // square norm-2 of dev vector
				for (double d : dev)
					sq_norm_dev += d * d;

				// backtracking
				double lhs = Double.MAX_VALUE;
				double f_x =  f.valueAt(result, j); // f(x) does not change in backtracking

				while (lhs > f_x - alpha * t * sq_norm_dev) {
					t = beta * t;
					for (int k = 0; k < dev.length; k++){
						temp[k] = result[k] - t * dev[k];
					}
					lhs = f.valueAt(temp, j);
				}
				// end of backtracking

				for (int k = 0; k < dev.length; k++){
					result[k] -= t * dev[k];
				}
//				System.out.println(f.valueAt(result)); // this line is for checking
			}

			// check convergence
			double cur_obj = f.valueAt(result);
			System.out.println("thread: " + id + " objective function:" + cur_obj);

			if (i > 1 && Math.abs(pre_obj - cur_obj) < funcTol) {
				this.isConv = true;
				break;
			}
			pre_obj = cur_obj;
		}

		return result;
	}
	
	/**
	 * 
	 * @param f
	 * @param init
	 * @param maxPasses
	 * @param funcTol
	 * @return
	 */
	public double[] minimizeNoBackTracking(AbstractSVRGFunction f, double[] init, int maxPasses, double funcTol) {		
		double[] result = new double[init.length];
		System.arraycopy(init, 0, result, 0, init.length);
		double pre_obj = Double.MAX_VALUE;
		
		int numThreads = 8; // TODO must make it more flexible

		int numDataPoints = f.getNumberOfDataPoints();
		int numChunks = numDataPoints / numThreads;
		if ( numDataPoints % numThreads != 0)
			numChunks++;
		
		double count = 0.0; // this counts how many loops we are running
		int[] index = new int[numThreads];

		for (int i = 0; i < maxPasses; i++){
			for (int j = 0; j < numChunks; j++) {
				// this loop get data in order so we need to shuffle data in the pre-processing step
				
				if (j == numChunks - 1) {
					numThreads = numDataPoints - (numChunks - 1) * numThreads;
					index = new int[numThreads];
					
				}
				for (int k = 0; k < numThreads; k++) {
					index[k] = j * numThreads + k;
				}

				count += 1.0;
				double t = getLearningRate(count); // ensure that learning rate will reach zero if number of iteration is to infinity

				double[] dev = f.takeDerivative(result, index);

				for (int k = 0; k < dev.length; k++) {
					result[k] -= t * dev[k];
				}
//				System.out.println(f.valueAt(result)); // this line is for checking
			}

			// check convergence
			double cur_obj = f.valueAt(result);
			System.out.println("objective function:" + cur_obj);

			if (i > 1 && Math.abs(pre_obj - cur_obj) < funcTol) {
				this.isConv = true;
				break;
			}
			pre_obj = cur_obj;
		}

		return result;
	}
	
	/**
	 * 
	 * @param count
	 * @return
	 */
	private double getLearningRate(double count) {
		return 1.0 / (10.0 + count);
	}
}
