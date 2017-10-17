/**
 * 
 */
package ml.shifu.shifu.core.dtrain;

/**
 * @author xchen7
 *
 */
public interface AdaptiveLearningRate {

	/**
	 * Calculate deltas.
	 * 
	 * @param currGradients
	 *            the current gradients
	 * @return the deltas in array
	 */
	double[] calculateDeltas(double[] currGradients);

	/**
	 * Calculate weights.
	 * 
	 * @param prevWeights
	 *            the previous weights
	 * @param currGradients
	 *            the current gradients
	 * @return the new weights in array
	 */
	double[] calculateWeights(double[] prevWeights, double[] currGradients);
}
