/**
 * 
 */
package ml.shifu.shifu.core.dtrain;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author xchen7
 *
 */
public abstract class AbstractAdaptiveLearningRate implements AdaptiveLearningRate {

	private static final Logger LOGGER = LoggerFactory.getLogger(AbstractAdaptiveLearningRate.class);

	protected final double decay;
	protected final double epsilon;
	protected final double learningRate;

	/**
	 * Constructor.
	 * 
	 * @param decay
	 *            the decay which should be (0, 1)
	 * @param epsilon
	 *            the epsilon which should be > 0
	 * @param learningRate
	 *            the learning rate
	 */
	protected AbstractAdaptiveLearningRate(double decay, double epsilon, double learningRate) {
		if (decay <= 0 || decay >= 1) {
			throw new IllegalArgumentException("Invalid decay: " + decay);
		}
		if (epsilon <= 0) {
			throw new IllegalArgumentException("Invalid epsilon: " + epsilon);
		}
		if (learningRate <= 0) {
			throw new IllegalArgumentException("Invalid learning rate: " + learningRate);
		}
		this.decay = decay;
		this.epsilon = epsilon;
		this.learningRate = learningRate;

		if (LOGGER.isDebugEnabled()) {
			LOGGER.debug("Initialized with decay " + decay + " and epsilon " + epsilon + " and learning rate "
					+ learningRate);
		}
	}

	@Override
	public double[] calculateWeights(double[] prevWeights, double[] currGradients) {
		if (null == prevWeights || 0 == prevWeights.length) {
			throw new IllegalArgumentException("Previous weights are null or empty: " + prevWeights);
		}
		if (null == currGradients || 0 == currGradients.length) {
			throw new IllegalArgumentException("Current gradients are null or empty: " + currGradients);
		}

		final int length = prevWeights.length;

		if (length != currGradients.length) {
			throw new IllegalArgumentException("Previous weights and current gradients don't match in length: " + length
					+ " <-> " + currGradients.length);
		}

		double[] deltas = calculateDeltas(currGradients);
		double[] weights = new double[length];
		for (int i = 0; i < length; ++i) {
			weights[i] = prevWeights[i] + deltas[i];
		}

		return weights;
	}

	/**
	 * Calculate RMS gradient.
	 * 
	 * @param prevGradient
	 *            the previous gradient
	 * @param currGradient
	 *            the current gradient
	 * @return the RMS gradient
	 */
	protected double calculateRMSGradient(double prevGradient, double currGradient) {
		double squaredAverage = decay * Math.pow(prevGradient, 2) + (1 - decay) * Math.pow(currGradient, 2);
		double rmsGradient = Math.sqrt(squaredAverage + epsilon);
		return rmsGradient;
	}

	/**
	 * New empty double array with all the elements set to 0.
	 * 
	 * @param length
	 *            the array length
	 * @return the empty double array
	 */
	protected static double[] newEmptyArray(int length) {
		double[] array = new double[length];
		Arrays.fill(array, 0.0D);
		return array;
	}

}
