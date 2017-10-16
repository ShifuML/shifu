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
public class AdaGrad {

	private static final Logger LOGGER = LoggerFactory.getLogger(AdaGrad.class);

	private static final double DEFAULT_EPSILON = Math.pow(10.0D, -8D);

	private final double epsilon;
	private final double learningRate;

	private double[] squaredGradients;

	/**
	 * Constructor.
	 * 
	 * @param learningRate
	 *            the learning rate
	 */
	public AdaGrad(double learningRate) {
		this(learningRate, DEFAULT_EPSILON);
	}

	/**
	 * Constructor.
	 * 
	 * @param learningRate
	 *            the learning rate
	 * @param epsilon
	 *            the epsilon which should be > 0
	 */
	public AdaGrad(double learningRate, double epsilon) {
		if (epsilon <= 0) {
			throw new IllegalArgumentException("Invalid epsilon: " + epsilon);
		}
		if (learningRate <= 0) {
			throw new IllegalArgumentException("Invalid learning rate: " + learningRate);
		}
		this.epsilon = epsilon;
		this.learningRate = learningRate;

		if (LOGGER.isDebugEnabled()) {
			LOGGER.debug("Initialized with learning rate: " + learningRate + " and epsilon " + epsilon);
		}
	}

	/**
	 * Calculate deltas.
	 * 
	 * @param currGradients
	 *            the current gradients
	 * @return the deltas in array
	 */
	public double[] calculateDeltas(double[] currGradients) {
		if (null == currGradients || 0 == currGradients.length) {
			throw new IllegalArgumentException("Current gradients are null or empty: " + currGradients);
		}

		final int length = currGradients.length;

		/* For initial state the squared gradients are 0。 */
		if (null == squaredGradients || 0 == squaredGradients.length) {
			squaredGradients = newEmptyArray(currGradients.length);
		}

		if (length != squaredGradients.length) {
			throw new IllegalArgumentException(
					"Cannot adjust grdients' size during runtime: " + squaredGradients.length + " -> " + length);
		}

		double[] currDeltas = new double[length];
		for (int i = 0; i < length; ++i) {

			/* Accumulate current squared gradients. */
			squaredGradients[i] = squaredGradients[i] + currGradients[i] * currGradients[i];

			currDeltas[i] = (-1) * currGradients[i] * (learningRate / Math.sqrt(squaredGradients[i] + epsilon));

			if (LOGGER.isDebugEnabled()) {
				LOGGER.debug("Calculate current deltas: index=" + i + "&currGradient=" + currGradients[i]
						+ "&squaredGradient=" + squaredGradients[i] + "&currDelta=" + currDeltas[i]);
			}
		}

		return currDeltas;
	}

	/**
	 * New empty double array with all the elements set to 0.
	 * 
	 * @param length
	 *            the array length
	 * @return the empty double array
	 */
	private static double[] newEmptyArray(int length) {
		double[] array = new double[length];
		Arrays.fill(array, 0.0D);
		return array;
	}
}
