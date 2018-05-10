/**
 * 
 */
package ml.shifu.shifu.core.dtrain;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author xchen7
 *
 */
public class AdaGrad extends AbstractAdaptiveLearningRate {

	private static final Logger LOGGER = LoggerFactory.getLogger(AdaGrad.class);

	private static final double DEFAULT_EPSILON = Math.pow(10.0D, -8D);

	private double[] squaredGradients;

	/**
	 * Constructor.
	 * 
	 * @param learningRate
	 *            the learning rate
	 */
	public AdaGrad(double learningRate) {
		this(DEFAULT_EPSILON, learningRate);
	}

	/**
	 * Constructor.
	 * 
	 * @see {@linkplain AbstractAdaptiveLearningRate#AbstractAdaptiveLearningRate(double, double, double)}
	 */
	public AdaGrad(double epsilon, double learningRate) {
		/* AdaGrad doesn't need decay anyway give it a positive value. */
		super(0.95D, epsilon, learningRate);
	}

	@Override
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

}
