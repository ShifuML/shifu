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
public class RMSProp extends AbstractAdaptiveLearningRate {

	private static final Logger LOGGER = LoggerFactory.getLogger(RMSProp.class);

	/*
	 * Geoff Hinton suggests γ to be set to 0.9, while a good default value for
	 * the learning rate η is 0.001.
	 */
	private static final double DEFAULT_DECAY = 0.90D;
	private static final double DEFAULT_EPSILON = Math.pow(10.0D, -8D);
	private static final double DEFAULT_LEARNING_RATE = 0.001D;

	private double[] prevGradients;

	/**
	 * Default constructor.
	 */
	public RMSProp() {
		this(DEFAULT_DECAY, DEFAULT_EPSILON, DEFAULT_LEARNING_RATE);
	}

	/**
	 * Constructor.
	 * 
	 * @see {@linkplain AbstractAdaptiveLearningRate#AbstractAdaptiveLearningRate(double, double, double)}
	 */
	public RMSProp(double decay, double epsilon, double learningRate) {
		super(decay, epsilon, learningRate);
	}

	@Override
	public double[] calculateDeltas(double[] currGradients) {
		if (null == currGradients || 0 == currGradients.length) {
			throw new IllegalArgumentException("Current gradients are null or empty: " + currGradients);
		}

		final int length = currGradients.length;

		/*
		 * For index 0 (initial state) the squared average gradients are 0 so
		 * set previous gradients to 0。
		 */
		if (null == prevGradients || 0 == prevGradients.length) {
			prevGradients = newEmptyArray(currGradients.length);
		}

		if (length != prevGradients.length) {
			throw new IllegalArgumentException(
					"Cannot adjust grdients' size during runtime: " + prevGradients.length + " -> " + length);
		}

		/* Calculate current deltas. */
		double[] rmsGradients = new double[length];
		double[] currDeltas = new double[length];
		for (int i = 0; i < length; ++i) {
			rmsGradients[i] = calculateRMSGradient(prevGradients[i], currGradients[i]);
			currDeltas[i] = (-1) * currGradients[i] * (learningRate / rmsGradients[i]);

			if (LOGGER.isDebugEnabled()) {
				LOGGER.debug("Calculate current deltas: index=" + i + "&currGradient=" + currGradients[i]
						+ "&rmsGradient=" + rmsGradients[i] + "&currDelta=" + currDeltas[i]);
			}
		}

		/* Update gradients after computation. */
		prevGradients = currGradients;

		return currDeltas;
	}

}
