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
public class AdaDelta {

	private static final Logger LOGGER = LoggerFactory.getLogger(AdaDelta.class);

	private static final double DEFAULT_DECAY = 0.95D;
	private static final double DEFAULT_EPSILON = Math.pow(10.0D, -8D);

	private final double decay;
	private final double epsilon;

	private double[] prevSquaredAverageDeltas;
	private double[] prevGradients;

	/**
	 * Default constructor.
	 */
	public AdaDelta() {
		decay = DEFAULT_DECAY;
		epsilon = DEFAULT_EPSILON;
	}

	/**
	 * Constructor.
	 * 
	 * @param decay
	 *            the decay which should be (0, 1)
	 * @param epsilon
	 *            the epsilon which should be > 0
	 */
	public AdaDelta(double decay, double epsilon) {
		if (decay <= 0 || decay >= 1) {
			throw new IllegalArgumentException("Invalid decay: " + decay);
		}
		if (epsilon <= 0) {
			throw new IllegalArgumentException("Invalid epsilon: " + epsilon);
		}
		this.decay = decay;
		this.epsilon = epsilon;

		if (LOGGER.isDebugEnabled()) {
			LOGGER.debug("Initialized with decay " + decay + " and epsilon " + epsilon);
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

		/*
		 * For index 0 (initial state) the squared average gradients are 0 so
		 * set previous gradients to 0。
		 */
		if (null == prevGradients || 0 == prevGradients.length) {
			prevGradients = newEmptyArray(currGradients.length);
		}

		/* For index 0 (initial state) the squared average deltas are 0。 */
		if (null == prevSquaredAverageDeltas || 0 == prevSquaredAverageDeltas.length) {
			prevSquaredAverageDeltas = newEmptyArray(length);
		}

		if (length != prevGradients.length || length != prevSquaredAverageDeltas.length) {
			throw new IllegalArgumentException("Cannot adjust grdients' size during runtime: " + prevGradients.length
					+ " -> " + length + " or " + prevSquaredAverageDeltas.length + " -> " + length);
		}

		/* Calculate current deltas. */
		double[] rmsGradients = new double[length];
		double[] prevRMSDeltas = new double[length];
		double[] currDeltas = new double[length];
		for (int i = 0; i < length; ++i) {
			rmsGradients[i] = calculateRMSGradient(prevGradients[i], currGradients[i]);
			prevRMSDeltas[i] = Math.sqrt(prevSquaredAverageDeltas[i] + epsilon);
			currDeltas[i] = (-1) * currGradients[i] * (prevRMSDeltas[i] / rmsGradients[i]);

			/* Accumulate squared average deltas. */
			prevSquaredAverageDeltas[i] = decay * Math.pow(prevSquaredAverageDeltas[i], 2)
					+ (1 - decay) * Math.pow(currDeltas[i], 2);

			if (LOGGER.isDebugEnabled()) {
				LOGGER.debug("Calculate current deltas: index=" + i + "&currGradient=" + currGradients[i]
						+ "&rmsGradient=" + rmsGradients[i] + "&prevRMSDelta=" + prevRMSDeltas[i] + "&currDelta="
						+ currDeltas[i] + "&prevSquaredAverageDelta=" + prevSquaredAverageDeltas[i]);
			}
		}

		/* Update gradients after computation. */
		prevGradients = currGradients;

		return currDeltas;
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
	private double calculateRMSGradient(double prevGradient, double currGradient) {
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
	private static double[] newEmptyArray(int length) {
		double[] array = new double[length];
		Arrays.fill(array, 0.0D);
		return array;
	}
}
