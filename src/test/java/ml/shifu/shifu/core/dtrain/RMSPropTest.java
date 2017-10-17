/**
 * 
 */
package ml.shifu.shifu.core.dtrain;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

/**
 * @author xchen7
 *
 */
public class RMSPropTest {

	private final static double TOLERANCE = 0.00000000000000001;

	@Test
	public void testCalculateDeltas() {
		AdaptiveLearningRate rmsProp = new RMSProp();

		/* ----- round 1 ----- */
		double[] gradients = new double[] { 0.0D };
		double[] deltas = rmsProp.calculateDeltas(gradients);
		for (int i = 0; i < gradients.length; ++i) {
			System.out.println("Index: " + i + " gradient: " + gradients[i] + " delta: " + deltas[i]);
		}
		assertEquals(0.0D, deltas[0], TOLERANCE);

		/* ----- round 2 ----- */
		gradients = new double[] { 19.88D };
		deltas = rmsProp.calculateDeltas(gradients);
		for (int i = 0; i < gradients.length; ++i) {
			System.out.println("Index: " + i + " gradient: " + gradients[i] + " delta: " + deltas[i]);
		}
		gradients = new double[] { 16.18D };
		deltas = rmsProp.calculateDeltas(gradients);
		for (int i = 0; i < gradients.length; ++i) {
			System.out.println("Index: " + i + " gradient: " + gradients[i] + " delta: " + deltas[i]);
		}
		assertEquals(-8.279793312133782E-4, deltas[0], TOLERANCE);
	}

}
