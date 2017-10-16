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
public class AdaGradTest {

	private final static double TOLERANCE = 0.00000000000000001;

	@Test
	public void testCalculateDeltas() {
		AdaGrad grad = new AdaGrad(1.23D);

		/* ----- round 1 ----- */
		double[] gradients = new double[] { 0.0D };
		double[] deltas = grad.calculateDeltas(gradients);
		for (int i = 0; i < gradients.length; ++i) {
			System.out.println("Index: " + i + " gradient: " + gradients[i] + " delta: " + deltas[i]);
		}
		assertEquals(0.0D, deltas[0], TOLERANCE);

		/* ----- round 2 ----- */
		gradients = new double[] { 199.888D };
		deltas = grad.calculateDeltas(gradients);
		gradients = new double[] { 188.666D };
		deltas = grad.calculateDeltas(gradients);
		gradients = new double[] { 86.666D };
		deltas = grad.calculateDeltas(gradients);
		for (int i = 0; i < gradients.length; ++i) {
			System.out.println("Index: " + i + " gradient: " + gradients[i] + " delta: " + deltas[i]);
		}
		assertEquals(-0.3698751117349966D, deltas[0], TOLERANCE);
	}

}
