/**
 * 
 */
package ml.shifu.shifu.core.dtrain;

import static org.junit.Assert.*;

import org.junit.Test;

/**
 * @author xchen7
 *
 */
public class AdaDeltaTest {

	private final static double TOLERANCE = 0.00000000000000001;

	@Test
	public void testCalculateDeltas() {
		AdaDelta delta = new AdaDelta();

		/* ----- round 1 ----- */
		double[] gradients = new double[] { 0.0D };
		double[] deltas = delta.calculateDeltas(gradients);
		for (int i = 0; i < gradients.length; ++i) {
			System.out.println("Index: " + i + " gradient: " + gradients[i] + " delta: " + deltas[i]);
		}
		assertEquals(0.0D, deltas[0], TOLERANCE);

		/* ----- round 2 ----- */
		gradients = new double[] { 9.99D };
		deltas = delta.calculateDeltas(gradients);
		for (int i = 0; i < gradients.length; ++i) {
			System.out.println("Index: " + i + " gradient: " + gradients[i] + " delta: " + deltas[i]);
		}
		assertEquals(-4.081801639405311D, deltas[0], TOLERANCE);

		/* ----- round 3 ----- */
		gradients = new double[] { 8.88D };
		deltas = delta.calculateDeltas(gradients);
		for (int i = 0; i < gradients.length; ++i) {
			System.out.println("Index: " + i + " gradient: " + gradients[i] + " delta: " + deltas[i]);
		}
		gradients = new double[] { 7.77D };
		deltas = delta.calculateDeltas(gradients);
		for (int i = 0; i < gradients.length; ++i) {
			System.out.println("Index: " + i + " gradient: " + gradients[i] + " delta: " + deltas[i]);
		}
		gradients = new double[] { 6.66D };
		deltas = delta.calculateDeltas(gradients);
		for (int i = 0; i < gradients.length; ++i) {
			System.out.println("Index: " + i + " gradient: " + gradients[i] + " delta: " + deltas[i]);
		}
		assertEquals(-1.073898925470707D, deltas[0], TOLERANCE);
	}

}
