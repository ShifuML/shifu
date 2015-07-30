/**
 * Copyright [2012-2014] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.dtrain;

/**
 * {@link Weight} is used to update NN weights according to propagation option. Which is also copied from Encog.
 * <p/>
 * <p/>
 * We'd like to reuse code from Encog but unfortunately the methods are private:(.
 */
public class Weight {
    /**
     * The zero tolerance to use.
     */
    private final static double ZERO_TOLERANCE = 0.00000000000000001;

    private double learningRate;

    private String algorithm;

    // for quick propagation
    private double decay = 0.0001d;
    private double[] lastDelta = null;
    private double[] lastGradient = null;
    private double outputEpsilon = 0.35;
    private double eps = 0.0;
    private double shrink = 0.0;

    // for back propagation
    private double momentum = 0.5;

    // for resilient propagation
    private double[] updateValues = null;
    private static final double DEFAULT_INITIAL_UPDATE = 0.1;
    private static final double DEFAULT_MAX_STEP = 50;

    public Weight(int numWeight, double numTrainSize, double rate, String algorithm) {

        this.lastDelta = new double[numWeight];
        this.lastGradient = new double[numWeight];
        this.eps = this.outputEpsilon / numTrainSize;
        this.shrink = rate / (1.0 + rate);
        this.setLearningRate(rate);
        this.algorithm = algorithm;
        this.updateValues = new double[numWeight];

        for(int i = 0; i < this.updateValues.length; i++) {
            this.updateValues[i] = DEFAULT_INITIAL_UPDATE;
            this.lastDelta[i] = 0;
        }
    }

    public double[] calculateWeights(double[] weights, double[] gradients) {
        for(int i = 0; i < gradients.length; i++) {
            weights[i] += updateWeight(i, weights, gradients);
        }

        return weights;
    }

    private double updateWeight(int index, double[] weights, double[] gradients) {
        if(this.algorithm.equalsIgnoreCase(DTrainUtils.BACK_PROPAGATION)) {
            return updateWeightBP(index, weights, gradients);
        } else if(this.algorithm.equalsIgnoreCase(DTrainUtils.QUICK_PROPAGATION)) {
            return updateWeightQBP(index, weights, gradients);
        } else if(this.algorithm.equalsIgnoreCase(DTrainUtils.MANHATTAN_PROPAGATION)) {
            return updateWeightMHP(index, weights, gradients);
        } else if(this.algorithm.equalsIgnoreCase(DTrainUtils.SCALEDCONJUGATEGRADIENT)) {
            return updateWeightSCG(index, weights, gradients);
        } else if(this.algorithm.equalsIgnoreCase(DTrainUtils.RESILIENTPROPAGATION)) {
            return updateWeightRLP(index, weights, gradients);
        }

        return 0.0;

    }

    private double updateWeightBP(int index, double[] weights, double[] gradients) {
        double delta = (gradients[index] * this.getLearningRate()) + (this.lastDelta[index] * this.momentum);
        this.lastDelta[index] = delta;
        return delta;
    }

    private double updateWeightQBP(int index, double[] weights, double[] gradients) {

        final double w = weights[index];
        final double d = this.lastDelta[index];
        final double s = -gradients[index] + this.decay * w;
        final double p = -lastGradient[index];
        double nextStep = 0.0;

        // The step must always be in direction opposite to the slope.
        if(d < 0.0) {
            // If last step was negative...
            if(s > 0.0) {
                // Add in linear term if current slope is still positive.
                nextStep -= this.eps * s;
            }
            // If current slope is close to or larger than prev slope...
            if(s >= (this.shrink * p)) {
                // Take maximum size negative step.
                nextStep += this.getLearningRate() * d;
            } else {
                // Else, use quadratic estimate.
                nextStep += d * s / (p - s);
            }
        } else if(d > 0.0) {
            // If last step was positive...
            if(s < 0.0) {
                // Add in linear term if current slope is still negative.
                nextStep -= this.eps * s;
            }
            // If current slope is close to or more neg than prev slope...
            if(s <= (this.shrink * p)) {
                // Take maximum size negative step.
                nextStep += this.getLearningRate() * d;
            } else {
                // Else, use quadratic estimate.
                nextStep += d * s / (p - s);
            }
        } else {
            // Last step was zero, so use only linear term.
            nextStep -= this.eps * s;
        }

        // update global data arrays
        this.lastDelta[index] = nextStep;
        this.lastGradient[index] = gradients[index];

        return nextStep;
    }

    private double updateWeightMHP(int index, double[] weights, double[] gradients) {
        if(Math.abs(gradients[index]) < ZERO_TOLERANCE) {
            return 0;
        } else if(gradients[index] > 0) {
            return this.getLearningRate();
        } else {
            return -this.getLearningRate();
        }
    }

    private double updateWeightSCG(int index, double[] weights, double[] gradients) {
        // TODO Auto-generated method stub
        return 0;
    }

    private double updateWeightRLP(int index, double[] weights, double[] gradients) {
        // multiply the current and previous gradient, and take the
        // sign. We want to see if the gradient has changed its sign.
        final int change = DTrainUtils.sign(gradients[index] * lastGradient[index]);
        double weightChange = 0;

        // if the gradient has retained its sign, then we increase the
        // delta so that it will converge faster
        if(change > 0) {
            double delta = this.updateValues[index] * DTrainUtils.POSITIVE_ETA;
            delta = Math.min(delta, DEFAULT_MAX_STEP);
            weightChange = DTrainUtils.sign(gradients[index]) * delta;
            this.updateValues[index] = delta;
            lastGradient[index] = gradients[index];
        } else if(change < 0) {
            // if change<0, then the sign has changed, and the last
            // delta was too big
            double delta = this.updateValues[index] * DTrainUtils.NEGATIVE_ETA;
            delta = Math.max(delta, DTrainUtils.DELTA_MIN);
            this.updateValues[index] = delta;
            weightChange = -this.lastDelta[index];
            // set the previous gradent to zero so that there will be no
            // adjustment the next iteration
            lastGradient[index] = 0;
        } else if(change == 0) {
            // if change==0 then there is no change to the delta
            final double delta = this.updateValues[index];
            weightChange = DTrainUtils.sign(gradients[index]) * delta;
            lastGradient[index] = gradients[index];
        }

        this.lastDelta[index] = weightChange;
        // apply the weight change, if any
        return weightChange;
    }

    /**
     * @return the learningRate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * @param learningRate
     *            the learningRate to set
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}
