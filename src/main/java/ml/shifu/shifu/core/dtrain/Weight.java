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

import java.util.Random;

import ml.shifu.shifu.core.dtrain.nn.update.AdaGradUpdate;
import ml.shifu.shifu.core.dtrain.nn.update.AdamUpdate;
import ml.shifu.shifu.core.dtrain.nn.update.MomentumUpdate;
import ml.shifu.shifu.core.dtrain.nn.update.NesterovUpdate;
import ml.shifu.shifu.core.dtrain.nn.update.RMSPropUpdate;
import ml.shifu.shifu.core.dtrain.nn.update.UpdateRule;
import ml.shifu.shifu.util.ClassUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link Weight} is used to update NN weights according to propagation option. Which is also copied from Encog.
 * 
 * <p>
 * We'd like to reuse code from Encog but unfortunately the methods are private:(.
 */
public class Weight {

    protected static final Logger LOG = LoggerFactory.getLogger(Weight.class);

    /**
     * The zero tolerance to use.
     */
    private final static double ZERO_TOLERANCE = 0.00000000000000001;

    public static final String ADAM_OPTIMIZATION = "ADAM";

    public static final String MOMENTUN_OPTIMIZATION = "MOMENTUM";

    public static final String RMSPROP_OPTIMIZATION = "RMSPROP";

    public static final String ADAGRAD_OPTIMIZATION = "ADAGRAD";

    public static final String NESTEROV_OPTIMIZATION = "NESTEROV";

    private double learningRate;

    private String algorithm;

    // for quick propagation
    private double decay = 0.0001d;
    private double[] lastDelta = null;
    private double[] lastGradient = null;
    private double outputEpsilon = 0.35;
    private double eps = 0.0;
    private double shrink = 0.0;

    // for resilient propagation
    private double[] updateValues = null;
    private static final double DEFAULT_INITIAL_UPDATE = 0.1;
    private static final double DEFAULT_MAX_STEP = 50;

    /**
     * L1 or L2 regulation parameter.
     */
    private double reg;

    /**
     * Number of training records
     */
    private double numTrainSize;

    /**
     * Regulazation level
     */
    private RegulationLevel rl = RegulationLevel.NONE;

    /**
     * Dropout rate.
     */
    private double dropoutRate = 0d;

    /**
     * Random object to do drop out
     */
    private Random random;

    /**
     * Enable Adam, Momentum, AdaGrad or RMSProp optimization, if {@link #updateRule} is null, by default old BGD.
     */
    private UpdateRule updateRule;

    /**
     * Number of weights in network
     */
    private int numWeight;

    /**
     * Learning Decay value, for {@link RMSPropUpdate}
     */
    private double learningDecay;

    /**
     * 'beta1' in Adam optimization, only for Adam
     */
    private double adamBeta1 = 0.9d;

    /**
     * 'beta2' in Adam optimization, only for Adam
     */
    private double adamBeta2 = 0.999d;

    // for back propagation
    private double momentum = 0.5;

    public Weight(int numWeight, double numTrainSize, double rate, String algorithm, double reg, RegulationLevel rl,
            double dropoutRate) {
        this(numWeight, numTrainSize, rate, algorithm, reg, rl, dropoutRate, null);
    }

    public Weight(int numWeight, double numTrainSize, double rate, String algorithm, double reg, RegulationLevel rl,
            double dropoutRate, String propagation) {
        this(numWeight, numTrainSize, rate, algorithm, reg, rl, dropoutRate, propagation, 0.5d, 0d, 0.9d, 0.999d);
    }

    public Weight(int numWeight, double numTrainSize, double rate, String algorithm, double reg, RegulationLevel rl,
            double dropoutRate, String propagation, double momentum, double learningDecay, double adamBeta1,
            double adamBeta2) {
        this.numWeight = numWeight;
        this.dropoutRate = dropoutRate;
        this.random = new Random();
        this.lastDelta = new double[numWeight];
        this.lastGradient = new double[numWeight];
        this.numTrainSize = numTrainSize;
        this.eps = this.outputEpsilon / numTrainSize;
        this.shrink = rate / (1.0 + rate);
        this.learningRate = rate;
        this.algorithm = algorithm;
        this.updateValues = new double[numWeight];

        for(int i = 0; i < this.updateValues.length; i++) {
            this.updateValues[i] = DEFAULT_INITIAL_UPDATE;
            this.lastDelta[i] = 0;
        }

        this.reg = reg;
        if(rl != null) {
            this.rl = rl;
        }

        this.momentum = momentum;
        this.learningDecay = learningDecay;
        this.adamBeta1 = adamBeta1;
        this.adamBeta2 = adamBeta2;

        // init update rule
        if(propagation == null || propagation.length() == 0) {
            this.updateRule = null;
        } else if(ADAM_OPTIMIZATION.equalsIgnoreCase(propagation)) {
            this.updateRule = new AdamUpdate();
        } else if(ADAGRAD_OPTIMIZATION.equalsIgnoreCase(propagation)) {
            this.updateRule = new AdaGradUpdate();
        } else if(RMSPROP_OPTIMIZATION.equalsIgnoreCase(propagation)) {
            this.updateRule = new RMSPropUpdate();
        } else if(MOMENTUN_OPTIMIZATION.equalsIgnoreCase(propagation)) {
            this.updateRule = new MomentumUpdate();
        } else if(NESTEROV_OPTIMIZATION.equalsIgnoreCase(propagation)) {
            this.updateRule = new NesterovUpdate();
        } else {
            try {
                this.updateRule = (UpdateRule) ClassUtils.newInstance(Class.forName(propagation));
            } catch (Exception e) {
                LOG.info("Class not found for {}, set update rule to null", propagation);
                this.updateRule = null;
            }
        }

        if(this.updateRule != null) {
            this.updateRule.init(this);
        }
    }

    public double[] calculateWeights(double[] weights, double[] gradients, int iteration) {
        if(this.updateRule != null) {
            this.updateRule.update(gradients, weights, iteration);
            return weights;
        } else {
            for(int i = 0; i < gradients.length; i++) {
                if(this.dropoutRate > 0 && this.random.nextDouble() < this.dropoutRate) {
                    // drop out, no need to update weight, just continue next weight
                    continue;
                }
                switch(this.rl) {
                    case NONE:
                        weights[i] += updateWeight(i, weights, gradients);
                        break;
                    case L1:
                        if(Double.compare(this.reg, 0d) == 0) {
                            weights[i] += updateWeight(i, weights, gradients);
                        } else {
                            double shrinkValue = this.reg / getNumTrainSize();
                            double delta = updateWeight(i, weights, gradients);
                            weights[i] = Math.signum(delta) * Math.max(0.0, Math.abs(delta) - shrinkValue);
                        }
                        break;
                    case L2:
                    default:
                        weights[i] += (updateWeight(i, weights, gradients) - this.reg * weights[i] / getNumTrainSize());
                        break;
                }
            }
            return weights;
        }
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
        double delta = (gradients[index] * this.getLearningRate()) + (this.lastDelta[index] * this.getMomentum());
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
        throw new RuntimeException("SCG propagation is not supported in distributed NN computing.");
    }

    private double updateWeightRLP(int index, double[] weights, double[] gradients) {
        // multiply the current and previous gradient, and take the sign. We want to see if the gradient has changed its
        // sign.
        final int change = DTrainUtils.sign(gradients[index] * lastGradient[index]);
        double weightChange = 0;

        // if the gradient has retained its sign, then we increase the delta so that it will converge faster
        if(change > 0) {
            double delta = this.updateValues[index] * DTrainUtils.POSITIVE_ETA;
            delta = Math.min(delta, DEFAULT_MAX_STEP);
            weightChange = DTrainUtils.sign(gradients[index]) * delta;
            this.updateValues[index] = delta;
            lastGradient[index] = gradients[index];
        } else if(change < 0) {
            // if change<0, then the sign has changed, and the last delta was too big
            double delta = this.updateValues[index] * DTrainUtils.NEGATIVE_ETA;
            delta = Math.max(delta, DTrainUtils.DELTA_MIN);
            this.updateValues[index] = delta;
            weightChange = -this.lastDelta[index];
            // set the previous gradient to zero so that there will be no adjustment the next iteration
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

    /**
     * @return the numTrainSize
     */
    public double getNumTrainSize() {
        return numTrainSize;
    }

    /**
     * @param numTrainSize
     *            the numTrainSize to set
     */
    public void setNumTrainSize(double numTrainSize) {
        this.numTrainSize = numTrainSize;
        this.eps = this.outputEpsilon / numTrainSize;
    }

    /**
     * @return the numWeight
     */
    public int getNumWeight() {
        return numWeight;
    }

    /**
     * @param numWeight
     *            the numWeight to set
     */
    public void setNumWeight(int numWeight) {
        this.numWeight = numWeight;
    }

    /**
     * @return the momentum
     */
    public double getMomentum() {
        return momentum;
    }

    /**
     * @param momentum
     *            the momentum to set
     */
    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    /**
     * @return the learningDecay
     */
    public double getLearningDecay() {
        return learningDecay;
    }

    /**
     * @param learningDecay
     *            the learningDecay to set
     */
    public void setLearningDecay(double learningDecay) {
        this.learningDecay = learningDecay;
    }

    /**
     * @return the adamBeta1
     */
    public double getAdamBeta1() {
        return adamBeta1;
    }

    /**
     * @param adamBeta1
     *            the adamBeta1 to set
     */
    public void setAdamBeta1(double adamBeta1) {
        this.adamBeta1 = adamBeta1;
    }

    /**
     * @return the adamBeta2
     */
    public double getAdamBeta2() {
        return adamBeta2;
    }

    /**
     * @param adamBeta2
     *            the adamBeta2 to set
     */
    public void setAdamBeta2(double adamBeta2) {
        this.adamBeta2 = adamBeta2;
    }
}
