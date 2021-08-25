/*
 * Encog(tm) Core v3.4 - Java Version
 * http://www.heatonresearch.com/encog/
 * https://github.com/encog/encog-java-core
 
 * Copyright 2008-2016 Heaton Research, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *   
 * For more information on Heaton Research copyrights, licenses 
 * and trademarks visit:
 * http://www.heatonresearch.com/copyright
 */
package ml.shifu.shifu.core.dtrain.nn.update;

import java.util.Set;

/**
 * Created by jeffh on 7/15/2016.
 * 
 * <p>
 * Copied from https://github.com/encog/encog-java-core/.
 */
public class AdamUpdate implements UpdateRule {
    private double[] m;
    private double[] v;
    private double beta1 = 0.9;
    private double beta2 = 0.999;
    private double eps = 1e-8;

    private double learningRate;

    private Update update;

    @Override
    public void init(Update update) {
        this.update = update;
        this.learningRate = update.getLearningRate();
        this.m = new double[update.getNumWeight()];
        this.v = new double[update.getNumWeight()];
        this.beta1 = update.getAdamBeta1();
        this.beta2 = update.getAdamBeta2();
    }

    @Override
    public void update(double[] gradients, double[] weights, int iteration, Set<Integer> fixedWeights) {
        for(int i = 0; i < weights.length; i++) {
            if(fixedWeights.contains(i))
                continue;

            double avgGrad = gradients[i] / this.update.getNumTrainSize();

            m[i] = (this.beta1 * m[i]) + (1 - this.beta1) * avgGrad;
            v[i] = (this.beta2 * v[i]) + (1 - this.beta2) * avgGrad * avgGrad;

            double mCorrect = m[i] / (1 - Math.pow(this.beta1, iteration));
            double vCorrect = v[i] / (1 - Math.pow(this.beta2, iteration));

            final double delta = (this.learningRate  * mCorrect)
                    / (Math.sqrt(vCorrect) + this.eps);
            weights[i] += delta;
        }
    }
}
