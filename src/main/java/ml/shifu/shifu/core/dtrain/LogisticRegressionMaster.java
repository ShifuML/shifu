/*
 * Copyright [2013-2014] eBay Software Foundation
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

import java.util.Arrays;
import java.util.Random;

import ml.shifu.guagua.master.MasterComputable;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.guagua.util.NumberFormatUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link LogisticRegressionMaster} defines logic to update global <a
 * href=http://en.wikipedia.org/wiki/Logistic_regression >logistic regression</a> model.
 * 
 * <p>
 * At first iteration, master builds a random model then send to all workers to start computing. This is to make all
 * workers use the same model at the starting time.
 * 
 * <p>
 * At other iterations, master works:
 * <ul>
 * <li>1. Accumulate all gradients from workers.</li>
 * <li>2. Update global models by using accumulated gradients.</li>
 * <li>3. Send new global model to workers by returning model parameters.</li>
 * </ul>
 */
// FIXME miss one parameter: size, the formula should be weights[i] -= learnRate * (1/size) * gradients[i]; pass from
// workers
public class LogisticRegressionMaster implements MasterComputable<LogisticRegressionParams, LogisticRegressionParams> {

    private static final Logger LOG = LoggerFactory.getLogger(LogisticRegressionMaster.class);

    private static final Random RANDOM = new Random();

    private int inputNum;

    private double[] weights;

    private double learnRate;

    private void init(MasterContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        this.inputNum = NumberFormatUtils.getInt(LogisticRegressionContants.LR_INPUT_NUM,
                LogisticRegressionContants.LR_INPUT_DEFAULT_NUM);
        this.learnRate = NumberFormatUtils.getDouble(LogisticRegressionContants.LR_LEARNING_RATE,
                LogisticRegressionContants.LR_LEARNING_DEFAULT_RATE);
    }

    @Override
    public LogisticRegressionParams compute(MasterContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        if(context.isFirstIteration()) {
            init(context);
            weights = new double[this.inputNum + 1];
            for(int i = 0; i < weights.length; i++) {
                weights[i] = RANDOM.nextDouble();
            }
        } else {
            double[] gradients = new double[this.inputNum + 1];
            double sumError = 0.0d;
            int size = 0;
            for(LogisticRegressionParams param: context.getWorkerResults()) {
                if(param != null) {
                    for(int i = 0; i < gradients.length; i++) {
                        gradients[i] += param.getParameters()[i];
                    }
                    sumError += param.getError();
                }
                size++;
            }
            for(int i = 0; i < weights.length; i++) {
                weights[i] -= learnRate * gradients[i];
            }
            LOG.debug("DEBUG: Weights: {}", Arrays.toString(this.weights));
            LOG.info("Iteration {} with error {}", context.getCurrentIteration(), sumError / size);
        }
        return new LogisticRegressionParams(weights);
    }

}
