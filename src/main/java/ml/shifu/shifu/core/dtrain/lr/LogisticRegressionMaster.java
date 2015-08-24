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
package ml.shifu.shifu.core.dtrain.lr;

import java.io.IOException;
import java.util.List;
import java.util.Properties;

import ml.shifu.guagua.master.MasterComputable;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ConvergeJudger;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.RegulationLevel;
import ml.shifu.shifu.core.dtrain.Weight;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.util.CommonUtils;

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
 * 
 * <p>
 * L1 and l2 regulations are supported by configuration: RegularizedConstant in model params of ModelConfig.json.
 */
public class LogisticRegressionMaster implements MasterComputable<LogisticRegressionParams, LogisticRegressionParams> {

    private static final Logger LOG = LoggerFactory.getLogger(LogisticRegressionMaster.class);

    /**
     * Input column number without bias
     */
    private int inputNum;

    /**
     * This is the model weights in LR which will be updated each iteration
     */
    private double[] weights;

    /**
     * Learning rate configured by user in params
     */
    private double learningRate = 1.0d;

    /**
     * Regulation parameter for l1 or l2
     */
    private double regularizedConstant = 0.0d;

    /**
     * To calculate weights according to last weights and accumulated gradients
     */
    private Weight weightCalculator = null;

    /**
     * Model configuration loaded from configuration file.
     */
    private ModelConfig modelConfig;

    /**
     * Column Config list read from HDFS
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * Convergence threshold setting by user in ModelConfig.json.
     */
    private double convergenceThreshold;

    /**
     * Convergence judger instance for convergence checking.
     */
    private ConvergeJudger judger = new ConvergeJudger();

    /**
     * Propagation type for lr model setting: Q, B, R, C
     */
    private String propagation = "Q";

    private void init(MasterContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        loadConfigFiles(context.getProps());
        this.learningRate = Double.valueOf(this.modelConfig.getParams().get(CommonConstants.LR_LEARNING_RATE)
                .toString());
        int[] inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(this.columnConfigList);
        this.inputNum = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];

        Double threshold = this.modelConfig.getTrain().getConvergenceThreshold();
        this.convergenceThreshold = threshold == null ? 0d : threshold.doubleValue();
        LOG.info("Convergence threshold in master is :{}", this.convergenceThreshold);

        this.propagation = (String) this.modelConfig.getParams().get(NNTrainer.PROPAGATION);

        Object rconstant = this.modelConfig.getParams().get(CommonConstants.LR_REGULARIZED_CONSTANT);
        this.regularizedConstant = NumberFormatUtils.getDouble(rconstant == null ? "" : rconstant.toString(), 0d);
    }

    @Override
    public LogisticRegressionParams compute(MasterContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        if(context.isFirstIteration()) {
            init(context);
            weights = new double[this.inputNum + 1];
            for(int i = 0; i < weights.length; i++) {
                weights[i] = nextDouble(-1, 1);
            }
            return new LogisticRegressionParams(weights);
        } else {
            // append bias
            double[] gradients = new double[this.inputNum + 1];
            double trainError = 0.0d, testError = 0d;
            long trainSize = 0, testSize = 0;
            for(LogisticRegressionParams param: context.getWorkerResults()) {
                if(param != null) {
                    for(int i = 0; i < gradients.length; i++) {
                        gradients[i] += param.getParameters()[i];
                    }
                    trainError += param.getTrainError();
                    testError += param.getTestError();
                    trainSize += param.getTrainSize();
                    testSize += param.getTestSize();
                }
            }

            if(this.weightCalculator == null) {
                this.weightCalculator = new Weight(weights.length, trainSize, learningRate, this.propagation,
                        this.regularizedConstant, RegulationLevel.to(this.modelConfig.getParams().get(
                                CommonConstants.REG_LEVEL_KEY)));
            } else {
                this.weightCalculator.setNumTrainSize(trainSize);
            }

            this.weights = this.weightCalculator.calculateWeights(this.weights, gradients);

            double finalTrainError = trainError / trainSize;
            double finalTestError = testError / testSize;
            LOG.info("Iteration {} with train error {}, test error {}", context.getCurrentIteration(), finalTrainError,
                    finalTestError);
            LogisticRegressionParams lrParams = new LogisticRegressionParams(weights, finalTrainError, finalTestError,
                    trainSize, testSize);
            if(judger.judge(finalTrainError + finalTestError / 2, convergenceThreshold)) {
                LOG.info("LRMaster compute iteration {} converged !", context.getCurrentIteration());
                lrParams.setHalt(true);
            } else {
                LOG.info("LRMaster compute iteration {} not converged yet !", context.getCurrentIteration());
            }
            return lrParams;
        }
    }

    private void loadConfigFiles(final Properties props) {
        try {
            SourceType sourceType = SourceType.valueOf(props.getProperty(NNConstants.NN_MODELSET_SOURCE_TYPE,
                    SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(NNConstants.SHIFU_NN_MODEL_CONFIG),
                    sourceType);
            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    props.getProperty(NNConstants.SHIFU_NN_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public final double nextDouble(final double min, final double max) {
        final double range = max - min;
        return (range * Math.random()) + min;
    }

}
