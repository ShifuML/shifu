/**
 * Copyright [2012-2014] eBay Software Foundation
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

import ml.shifu.guagua.master.MasterComputable;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ConvergeJudger;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.util.CommonUtils;

import org.encog.neural.networks.BasicNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * {@link NNMaster} is used to accumulate all workers NN parameters.
 * <p/>
 * <p/>
 * We accumulate all gradients from workers to calculate model weights. And set weights to workers. Then workers use
 * weights to set their models and train for another iteration.
 * <p/>
 * <p/>
 * This logic follows Encog multi-core implementation.
 * <p/>
 * <p/>
 * Make sure workers and master use the same initialization weights.
 */
public class NNMaster implements MasterComputable<NNParams, NNParams> {

    private static final Logger LOG = LoggerFactory.getLogger(NNMaster.class);

    /**
     * Global master NN parameters instance which is used to update model weights by using accumulated gradients.
     */
    private NNParams globalNNParams = new NNParams();

    /**
     * Whether some configurations are initialized
     */
    private AtomicBoolean isInitialized = new AtomicBoolean(false);

    /**
     * Model configuration loaded from configuration file.
     */
    private ModelConfig modelConfig;

    /**
     * To calculate weights according to last weights and accumulated gradients
     */
    private Weight weightCalculator = null;

    /**
     * Column configuration loaded from configuration file.
     */
    private List<ColumnConfig> columnConfigList;

    private String propagation = "Q";

    private Double rawLearningRate = 0.1d;

    private Double learningRate = 0.1d;

    private double learningDecay = 0d;

    private double convergenceThreshold = 0d;
    
    /**
     * Convergence judger instance for convergence criteria checking.
     */
    private ConvergeJudger judger = new ConvergeJudger();
    
    @Override
    public NNParams compute(MasterContext<NNParams, NNParams> context) {
        // For first step, we not only initialize whole context but also return weights to master to make sure all
        // workers and master are using the same weights.
        if(this.isInitialized.compareAndSet(false, true)) {
            // initilize configuration
            init(context);

            // first iteration is used to set initial weights
            NNParams params = initWeights();
            // should be set here to make sure master and workers use the same weights
            this.globalNNParams.setWeights(params.getWeights());

            return params;
        }

        if(context.getWorkerResults() == null) {
            throw new IllegalArgumentException("workers' results are null.");
        }

        double totalTestError = 0;
        double totalTrainError = 0;
        int size = 0;

        // before accumulate, reset gradients and train size
        this.globalNNParams.reset();

        for(NNParams nn: context.getWorkerResults()) {
            totalTestError += nn.getTestError();
            totalTrainError += nn.getTrainError();
            this.globalNNParams.accumulateGradients(nn.getGradients());
            this.globalNNParams.accumulateTrainSize(nn.getTrainSize());
            size++;
        }

        // worker result size is 0. throw exception because shouldn't happen
        if(size == 0) {
            throw new IllegalArgumentException("workers' results are empty.");
        }

        // initialize weightCalCulater.
        if(this.weightCalculator == null) {
            this.learningRate = this.rawLearningRate;
            this.weightCalculator = new Weight(this.globalNNParams.getGradients().length,
                    this.globalNNParams.getTrainSize(), learningRate, propagation);
        } else {
            this.learningRate = this.learningRate * (1.0d - this.learningDecay);
            this.weightCalculator.setLearningRate(this.learningRate);
        }

        // use last weights and current gradients to calculate
        double[] weights = this.weightCalculator.calculateWeights(this.globalNNParams.getWeights(),
                this.globalNNParams.getGradients());

        this.globalNNParams.setWeights(weights);

        double currentTestError = totalTestError / size;
        double currentTrainError = totalTrainError / size;

        LOG.info("NNMaster compute iteration {} ( avg train error {}, avg validation error {} )", new Object[] {
                context.getCurrentIteration(), currentTrainError, currentTestError });

        NNParams params = new NNParams();
        params.setTrainError(currentTrainError);
        params.setTestError(currentTestError);
        // prevent null point
        params.setGradients(new double[0]);
        params.setWeights(weights);
        LOG.debug("master result {} in iteration {}", params, context.getCurrentIteration());
        
        // Convergence judging part
        LOG.info("Judging convergence :");
        
        double avgErr = (currentTrainError + currentTestError) / 2;
        
        LOG.info("NNMaster compute iteration {} Average error: {} , Threshold: {}"
                , context.getCurrentIteration(), avgErr, convergenceThreshold);
        
        if (judger.judge(avgErr, convergenceThreshold)) {
            LOG.info("NNMaster compute iteration {} converged !", context.getCurrentIteration());
            params.setHalt(true);
        } else {
            LOG.info("NNMaster compute iteration {} not converged yet !", context.getCurrentIteration());
        }

        return params;
    }

    @SuppressWarnings({ "unchecked" })
    private NNParams initWeights() {
        NNParams params = new NNParams();

        int[] inputAndOutput = NNUtils.getInputOutputCandidateCounts(this.columnConfigList);
        int inputNodeCount = inputAndOutput[0] == 0 ? inputAndOutput[2] : inputAndOutput[0];
        int outputNodeCount = inputAndOutput[1];

        int numLayers = (Integer) this.modelConfig.getParams().get(NNTrainer.NUM_HIDDEN_LAYERS);
        List<String> actFunc = (List<String>) this.modelConfig.getParams().get(NNTrainer.ACTIVATION_FUNC);
        List<Integer> hiddenNodeList = (List<Integer>) this.modelConfig.getParams().get(NNTrainer.NUM_HIDDEN_NODES);

        BasicNetwork network = NNUtils.generateNetwork(inputNodeCount, outputNodeCount, numLayers, actFunc,
                hiddenNodeList);

        params.setTrainError(0);
        params.setTestError(0);
        // prevent null point
        params.setGradients(new double[0]);
        params.setWeights(network.getFlat().getWeights());
        return params;
    }

    public void init(MasterContext<NNParams, NNParams> context) {
        Properties props = context.getProps();

        try {
            SourceType sourceType = SourceType.valueOf(props.getProperty(NNConstants.NN_MODELSET_SOURCE_TYPE,
                    SourceType.HDFS.toString()));

            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(NNConstants.SHIFU_NN_MODEL_CONFIG),
                    sourceType);

            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    props.getProperty(NNConstants.SHIFU_NN_COLUMN_CONFIG), sourceType);
            this.propagation = (String) this.modelConfig.getParams().get(NNTrainer.PROPAGATION);
            this.rawLearningRate = Double.valueOf(this.modelConfig.getParams().get(NNTrainer.LEARNING_RATE).toString());
            Object learningDecayO = this.modelConfig.getParams().get("LearningDecay");
            if(learningDecayO != null) {
                this.learningDecay = Double.valueOf(learningDecayO.toString());
            }
            Double threshold =  this.modelConfig.getTrain().getConvergenceThreshold();
            this.convergenceThreshold = threshold == null ? 0d : threshold.doubleValue();
            
            LOG.info("learningDecay in master is :{}", learningDecay);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
