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

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.List;
import java.util.Properties;

import ml.shifu.guagua.hadoop.io.GuaguaLineRecordReader;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.util.MemoryDiskList;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.guagua.worker.AbstractWorkerComputable;
import ml.shifu.guagua.worker.WorkerContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.math3.distribution.PoissonDistribution;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.encog.mathutil.BoundMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Splitter;

/**
 * {@link LogisticRegressionWorker} defines logic to accumulate local <a
 * href=http://en.wikipedia.org/wiki/Logistic_regression >logistic regression</a> gradients.
 * 
 * <p>
 * At first iteration, wait for master to use the consistent initiating model.
 * 
 * <p>
 * At other iterations, workers include:
 * <ul>
 * <li>1. Update local model by using global model from last step..</li>
 * <li>2. Accumulate gradients by using local worker input data.</li>
 * <li>3. Send new local gradients to master by returning parameters.</li>
 * </ul>
 * 
 * <p>
 * L1 and l2 regulations are supported by configuration: RegularizedConstant in model params of ModelConfig.json.
 */
public class LogisticRegressionWorker
        extends
        AbstractWorkerComputable<LogisticRegressionParams, LogisticRegressionParams, GuaguaWritableAdapter<LongWritable>, GuaguaWritableAdapter<Text>> {

    private static final Logger LOG = LoggerFactory.getLogger(LogisticRegressionWorker.class);

    /**
     * Flat spot value to smooth lr derived function: result * (1 - result): This value sometimes may be close to zero.
     * Add flat sport to improve it: result * (1 - result) + 0.1d
     */
    private static final double FLAT_SPOT_VALUE = 0.1d;

    /**
     * Input column number
     */
    private int inputNum;

    /**
     * Output column number
     */
    private int outputNum;

    /**
     * Candidate column number
     */
    private int candidateNum;

    /**
     * Record count
     */
    private int count;

    /**
     * Testing data set.
     */
    private MemoryDiskList<Data> testingData;

    /**
     * Training data set.
     */
    private MemoryDiskList<Data> trainingData;

    /**
     * Local logistic regression model.
     */
    private double[] weights;

    /**
     * Model Config read from HDFS
     */
    private ModelConfig modelConfig;

    /**
     * Column Config list read from HDFS
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * A splitter to split data with specified delimiter.
     */
    private Splitter splitter = Splitter.on("|").trimResults();

    /**
     * PoissonDistribution which is used for poisson sampling for bagging with replacement.
     */
    protected PoissonDistribution rng = null;

    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        this.setRecordReader(new GuaguaLineRecordReader(fileSplit));
    }

    @Override
    public void init(WorkerContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        loadConfigFiles(context.getProps());
        int[] inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(this.columnConfigList);
        this.inputNum = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
        this.outputNum = inputOutputIndex[1];
        this.candidateNum = inputOutputIndex[2];
        if(this.inputNum == 0) {
            throw new IllegalStateException("No any variables are selected, please try variable select step firstly.");
        }
        this.rng = new PoissonDistribution(1.0d);
        double memoryFraction = Double.valueOf(context.getProps().getProperty("guagua.data.memoryFraction", "0.35"));
        String tmpFolder = context.getProps().getProperty("guagua.data.tmpfolder", "tmp");
        this.testingData = new MemoryDiskList<Data>((long) (Runtime.getRuntime().maxMemory() * memoryFraction),
                tmpFolder + File.separator + "test-" + System.currentTimeMillis());
        this.trainingData = new MemoryDiskList<Data>((long) (Runtime.getRuntime().maxMemory() * memoryFraction),
                tmpFolder + File.separator + "train-" + System.currentTimeMillis());
        // cannot find a good place to close these two data set, using Shutdown hook
        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            @Override
            public void run() {
                LogisticRegressionWorker.this.testingData.close();
                LogisticRegressionWorker.this.trainingData.close();
            }
        }));
    }

    @Override
    public LogisticRegressionParams doCompute(WorkerContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        if(context.isFirstIteration()) {
            return new LogisticRegressionParams();
        } else {
            this.weights = context.getLastMasterResult().getParameters();
            double[] gradients = new double[this.inputNum + 1];
            double trainingFinalError = 0.0d;
            double testingFinalError = 0.0d;
            long trainingSize = this.trainingData.size();
            long testingSize = this.testingData.size();
            this.trainingData.reOpen();
            for(Data data: trainingData) {
                double result = sigmoid(data.inputs, this.weights);
                double error = data.outputs[0] - result;
                trainingFinalError += caculateMSEError(error);
                for(int i = 0; i < gradients.length; i++) {
                    if(i < gradients.length - 1) {
                        // compute gradient for each weight, this is not like traditional LR (no derived function), with
                        // derived function, we see good convergence speed in our models.
                        // TODO extract function to provide traditional lr gradients and derived version for user to
                        // configure
                        gradients[i] += error * data.inputs[i] * (derivedFunction(result) + FLAT_SPOT_VALUE)
                                * data.getSignificance();
                    } else {
                        // for bias parameter, input is a constant 1d
                        gradients[i] += error * 1d * (derivedFunction(result) + FLAT_SPOT_VALUE)
                                * data.getSignificance();
                    }
                }
            }

            this.testingData.reOpen();
            // TODO here we should use current weights+gradients to compute testing error, so far it is for last error
            // computing.
            for(Data data: testingData) {
                double result = sigmoid(data.inputs, this.weights);
                double error = result - data.outputs[0];
                testingFinalError += caculateMSEError(error);
            }
            LOG.info("Iteration {} training data with error {}", context.getCurrentIteration(), trainingFinalError
                    / trainingSize);
            LOG.info("Iteration {} testing data with error {}", context.getCurrentIteration(), testingFinalError
                    / testingSize);
            return new LogisticRegressionParams(gradients, trainingFinalError, testingFinalError, trainingSize,
                    testingSize);
        }
    }

    /**
     * MSE value computation. We can provide more for user to configure in the future.
     */
    private double caculateMSEError(double error) {
        return error * error;
    }

    /**
     * Derived function for simmoid function.
     */
    private double derivedFunction(double result) {
        return result * (1d - result);
    }

    /**
     * Compute sigmoid value by dot operation of two vectors.
     */
    private double sigmoid(double[] inputs, double[] weights) {
        double value = 0.0d;
        for(int i = 0; i < inputs.length; i++) {
            value += weights[i] * inputs[i];
        }
        // append bias
        value += weights[inputs.length] * 1d;

        return 1.0d / (1.0d + BoundMath.exp(-1 * value));
    }

    @SuppressWarnings("unused")
    private double cost(double result, double output) {
        if(output == 1.0d) {
            return -Math.log(result);
        } else {
            return -Math.log(1 - result);
        }
    }

    @Override
    protected void postLoad(WorkerContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        this.trainingData.switchState();
        this.testingData.switchState();
        LOG.info("    - # Records of the Master Data Set: {}.", this.count);
        LOG.info("    - Bagging Sample Rate: {}.", this.modelConfig.getBaggingSampleRate());
        LOG.info("    - Bagging With Replacement: {}.", this.modelConfig.isBaggingWithReplacement());
        LOG.info("        - Cross Validation Rate: {}.", this.modelConfig.getCrossValidationRate());
        LOG.info("        - # Records of the Training Set: {}.", this.trainingData.size());
        LOG.info("        - # Records of the Validation Set: {}.", this.testingData.size());
    }

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        ++this.count;
        if((this.count) % 100000 == 0) {
            LOG.info("Read {} records.", this.count);
        }
        double baggingSampleRate = this.modelConfig.getBaggingSampleRate();
        // if fixInitialInput = false, we only compare random value with baggingSampleRate to avoid parsing data.
        // if fixInitialInput = true, we should use hashcode after parsing.
        if(!this.modelConfig.isFixInitialInput() && Double.compare(Math.random(), baggingSampleRate) >= 0) {
            return;
        }
        String line = currentValue.getWritable().toString();
        double[] inputData = new double[inputNum];
        double[] outputData = new double[outputNum];
        int index = 0, inputIndex = 0, outputIndex = 0;
        long hashcode = 0;
        double significance = CommonConstants.DEFAULT_SIGNIFICANCE_VALUE;
        for(String unit: splitter.split(line)) {
            double doubleValue = NumberFormatUtils.getDouble(unit.trim(), 0.0d);
            // no idea about why NaN in input data, we should process it as missing value TODO , according to norm type
            if(Double.isNaN(doubleValue)) {
                doubleValue = 0d;
            }
            if(index == this.columnConfigList.size()) {
                significance = NumberFormatUtils.getDouble(unit.trim(), 1.0d);
                break;
            } else {
                ColumnConfig columnConfig = this.columnConfigList.get(index);
                if(columnConfig != null && columnConfig.isTarget()) {
                    outputData[outputIndex++] = doubleValue;
                } else {
                    if(this.inputNum == this.candidateNum) {
                        // no variable selected, good candidate but not meta and not target choosed
                        if(!columnConfig.isMeta() && !columnConfig.isTarget()
                                && CommonUtils.isGoodCandidate(columnConfig)) {
                            inputData[inputIndex++] = doubleValue;
                            hashcode = hashcode * 31 + Double.valueOf(doubleValue).hashCode();
                        }
                    } else {
                        // final select some variables but meta and target are not included
                        if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                                && columnConfig.isFinalSelect()) {
                            inputData[inputIndex++] = doubleValue;
                            // only fixInitialInput=true, hashcode is effective. Remove Arrays.hashcode to avoid one
                            // iteration for the input columns. Last weight column should be excluded.
                            hashcode = hashcode * 31 + Double.valueOf(doubleValue).hashCode();
                        }
                    }
                }
            }
            index += 1;
        }
        this.addDataPairToDataSet(hashcode, new Data(inputData, outputData, significance));
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

    /**
     * Add data pair to data set according to setting parameters. Still set hashCode to long to make double and long
     * friendly.
     */
    private void addDataPairToDataSet(long hashcode, Data record) {
        double crossValidationRate = this.modelConfig.getCrossValidationRate();
        if(this.modelConfig.isFixInitialInput()) {
            long longCrossValidation = Double.valueOf(crossValidationRate * 100).longValue();
            if(hashcode % 100 < longCrossValidation) {
                this.testingData.append(record);
            } else {
                this.trainingData.append(record);
            }
        } else {
            double random = Math.random();
            if(this.modelConfig.isBaggingWithReplacement()) {
                int count = rng.sample();
                if(count > 0) {
                    record.setSignificance(record.significance * count);
                    if(Double.compare(random, crossValidationRate) < 0) {
                        this.testingData.append(record);
                    } else {
                        this.trainingData.append(record);
                    }
                }
            } else {
                addDataPairToDataSet(record, crossValidationRate, random);
            }
        }
    }

    /**
     * Add data pair to data set according to random number compare with crossValidationRate.
     */
    private void addDataPairToDataSet(Data record, double crossValidationRate, double random) {
        if(Double.compare(random, crossValidationRate) < 0) {
            this.testingData.append(record);
        } else {
            this.trainingData.append(record);
        }
    }

    private static class Data implements Serializable {

        private static final long serialVersionUID = 903201066309036170L;

        private double significance;
        private final double[] inputs;
        private final double[] outputs;

        public Data(double[] inputs, double[] outputs, double significance) {
            this.inputs = inputs;
            this.outputs = outputs;
            this.significance = significance;
        }

        /**
         * @return the significance
         */
        public double getSignificance() {
            return significance;
        }

        /**
         * @param significance
         *            the significance to set
         */
        public void setSignificance(double significance) {
            this.significance = significance;
        }

    }

}
