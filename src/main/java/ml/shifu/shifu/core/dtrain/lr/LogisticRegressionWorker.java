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

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

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
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
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
     * sampled input record size.
     */
    protected long sampleCount;

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

    /**
     * PoissonDistribution which is used for up sampleing positive records.
     */
    protected PoissonDistribution upSampleRng = null;
    
    private final Data endMark = new Data(null,null,0);
    
    private int threadCount = 1;

    protected boolean isUpSampleEnabled() {
        return this.upSampleRng != null;
    }

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
        Double upSampleWeight = modelConfig.getTrain().getUpSampleWeight();
        if(Double.compare(upSampleWeight, 1d) != 0) {
            // set mean to upSampleWeight -1 and get sample + 1to make sure no zero sample value
            LOG.info("Enable up sampling with weight {}.", upSampleWeight);
            this.upSampleRng = new PoissonDistribution(upSampleWeight - 1);
        }
        double memoryFraction = Double.valueOf(context.getProps().getProperty("guagua.data.memoryFraction", "0.6"));
        LOG.info("Max heap memory: {}, fraction: {}", Runtime.getRuntime().maxMemory(), memoryFraction);
        double crossValidationRate = this.modelConfig.getCrossValidationRate();
        String tmpFolder = context.getProps().getProperty("guagua.data.tmpfolder", "tmp");
        this.trainingData = new MemoryDiskList<Data>(
                (long) (Runtime.getRuntime().maxMemory() * memoryFraction * (1 - crossValidationRate)), tmpFolder
                        + File.separator + "train-" + System.currentTimeMillis());
        this.testingData = new MemoryDiskList<Data>(
                (long) (Runtime.getRuntime().maxMemory() * memoryFraction * crossValidationRate), tmpFolder
                        + File.separator + "test-" + System.currentTimeMillis());
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
            LinkedBlockingQueue<Data> trainingDataQ = new LinkedBlockingQueue<Data>();
            LinkedBlockingQueue<Data> testingDataQ = new LinkedBlockingQueue<Data>();
            List<DataWorker> trainingWorkers = new ArrayList<DataWorker>();
            List<DataWorker> testingWorkers = new ArrayList<DataWorker>();
            ExecutorService executor = Executors.newFixedThreadPool(threadCount);

            try {
                for(int i = 0; i < threadCount; i++) {
                    DataWorker dw = new DataWorker(trainingDataQ, true);
                    trainingWorkers.add(dw);
                    executor.execute(dw);
                }
                this.trainingData.reOpen();
                for(Data data: trainingData) {
                    trainingDataQ.put(data);
                }
                trainingDataQ.put(endMark);
                // TODO here we should use current weights+gradients to compute testing error, so far it is for last
                // error
                // computing.
                for(int i = 0; i < threadCount; i++) {
                    DataWorker dw = new DataWorker(testingDataQ, false);
                    testingWorkers.add(dw);
                    executor.execute(dw);
                }
                this.testingData.reOpen();
                for(Data data: testingData) {
                    testingDataQ.put(data);
                }
                testingDataQ.put(endMark);
                
                executor.shutdown();
                executor.awaitTermination(60, TimeUnit.SECONDS);

                for(DataWorker dw:trainingWorkers){
                    trainingFinalError+=dw.finalError;
                    for(int i = 0; i < gradients.length; i++) {
                        gradients[i]+= dw.gradients[i];
                    }
                }
                for(DataWorker dw:testingWorkers){
                    testingFinalError+=dw.finalError;
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
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

        // if fixInitialInput = true, we should use hashcode to sample.
        long longBaggingSampleRate = Double.valueOf(baggingSampleRate * 100).longValue();
        if(modelConfig.isFixInitialInput() && hashcode % 100 >= longBaggingSampleRate) {
            return;
        }

        this.sampleCount += 1;

        if(modelConfig.isBinaryClassification() && isUpSampleEnabled() && Double.compare(outputData[0], 1d) == 0) {
            // Double.compare(ideal[0], 1d) == 0 means positive tags; sample + 1 to avoid sample count to 0
            significance *= (this.upSampleRng.sample() + 1);
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
    
    class DataWorker implements Runnable {
        private LinkedBlockingQueue<Data> queue;
        private boolean isTraining = true;
        double[] gradients = new double[inputNum + 1];
        double finalError = 0.0d;
        

        public DataWorker(LinkedBlockingQueue<Data> queue, boolean isTraining) {
            this.queue = queue;
            this.isTraining = isTraining;
        }

        public void run() {
            Data data = null;
            try {
                while((data = queue.take()) != endMark) {
                    double result = sigmoid(data.inputs, weights);
                    double error = data.outputs[0] - result;
                    finalError += caculateMSEError(error);
                    if(isTraining) {
                        for(int i = 0; i < gradients.length; i++) {
                            if(i < gradients.length - 1) {
                                // compute gradient for each weight, this is not like traditional LR (no derived
                                // function),
                                // with
                                // derived function, we see good convergence speed in our models.
                                // TODO extract function to provide traditional lr gradients and derived version for
                                // user to
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
                }
                queue.put(endMark);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

    }

}
