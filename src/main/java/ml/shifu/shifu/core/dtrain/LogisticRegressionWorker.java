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
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.lang.math.RandomUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Splitter;

/**
 * {@link LogisticRegressionWorker} defines logic to accumulate local <a
 * href=http://en.wikipedia.org/wiki/Logistic_regression >logistic
 * regression</a> gradients.
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
 */
public class LogisticRegressionWorker
        extends
        AbstractWorkerComputable<LogisticRegressionParams, LogisticRegressionParams, GuaguaWritableAdapter<LongWritable>, GuaguaWritableAdapter<Text>> {

    private static final Logger LOG = LoggerFactory.getLogger(LogisticRegressionWorker.class);

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
    
    
    private double regularizedRate = 0.0d;

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
    private Splitter splitter = Splitter.on("|");

    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        this.setRecordReader(new GuaguaLineRecordReader(fileSplit));
    }

    @Override
    public void init(WorkerContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        loadConfigFiles(context.getProps());
        // this.inputNum =
        // NumberFormatUtils.getInt(LogisticRegressionContants.LR_INPUT_NUM,
        // LogisticRegressionContants.LR_INPUT_DEFAULT_NUM);
        int[] inputOutputIndex = NNUtils.getInputOutputCandidateCounts(this.columnConfigList);
        this.inputNum = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
        this.outputNum = 1;
        this.candidateNum = inputOutputIndex[2];
        this.regularizedRate = Double.valueOf(this.modelConfig.getParams().get(LogisticRegressionContants.LR_REGULARIZED_RATE).toString());
        LOG.info("regularizedRate:" + this.regularizedRate);
        LOG.info("inputNum:" + this.inputNum);
        double memoryFraction = Double.valueOf(context.getProps().getProperty("guagua.data.memoryFraction", "0.5"));
        String tmpFolder = context.getProps().getProperty("guagua.data.tmpfolder", "tmp");
        this.testingData = new MemoryDiskList<Data>((long) (Runtime.getRuntime().maxMemory() * memoryFraction),
                tmpFolder + File.separator + System.currentTimeMillis());
        this.trainingData = new MemoryDiskList<Data>((long) (Runtime.getRuntime().maxMemory() * memoryFraction),
                tmpFolder + File.separator + System.currentTimeMillis());
        // cannot find a good place to close these two data set, using Shutdown
        // hook
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
            double[] gradients = new double[this.inputNum];
            double trainingFinalError = 0.0d;
            double testingFinalError = 0.0d;
            int trainingSize = (int)this.trainingData.size();
            int testingSize = (int)this.testingData.size();
            LOG.info("training_size:" + trainingSize);
            LOG.info("testing_size:" + testingSize);
            this.trainingData.reOpen();
            for(Data data: trainingData) {
                double result = sigmoid(data.inputs, this.weights);
                double error = result - data.outputs[0];
                trainingFinalError += cost(result, data.outputs[0]);
                for(int i = 0; i < gradients.length; i++) {
                    gradients[i] += error * data.inputs[i] * data.significance;
                }
            }
            
            this.testingData.reOpen();
            for(Data data: testingData) {
                double result = sigmoid(data.inputs, this.weights);
                testingFinalError += cost(result, data.outputs[0]);
            }
            double trainingReg = this.regularizedParameter(this.regularizedRate, trainingSize);
            double testingReg = this.regularizedParameter(this.regularizedRate, testingSize);
            LOG.info("training_finish_final:" + trainingFinalError);
            LOG.info("Iteration {} training data with error {}", context.getCurrentIteration(), trainingFinalError / trainingSize+trainingReg);
            LOG.info("Iteration {} testing data with error {}", context.getCurrentIteration(), testingFinalError / testingSize+testingReg);
            return new LogisticRegressionParams(gradients, trainingFinalError / trainingSize+trainingReg,trainingSize);
        }
    }

    /**
     * Compute sigmoid value by dot operation of two vectors.
     */
    private double sigmoid(double[] inputs, double[] weights) {
        double value = 0.0d;
        for(int i = 0; i < weights.length; i++) {
            value += weights[i] * inputs[i];
        }
        return 1.0d / (1.0d + Math.exp(-value));
    }

    private double cost(double result, double output) {
        if(output == 1.0d) {
            return -Math.log(result);
        } else {
            return -Math.log(1 - result);
        }
    }
    
    private double regularizedParameter(double regularizedRate,int recordCount){
        if(regularizedRate == 0.0d){
            return 0.0d;
        }
        double sumSqureWeights = 0.0d;
        for(int i = 0;i<this.weights.length;i++){
            sumSqureWeights+= this.weights[i]*this.weights[i];
        }
        double result = regularizedRate*sumSqureWeights/recordCount*0.5d;
        return result;
    }

    @Override
    protected void postLoad(WorkerContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        this.trainingData.switchState();
        this.testingData.switchState();
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
        int count = 0, inputIndex = 0, outputIndex = 0;
        long hashcode = 0;
        double significance = CommonConstants.DEFAULT_SIGNIFICANCE_VALUE;
        for(String unit: splitter.split(line)) {
            double doubleValue = NumberFormatUtils.getDouble(unit.trim(), 0.0d);
            if(count == this.columnConfigList.size()) {
                significance = NumberFormatUtils.getDouble(unit.trim(), CommonConstants.DEFAULT_SIGNIFICANCE_VALUE);
                break;
            } else {
                ColumnConfig columnConfig = this.columnConfigList.get(count);
                if(columnConfig != null && columnConfig.isTarget()) {
                    if(doubleValue != 1.0d && doubleValue != 0d) {
                        throw new ShifuException(ShifuErrorCode.ERROR_INVALID_TARGET_VALUE);
                    }
                    outputData[outputIndex++] = doubleValue;
                } else {
                    if(this.inputNum == this.candidateNum) {
                        // all variables are not set final-selectByFilter
                        if(CommonUtils.isGoodCandidate(columnConfig)) {
                            inputData[inputIndex++] = doubleValue;
                            hashcode = hashcode * 31 + Double.valueOf(doubleValue).hashCode();
                        }
                    } else {
                        // final select some variables
                        if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                                && columnConfig.isFinalSelect()) {
                            inputData[inputIndex++] = doubleValue;
                            // only fixInitialInput=true, hashcode is effective.
                            // Remove Arrays.hashcode to avoid one
                            // iteration for the input columns. Last weight
                            // column should be excluded.
                            hashcode = hashcode * 31 + Double.valueOf(doubleValue).hashCode();
                        }
                    }
                }
            }
            count++;
        }
        this.addDataPairToDataSet(hashcode,new Data(inputData, outputData, significance));
       // this.dataList.append(new Data(inputData, outputData, significance));
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
//            if(isBaggingReplacementTrigged(random)) {
//                mockRandomRepeatData(crossValidationRate, random);
//            } else {
                addDataPairToDataSet(record, crossValidationRate, random);
           // }
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

    
    /**
     * Only baggingWithReplacement is set and size over NNConstants.NN_BAGGING_THRESHOLD, and random value <= 1/size. We
     * choose use existing data to add training data set and testing data set.
     */
    private boolean isBaggingReplacementTrigged(double random) {
        long trainingSize = this.trainingData.size();
        long testingSize = this.testingData.size();
        // size should be equals to sampleCount:)
        long size = trainingSize + testingSize;
        return this.modelConfig.isBaggingWithReplacement() && (testingSize > 0) && (trainingSize > 0)
                && (size > NNConstants.NN_BAGGING_THRESHOLD)
                && (Double.compare(random, 0.5d) < 0);
    }

    /**
     * From Trainer, the logic is to random choose items in master dataset, but I don't want to load data twice for
     * saving memory. Use this to mock raw random repeat logic. This should be some logic difference because of data are
     * not loaded into data set, not random.
     */
//    private void mockRandomRepeatData(double crossValidationRate, double random) {
//        long trainingSize = this.trainingData.getRecordCount();
//        long testingSize = this.testingData.getRecordCount();
//        long size = trainingSize + testingSize;
//        // here we used a strong cast from long to int since it's just a random choosing algorithm
//        int next = RandomUtils.nextInt((int) size);
//        MLDataPair dataPair = new BasicMLDataPair(new BasicMLData(new double[this.inputNodeCount]), new BasicMLData(
//                new double[this.outputNodeCount]));
//        if(next >= trainingSize) {
//            this.testingData.getRecord(next - trainingSize, dataPair);
//        } else {
//            this.trainingData.getRecord(next, dataPair);
//        }
//
//        if(Double.compare(random, crossValidationRate) < 0) {
//            this.testingData.add(dataPair);
//        } else {
//            this.trainingData.add(dataPair);
//        }
//    }

    private static class Data implements Serializable {

        private static final long serialVersionUID = 903201066309036170L;

        public Data(double[] inputs, double[] outputs, double significance) {
            this.inputs = inputs;
            this.outputs = outputs;
            this.significance = significance;
        }

        private final double significance;
        private final double[] inputs;
        private final double[] outputs;
    }

    public void test(String input) {
        int count = 0;
        for(String unit: splitter.split(input)) {
            System.out.println("count:" + count);
            System.out.println("current_value:" + unit);
            count++;

        }
    }

    public static void main(String[] args) {
        String line = "|1|1.126082|-1.979292|1.307047|1.018816|1.555019|3.408517|2.816882|2.619519|2.274134|2.31";
        LogisticRegressionWorker w = new LogisticRegressionWorker();
        // w.test(line);
        w.cost(1, 1);

    }

}
