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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import ml.shifu.guagua.ComputableMonitor;
import ml.shifu.guagua.hadoop.io.GuaguaLineRecordReader;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.Bytable;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.util.BytableMemoryDiskList;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.guagua.worker.AbstractWorkerComputable;
import ml.shifu.guagua.worker.WorkerContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.lang.StringUtils;
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
@ComputableMonitor(timeUnit = TimeUnit.SECONDS, duration = 300)
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
    private BytableMemoryDiskList<Data> validationData;

    /**
     * Training data set.
     */
    private BytableMemoryDiskList<Data> trainingData;

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

    /**
     * Indicates if there are cross validation data sets.
     */
    protected boolean isSpecificValidation = false;

    /**
     * If stratified sampling or random sampling
     */
    protected boolean isStratifiedSampling = false;

    /**
     * Positive count in training data list, only be effective in 0-1 regression or onevsall classification
     */
    protected long positiveTrainCount;

    /**
     * Positive count in training data list and being selected in training, only be effective in 0-1 regression or
     * onevsall classification
     */
    protected long positiveSelectedTrainCount;

    /**
     * Negative count in training data list , only be effective in 0-1 regression or onevsall classification
     */
    protected long negativeTrainCount;

    /**
     * Negative count in training data list and being selected, only be effective in 0-1 regression or onevsall
     * classification
     */
    protected long negativeSelectedTrainCount;

    /**
     * Positive count in validation data list, only be effective in 0-1 regression or onevsall classification
     */
    protected long positiveValidationCount;

    /**
     * Negative count in validation data list, only be effective in 0-1 regression or onevsall classification
     */
    protected long negativeValidationCount;

    /**
     * PoissonDistribution which is used for poission sampling for bagging with replacement.
     */

    protected Map<Integer, PoissonDistribution> baggingRngMap = new HashMap<Integer, PoissonDistribution>();

    /**
     * Construct a bagging random map for different classes. For stratified sampling, this is useful for each class
     * sampling.
     */
    protected Map<Integer, Random> baggingRandomMap = new HashMap<Integer, Random>();

    /**
     * Construct a validation random map for different classes. For stratified sampling, this is useful for each class
     * sampling.
     */
    protected Map<Integer, Random> validationRandomMap = new HashMap<Integer, Random>();

    /**
     * Trainer id used to tag bagging training job, starting from 0, 1, 2 ...
     */
    private Integer trainerId;

    /**
     * If k-fold cross validation
     */
    private boolean isKFoldCV;

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
        int[] inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(modelConfig.getNormalizeType(), this.columnConfigList);
        this.inputNum = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
        this.outputNum = inputOutputIndex[1];
        this.candidateNum = inputOutputIndex[2];
        this.isSpecificValidation = (modelConfig.getValidationDataSetRawPath() != null && !"".equals(modelConfig
                .getValidationDataSetRawPath()));
        this.isStratifiedSampling = this.modelConfig.getTrain().getStratifiedSample();
        this.trainerId = Integer.valueOf(context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID, "0"));
        Integer kCrossValidation = this.modelConfig.getTrain().getNumKFold();
        if(kCrossValidation != null && kCrossValidation > 0) {
            isKFoldCV = true;
        }

        if(this.inputNum == 0) {
            throw new IllegalStateException("No any variables are selected, please try variable select step firstly.");
        }
        this.rng = new PoissonDistribution(1.0d);
        Double upSampleWeight = modelConfig.getTrain().getUpSampleWeight();
        if(Double.compare(upSampleWeight, 1d) != 0) {
            // set mean to upSampleWeight -1 and get sample + 1 to make sure no zero sample value
            LOG.info("Enable up sampling with weight {}.", upSampleWeight);
            this.upSampleRng = new PoissonDistribution(upSampleWeight - 1);
        }
        double memoryFraction = Double.valueOf(context.getProps().getProperty("guagua.data.memoryFraction", "0.6"));
        LOG.info("Max heap memory: {}, fraction: {}", Runtime.getRuntime().maxMemory(), memoryFraction);
        double crossValidationRate = this.modelConfig.getValidSetRate();
        String tmpFolder = context.getProps().getProperty("guagua.data.tmpfolder", "tmp");

        if(StringUtils.isNotBlank(modelConfig.getValidationDataSetRawPath())) {
            // fixed 0.6 and 0.4 of max memory for trainingData and validationData
            this.trainingData = new BytableMemoryDiskList<Data>((long) (Runtime.getRuntime().maxMemory()
                    * memoryFraction * 0.6), tmpFolder + File.separator + "train-" + System.currentTimeMillis(),
                    Data.class.getName());
            this.validationData = new BytableMemoryDiskList<Data>((long) (Runtime.getRuntime().maxMemory()
                    * memoryFraction * 0.4), tmpFolder + File.separator + "test-" + System.currentTimeMillis(),
                    Data.class.getName());
        } else {
            this.trainingData = new BytableMemoryDiskList<Data>((long) (Runtime.getRuntime().maxMemory()
                    * memoryFraction * (1 - crossValidationRate)), tmpFolder + File.separator + "train-"
                    + System.currentTimeMillis(), Data.class.getName());
            this.validationData = new BytableMemoryDiskList<Data>((long) (Runtime.getRuntime().maxMemory()
                    * memoryFraction * crossValidationRate), tmpFolder + File.separator + "test-"
                    + System.currentTimeMillis(), Data.class.getName());
        }
        // cannot find a good place to close these two data set, using Shutdown hook
        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            @Override
            public void run() {
                LogisticRegressionWorker.this.validationData.close();
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
            long testingSize = this.validationData.size();
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

            this.validationData.reOpen();
            // TODO here we should use current weights+gradients to compute testing error, so far it is for last error
            // computing.
            for(Data data: validationData) {
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
    private double sigmoid(float[] inputs, double[] weights) {
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
        if(validationData != null) {
            this.validationData.switchState();
        }
        LOG.info("    - # Records of the Total Data Set: {}.", this.count);
        LOG.info("    - Bagging Sample Rate: {}.", this.modelConfig.getBaggingSampleRate());
        LOG.info("    - Bagging With Replacement: {}.", this.modelConfig.isBaggingWithReplacement());
        if(this.isKFoldCV) {
            LOG.info("        - Validation Rate(kFold): {}.", 1d / this.modelConfig.getTrain().getNumKFold());
        } else {
            LOG.info("        - Validation Rate: {}.", this.modelConfig.getValidSetRate());
        }
        LOG.info("        - # Records of the Training Set: {}.", this.trainingData.size());
        if(modelConfig.isRegression() || modelConfig.getTrain().isOneVsAll()) {
            LOG.info("        - # Positive Bagging Selected Records of the Training Set: {}.",
                    this.positiveSelectedTrainCount);
            LOG.info("        - # Negative Bagging Selected Records of the Training Set: {}.",
                    this.negativeSelectedTrainCount);
            LOG.info("        - # Positive Raw Records of the Training Set: {}.", this.positiveTrainCount);
            LOG.info("        - # Negative Raw Records of the Training Set: {}.", this.negativeTrainCount);
        }

        if(validationData != null) {
            LOG.info("        - # Records of the Validation Set: {}.", this.validationData.size());
            if(modelConfig.isRegression() || modelConfig.getTrain().isOneVsAll()) {
                LOG.info("        - # Positive Records of the Validation Set: {}.", this.positiveValidationCount);
                LOG.info("        - # Negative Records of the Validation Set: {}.", this.negativeValidationCount);
            }
        }
    }

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        ++this.count;
        if((this.count) % 100000 == 0) {
            LOG.info("Read {} records.", this.count);
        }
        String line = currentValue.getWritable().toString();
        float[] inputData = new float[inputNum];
        float[] outputData = new float[outputNum];
        int index = 0, inputIndex = 0, outputIndex = 0;
        long hashcode = 0;
        double significance = CommonConstants.DEFAULT_SIGNIFICANCE_VALUE;
        boolean hasCandidates = CommonUtils.hasCandidateColumns(this.columnConfigList);

        for(String unit: splitter.split(line)) {
            // check here to avoid bad performance in failed NumberFormatUtils.getFloat(input, 0f)
            float floatValue = unit.length() == 0 ? 0f : NumberFormatUtils.getFloat(unit, 0f);
            // no idea about why NaN in input data, we should process it as missing value TODO , according to norm type
            floatValue = (Float.isNaN(floatValue) || Double.isNaN(floatValue)) ? 0f : floatValue;

            if(index == this.columnConfigList.size()) {
                // do we need to check if not weighted directly set to 1f; if such logic non-weight at first, then
                // weight, how to process???
                if(StringUtils.isBlank(modelConfig.getWeightColumnName())) {
                    significance = 1d;
                    // break here if we reach weight column which is last column
                    break;
                }

                // check here to avoid bad performance in failed NumberFormatUtils.getDouble(input, 1)
                significance = unit.length() == 0 ? 1f : NumberFormatUtils.getDouble(unit, 1d);
                // if invalid weight, set it to 1f and warning in log
                if(Double.compare(significance, 0d) < 0) {
                    LOG.warn("The {} record in current worker weight {} is less than 0f, it is invalid, set it to 1.",
                            count, significance);
                    significance = 1d;
                }
                // the last field is significance, break here
                break;
            } else {
                ColumnConfig columnConfig = this.columnConfigList.get(index);
                if(columnConfig != null && columnConfig.isTarget()) {
                    outputData[outputIndex++] = floatValue;
                } else {
                    if(this.inputNum == this.candidateNum) {
                        // no variable selected, good candidate but not meta and not target choosed
                        if(!columnConfig.isMeta() && !columnConfig.isTarget()
                                && CommonUtils.isGoodCandidate(columnConfig, hasCandidates)) {
                            inputData[inputIndex++] = floatValue;
                            hashcode = hashcode * 31 + Float.valueOf(floatValue).hashCode();
                        }
                    } else {
                        // final select some variables but meta and target are not included
                        if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                                && columnConfig.isFinalSelect()) {
                            inputData[inputIndex++] = floatValue;
                            // only fixInitialInput=true, hashcode is effective. Remove Arrays.hashcode to avoid one
                            // iteration for the input columns. Last weight column should be excluded.
                            hashcode = hashcode * 31 + Float.valueOf(floatValue).hashCode();
                        }
                    }
                }
            }
            index += 1;
        }

        // sample negative only logic here
        if(modelConfig.getTrain().getSampleNegOnly()) {
            if(this.modelConfig.isFixInitialInput()) {
                // if fixInitialInput, sample hashcode in 1-sampleRate range out if negative records
                int startHashCode = (100 / this.modelConfig.getBaggingNum()) * this.trainerId;
                // here BaggingSampleRate means how many data will be used in training and validation, if it is 0.8, we
                // should take 1-0.8 to check endHashCode
                int endHashCode = startHashCode
                        + Double.valueOf((1d - this.modelConfig.getBaggingSampleRate()) * 100).intValue();
                if((modelConfig.isRegression() || (modelConfig.isClassification() && modelConfig.getTrain()
                        .isOneVsAll())) // regression or onevsall
                        && (int) (outputData[0] + 0.01d) == 0 // negative record
                        && isInRange(hashcode, startHashCode, endHashCode)) {
                    return;
                }
            } else {
                // if not fixed initial input, and for regression or onevsall multiple classification (regression also).
                // if negative record
                if((modelConfig.isRegression() || (modelConfig.isClassification() && modelConfig.getTrain()
                        .isOneVsAll())) // regression or onevsall
                        && (int) (outputData[0] + 0.01d) == 0 // negative record
                        && Double.compare(Math.random(), this.modelConfig.getBaggingSampleRate()) >= 0) {
                    return;
                }
            }
        }

        Data data = new Data(inputData, outputData, significance);

        // up sampling logic, just add more weights while bagging sampling rate is still not changed
        if(modelConfig.isRegression() && isUpSampleEnabled() && Double.compare(outputData[0], 1d) == 0) {
            // Double.compare(ideal[0], 1d) == 0 means positive tags; sample + 1 to avoids sample count to 0
            data.setSignificance(data.significance * (this.upSampleRng.sample() + 1));
        }

        boolean isValidation = false;
        if(context.getAttachment() != null && context.getAttachment() instanceof Boolean) {
            isValidation = (Boolean) context.getAttachment();
        }

        boolean isInTraining = addDataPairToDataSet(hashcode, data, isValidation);

        // do bagging sampling only for training data
        if(isInTraining) {
            float subsampleWeights = sampleWeights(outputData[0]);
            if(isPositive(outputData[0])) {
                this.positiveSelectedTrainCount += subsampleWeights * 1L;
            } else {
                this.negativeSelectedTrainCount += subsampleWeights * 1L;
            }
            // set weights to significance, if 0, significance will be 0, that is bagging sampling
            data.setSignificance(data.significance * subsampleWeights);
        } else {
            // for validation data, according bagging sampling logic, we may need to sampling validation data set, while
            // validation data set are only used to compute validation error, not to do real sampling is ok.
        }
    }

    protected float sampleWeights(float label) {
        float sampleWeights = 1f;
        // sample negative or kFoldCV, sample rate is 1d
        double sampleRate = (modelConfig.getTrain().getSampleNegOnly() || this.isKFoldCV) ? 1d : modelConfig.getTrain()
                .getBaggingSampleRate();
        int classValue = (int) (label + 0.01f);
        if(!modelConfig.isBaggingWithReplacement()) {
            Random random = null;
            if(this.isStratifiedSampling) {
                random = baggingRandomMap.get(classValue);
                if(random == null) {
                    random = DTrainUtils.generateRandomBySampleSeed(modelConfig.getTrain().getBaggingSampleSeed(),
                            CommonConstants.NOT_CONFIGURED_BAGGING_SEED);
                    baggingRandomMap.put(classValue, random);
                }
            } else {
                random = baggingRandomMap.get(0);
                if(random == null) {
                    random = DTrainUtils.generateRandomBySampleSeed(modelConfig.getTrain().getBaggingSampleSeed(),
                            CommonConstants.NOT_CONFIGURED_BAGGING_SEED);
                    baggingRandomMap.put(0, random);
                }
            }
            if(random.nextDouble() <= sampleRate) {
                sampleWeights = 1f;
            } else {
                sampleWeights = 0f;
            }
        } else {
            // bagging with replacement sampling in training data set, take PoissonDistribution for sampling with
            // replacement
            if(this.isStratifiedSampling) {
                PoissonDistribution rng = this.baggingRngMap.get(classValue);
                if(rng == null) {
                    rng = new PoissonDistribution(sampleRate);
                    this.baggingRngMap.put(classValue, rng);
                }
                sampleWeights = rng.sample();
            } else {
                PoissonDistribution rng = this.baggingRngMap.get(0);
                if(rng == null) {
                    rng = new PoissonDistribution(sampleRate);
                    this.baggingRngMap.put(0, rng);
                }
                sampleWeights = rng.sample();
            }
        }
        return sampleWeights;
    }

    private void loadConfigFiles(final Properties props) {
        try {
            SourceType sourceType = SourceType.valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE,
                    SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);
            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    props.getProperty(CommonConstants.SHIFU_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    protected boolean isPositive(float value) {
        return Float.compare(1f, value) == 0 ? true : false;
    }

    /**
     * Add to training set or validation set according to validation rate.
     * 
     * @param hashcode
     *            the hash code of the data
     * @param data
     *            data instance
     * @param isValidation
     *            if it is validation
     * @return if in training, training is true, others are false.
     */
    protected boolean addDataPairToDataSet(long hashcode, Data data, boolean isValidation) {
        if(this.isKFoldCV) {
            int k = this.modelConfig.getTrain().getNumKFold();
            if(hashcode % k == this.trainerId) {
                this.validationData.append(data);
                if(isPositive(data.outputs[0])) {
                    this.positiveValidationCount += 1L;
                } else {
                    this.negativeValidationCount += 1L;
                }
                return false;
            } else {
                this.trainingData.append(data);
                if(isPositive(data.outputs[0])) {
                    this.positiveTrainCount += 1L;
                } else {
                    this.negativeTrainCount += 1L;
                }
                return true;
            }
        }

        if(this.isSpecificValidation) {
            if(isValidation) {
                this.validationData.append(data);
                if(isPositive(data.outputs[0])) {
                    this.positiveValidationCount += 1L;
                } else {
                    this.negativeValidationCount += 1L;
                }
                return false;
            } else {
                this.trainingData.append(data);
                if(isPositive(data.outputs[0])) {
                    this.positiveTrainCount += 1L;
                } else {
                    this.negativeTrainCount += 1L;
                }
                return true;
            }
        } else {
            if(Double.compare(this.modelConfig.getValidSetRate(), 0d) != 0) {
                int classValue = (int) (data.outputs[0] + 0.01f);
                Random random = null;
                if(this.isStratifiedSampling) {
                    // each class use one random instance
                    random = validationRandomMap.get(classValue);
                    if(random == null) {
                        random = new Random();
                        this.validationRandomMap.put(classValue, random);
                    }
                } else {
                    // all data use one random instance
                    random = validationRandomMap.get(0);
                    if(random == null) {
                        random = new Random();
                        this.validationRandomMap.put(0, random);
                    }
                }

                if(this.modelConfig.isFixInitialInput()) {
                    // for fix initial input, if hashcode%100 is in [start-hashcode, end-hashcode), validation,
                    // otherwise training. start hashcode in different job is different to make sure bagging jobs have
                    // different data. if end-hashcode is over 100, then check if hashcode is in [start-hashcode, 100]
                    // or [0, end-hashcode]
                    int startHashCode = (100 / this.modelConfig.getBaggingNum()) * this.trainerId;
                    int endHashCode = startHashCode
                            + Double.valueOf(this.modelConfig.getValidSetRate() * 100).intValue();
                    if(isInRange(hashcode, startHashCode, endHashCode)) {
                        this.validationData.append(data);
                        if(isPositive(data.outputs[0])) {
                            this.positiveValidationCount += 1L;
                        } else {
                            this.negativeValidationCount += 1L;
                        }
                        return false;
                    } else {
                        this.trainingData.append(data);
                        if(isPositive(data.outputs[0])) {
                            this.positiveTrainCount += 1L;
                        } else {
                            this.negativeTrainCount += 1L;
                        }
                        return true;
                    }
                } else {
                    // not fixed initial input, if random value >= validRate, training, otherwise validation.
                    if(random.nextDouble() >= this.modelConfig.getValidSetRate()) {
                        this.trainingData.append(data);
                        if(isPositive(data.outputs[0])) {
                            this.positiveTrainCount += 1L;
                        } else {
                            this.negativeTrainCount += 1L;
                        }
                        return true;
                    } else {
                        this.validationData.append(data);
                        if(isPositive(data.outputs[0])) {
                            this.positiveValidationCount += 1L;
                        } else {
                            this.negativeValidationCount += 1L;
                        }
                        return false;
                    }
                }
            } else {
                this.trainingData.append(data);
                if(isPositive(data.outputs[0])) {
                    this.positiveTrainCount += 1L;
                } else {
                    this.negativeTrainCount += 1L;
                }
                return true;
            }
        }
    }

    private boolean isInRange(long hashcode, int startHashCode, int endHashCode) {
        // check if in [start, end] or if in [start, 100) and [0, end-100)
        int hashCodeIn100 = (int) hashcode % 100;
        if(endHashCode <= 100) {
            // in range [start, end)
            return hashCodeIn100 >= startHashCode && hashCodeIn100 < endHashCode;
        } else {
            // in range [start, 100) or [0, endHashCode-100)
            return hashCodeIn100 >= startHashCode || hashCodeIn100 < (endHashCode % 100);
        }
    }

    private static class Data implements Bytable {

        private double significance;
        private float[] inputs;
        private float[] outputs;

        public Data(float[] inputs, float[] outputs, double significance) {
            this.inputs = inputs;
            this.outputs = outputs;
            this.significance = significance;
        }

        @SuppressWarnings("unused")
        public Data() {
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

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeDouble(significance);
            out.writeInt(inputs.length);
            out.writeInt(outputs.length);
            for(int i = 0; i < inputs.length; i++) {
                out.writeFloat(inputs[i]);
            }
            for(int i = 0; i < outputs.length; i++) {
                out.writeFloat(outputs[i]);
            }
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            this.significance = in.readDouble();
            int inputsLen = in.readInt();
            int outputsLen = in.readInt();
            this.inputs = new float[inputsLen];
            this.outputs = new float[outputsLen];
            for(int i = 0; i < inputsLen; i++) {
                inputs[i] = in.readFloat();
            }
            for(int i = 0; i < outputsLen; i++) {
                outputs[i] = in.readFloat();
            }
        }
    }

}