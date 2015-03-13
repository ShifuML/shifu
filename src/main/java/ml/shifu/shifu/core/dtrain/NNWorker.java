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

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Properties;

import ml.shifu.guagua.GuaguaRuntimeException;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.mapreduce.GuaguaLineRecordReader;
import ml.shifu.guagua.mapreduce.GuaguaWritableAdapter;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.guagua.worker.AbstractWorkerComputable;
import ml.shifu.guagua.worker.WorkerContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.lang.math.RandomUtils;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.data.buffer.BufferedMLDataSet;
import org.encog.neural.error.LinearErrorFunction;
import org.encog.neural.flat.FlatNetwork;
import org.encog.neural.networks.BasicNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Splitter;

/**
 * {@link NNWorker} is used to compute NN model according to splits assigned. The result will be sent to master for
 * accumulation.
 * <p/>
 * <p/>
 * Gradients in each worker will be sent to master to update weights of model in worker, which follows Encog's
 * multi-core implementation.
 */
public class NNWorker extends
        AbstractWorkerComputable<NNParams, NNParams, GuaguaWritableAdapter<LongWritable>, GuaguaWritableAdapter<Text>> {

    private static final Logger LOG = LoggerFactory.getLogger(NNWorker.class);

    /**
     * Default splitter used to split input record. Use one instance to prevent more news in Splitter.on.
     */
    private static final Splitter DEFAULT_SPLITTER = Splitter.on(NNConstants.NN_DEFAULT_COLUMN_SEPARATOR);

    /**
     * Training data set
     */
    private MLDataSet trainingData = null;

    /**
     * Testing data set
     */
    private MLDataSet testingData = null;

    /**
     * NN algorithm runner instance.
     */
    private Gradient gradient;

    /**
     * Model Config read from HDFS
     */
    private ModelConfig modelConfig;

    /**
     * Column Config list read from HDFS
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * Basic input node count for NN model
     */
    private int inputNodeCount;

    /**
     * Basic output node count for NN model
     */
    private int outputNodeCount;

    /**
     * {@link #candidateCount} is used to check if no variable is selected. If {@link #inputNodeCount} equals
     * {@link #candidateCount}, which means no column is selected or all columns are selected.
     */
    private int candidateCount;

    /**
     * input record size, inc one by one.
     */
    private long count;

    /**
     * sampled input record size.
     */
    private long sampleCount;

    /**
     * Whether the training is dry training.
     */
    private boolean isDry;

    private int epochsPerIteration = 1;

    /**
     * Load all configurations for modelConfig and columnConfigList from source type.
     */
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
     * Create memory data set object
     */
    @SuppressWarnings("unused")
    private void initMemoryDataSet() {
        this.trainingData = new BasicMLDataSet();
        this.testingData = new BasicMLDataSet();
    }

    /**
     * For disk data set , initialize it with parameters and other work about creating files.
     * 
     * @throws IOException
     *             if any exception on local fs operations.
     * @throws RuntimeException
     *             if error on deleting testing or training file.
     */
    private void initDiskDataSet() throws IOException {
        Path trainingFile = NNUtils.getTrainingFile();
        Path testingFile = NNUtils.getTestingFile();

        LOG.debug("Use disk to store training data and testing data. Training data file:{}; Testing data file:{} ",
                trainingFile.toString(), testingFile.toString());

        this.trainingData = new BufferedMLDataSet(new File(trainingFile.toString()));
        ((BufferedMLDataSet) this.trainingData).beginLoad(getInputNodeCount(), getOutputNodeCount());

        this.testingData = new BufferedMLDataSet(new File(testingFile.toString()));
        ((BufferedMLDataSet) this.testingData).beginLoad(getInputNodeCount(), getOutputNodeCount());
    }

    @Override
    public void init(WorkerContext<NNParams, NNParams> context) {
        loadConfigFiles(context.getProps());

        Integer epochsPerIterationInteger = this.modelConfig.getTrain().getEpochsPerIteration();
        this.epochsPerIteration = epochsPerIterationInteger == null ? 1 : epochsPerIterationInteger.intValue();
        LOG.info("epochsPerIteration in worker is :{}", epochsPerIteration);

        int[] inputOutputIndex = NNUtils.getInputOutputCandidateCounts(this.columnConfigList);
        this.inputNodeCount = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
        this.outputNodeCount = inputOutputIndex[1];
        this.candidateCount = inputOutputIndex[2];

        this.isDry = Boolean.TRUE.toString().equalsIgnoreCase(context.getProps().getProperty(NNConstants.NN_DRY_TRAIN));

        if(isOnDisk()) {
            LOG.info("NNWorker is loading data into disk.");
            try {
                initDiskDataSet();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            // cannot find a good place to close these two data set, using Shutdown hook
            Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
                @Override
                public void run() {
                    ((BufferedMLDataSet) (NNWorker.this.trainingData)).close();
                    ((BufferedMLDataSet) (NNWorker.this.testingData)).close();
                }
            }));
        } else {
            LOG.info("NNWorker is loading data into memory.");
            double memoryFraction = Double.valueOf(context.getProps().getProperty("guagua.data.memoryFraction", "0.5"));
            long memoryStoreSize = (long) (Runtime.getRuntime().maxMemory() * memoryFraction);
            double crossValidationRate = this.modelConfig.getCrossValidationRate();
            try {
                this.trainingData = new MemoryDiskMLDataSet((long) (memoryStoreSize * (1 - crossValidationRate)),
                        NNUtils.getTrainingFile().toString(), this.inputNodeCount, this.outputNodeCount);
                this.testingData = new MemoryDiskMLDataSet((long) (memoryStoreSize * crossValidationRate), NNUtils
                        .getTestingFile().toString(), this.inputNodeCount, this.outputNodeCount);
                // cannot find a good place to close these two data set, using Shutdown hook
                Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
                    @Override
                    public void run() {
                        ((MemoryDiskMLDataSet) (NNWorker.this.trainingData)).close();
                        ((MemoryDiskMLDataSet) (NNWorker.this.testingData)).close();
                    }
                }));
            } catch (IOException e) {
                throw new GuaguaRuntimeException(e);
            }
        }
    }

    private boolean isOnDisk() {
        return this.modelConfig.getTrain().getTrainOnDisk() != null
                && this.modelConfig.getTrain().getTrainOnDisk().booleanValue();
    }

    @Override
    public NNParams doCompute(WorkerContext<NNParams, NNParams> workerContext) {
        // For dry option, return empty result.
        // For first iteration, we don't do anything, just wait for master to update weights in next iteration. This
        // make sure all workers in the 1st iteration to get the same weights.
        if(this.isDry || workerContext.isFirstIteration()) {
            return buildEmptyNNParams(workerContext);
        }

        if(workerContext.getLastMasterResult() == null) {
            // This may not happen since master will set initialization weights firstly.
            LOG.warn("Master result of last iteration is null.");
            return null;
        }
        LOG.debug("Set current model with params {}", workerContext.getLastMasterResult());

        // initialize gradients if null
        if(gradient == null) {
            initGradient(this.trainingData, workerContext.getLastMasterResult().getWeights());
        }

        // using the weights from master to train model in current iteration
        this.gradient.setWeights(workerContext.getLastMasterResult().getWeights());

        for(int i = 0; i < epochsPerIteration; i++) {
            this.gradient.run();
            this.gradient.setWeights(this.gradient.getWeights());
        }
        // get train errors and test errors
        double trainError = this.gradient.getError();
        double testError = this.testingData.getRecordCount() > 0 ? (this.gradient.getNetwork()
                .calculateError(this.testingData)) : this.gradient.getError();
        // if the validation set is 0%, then the validation error should be "N/A"
        LOG.info("NNWorker compute iteration {} (train error {} validation error {})",
                new Object[] { workerContext.getCurrentIteration(), trainError,
                        (this.testingData.getRecordCount() > 0 ? testError : "N/A") });

        NNParams params = new NNParams();

        params.setTestError(testError);
        params.setTrainError(trainError);
        params.setGradients(this.gradient.getGradients());
        // prevent null point;
        params.setWeights(new double[0]);
        params.setTrainSize(this.trainingData.getRecordCount());
        return params;
    }

    @SuppressWarnings("unchecked")
    private void initGradient(MLDataSet training, double[] weights) {
        int numLayers = (Integer) getModelConfig().getParams().get(NNTrainer.NUM_HIDDEN_LAYERS);
        List<String> actFunc = (List<String>) getModelConfig().getParams().get(NNTrainer.ACTIVATION_FUNC);
        List<Integer> hiddenNodeList = (List<Integer>) getModelConfig().getParams().get(NNTrainer.NUM_HIDDEN_NODES);

        BasicNetwork network = NNUtils.generateNetwork(this.inputNodeCount, this.outputNodeCount, numLayers, actFunc,
                hiddenNodeList);
        // use the weights from master
        network.getFlat().setWeights(weights);

        FlatNetwork flat = network.getFlat();
        // copy Propagation from encog
        double[] flatSpot = new double[flat.getActivationFunctions().length];
        for(int i = 0; i < flat.getActivationFunctions().length; i++) {
            flatSpot[i] = flat.getActivationFunctions()[i] instanceof ActivationSigmoid ? 0.1 : 0.0;
        }

        this.gradient = new Gradient(flat, training, flatSpot, new LinearErrorFunction());
    }

    private NNParams buildEmptyNNParams(WorkerContext<NNParams, NNParams> workerContext) {
        NNParams params = new NNParams();
        params.setWeights(new double[0]);
        params.setGradients(new double[0]);
        params.setTestError(NNConstants.DRY_ERROR);
        params.setTrainError(NNConstants.DRY_ERROR);
        return params;
    }

    @Override
    protected void postLoad(WorkerContext<NNParams, NNParams> workerContext) {
        if(isOnDisk()) {
            ((BufferedMLDataSet) this.trainingData).endLoad();
            ((BufferedMLDataSet) this.testingData).endLoad();
        } else {
            ((MemoryDiskMLDataSet) this.trainingData).endLoad();
            ((MemoryDiskMLDataSet) this.testingData).endLoad();
        }
        LOG.info("    - # Records of the Master Data Set: {}.", this.count);
        LOG.info("    - Bagging Sample Rate: {}.", this.modelConfig.getBaggingSampleRate());
        LOG.info("    - Bagging With Replacement: {}.", this.modelConfig.isBaggingWithReplacement());
        LOG.info("    - # Records of the Selected Data Set:{}.", this.sampleCount);
        LOG.info("        - Cross Validation Rate: {}.", this.modelConfig.getCrossValidationRate());
        LOG.info("        - # Records of the Training Set: {}.", this.trainingData.getRecordCount());
        LOG.info("        - # Records of the Validation Set: {}.", this.testingData.getRecordCount());
    }

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<NNParams, NNParams> workerContext) {
        ++this.count;
        if((this.count) % 100000 == 0) {
            LOG.info("Read {} records.", this.count);
        }

        double baggingSampleRate = this.modelConfig.getBaggingSampleRate();
        // if fixInitialInput = false, we only compare random value with baggingSampleRate to avoid parsing data.
        // if fixInitialInput = true, we should use hashcode after parsing.
        if(!this.modelConfig.isFixInitialInput() && Double.valueOf(Math.random()).compareTo(baggingSampleRate) >= 0) {
            return;
        }

        double[] inputs = new double[this.inputNodeCount];
        double[] ideal = new double[this.outputNodeCount];

        if(this.isDry) {
            // dry train, use empty data.
            addDataPairToDataSet(0, new BasicMLDataPair(new BasicMLData(inputs), new BasicMLData(ideal)));
            return;
        }

        long hashcode = 0;
        double significance = NNConstants.DEFAULT_SIGNIFICANCE_VALUE;
        // use guava Splitter to iterate only once
        // use NNConstants.NN_DEFAULT_COLUMN_SEPARATOR to replace getModelConfig().getDataSetDelimiter(), this follows
        // the function in akka mode.
        int index = 0, inputsIndex = 0, outputIndex = 0;
        for(String input: DEFAULT_SPLITTER.split(currentValue.getWritable().toString())) {
            double doubleValue = NumberFormatUtils.getDouble(input.trim(), 0.0d);
            if(index == this.columnConfigList.size()) {
                significance = NumberFormatUtils.getDouble(input, NNConstants.DEFAULT_SIGNIFICANCE_VALUE);
                break;
            } else {
                ColumnConfig columnConfig = this.columnConfigList.get(index);

                if(columnConfig != null && columnConfig.isTarget()) {
                    ideal[outputIndex++] = doubleValue;
                } else {
                    if(this.inputNodeCount == this.candidateCount) {
                        // all variables are not set final-selectByFilter
                        if(CommonUtils.isGoodCandidate(columnConfig)) {
                            inputs[inputsIndex++] = doubleValue;
                            hashcode = hashcode * 31 + Double.valueOf(doubleValue).hashCode();
                        }
                    } else {
                        // final select some variables
                        if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                                && columnConfig.isFinalSelect()) {
                            inputs[inputsIndex++] = doubleValue;
                            // only fixInitialInput=true, hashcode is effective. Remove Arrays.hashcode to avoid one
                            // iteration for the input columns. Last weight column should be excluded.
                            hashcode = hashcode * 31 + Double.valueOf(doubleValue).hashCode();
                        }
                    }
                }
            }
            index++;
        }

        // if fixInitialInput = true, we should use hashcode to sample.
        long longBaggingSampleRate = Double.valueOf(baggingSampleRate * 100).longValue();
        if(this.modelConfig.isFixInitialInput() && hashcode % 100 >= longBaggingSampleRate) {
            return;
        }

        ++this.sampleCount;

        MLDataPair pair = new BasicMLDataPair(new BasicMLData(inputs), new BasicMLData(ideal));
        pair.setSignificance(significance);

        addDataPairToDataSet(hashcode, pair);
    }

    /**
     * Add data pair to data set according to setting parameters. Still set hashCode to long to make double and long
     * friendly.
     */
    private void addDataPairToDataSet(long hashcode, MLDataPair pair) {
        double crossValidationRate = this.modelConfig.getCrossValidationRate();
        if(this.modelConfig.isFixInitialInput()) {
            long longCrossValidation = Double.valueOf(crossValidationRate * 100).longValue();
            if(hashcode % 100 < longCrossValidation) {
                this.testingData.add(pair);
            } else {
                this.trainingData.add(pair);
            }
        } else {
            double random = Math.random();
            if(isBaggingReplacementTrigged(random)) {
                mockRandomRepeatData(crossValidationRate, random);
            } else {
                addDataPairToDataSet(pair, crossValidationRate, random);
            }
        }
    }

    /**
     * Only baggingWithReplacement is set and size over NNConstants.NN_BAGGING_THRESHOLD, and random value <= 1/size. We
     * choose use existing data to add training data set and testing data set.
     */
    private boolean isBaggingReplacementTrigged(double random) {
        long trainingSize = this.trainingData.getRecordCount();
        long testingSize = this.testingData.getRecordCount();
        // size should be equals to sampleCount:)
        long size = trainingSize + testingSize;
        return this.modelConfig.isBaggingWithReplacement() && (testingSize > 0) && (trainingSize > 0)
                && (size > NNConstants.NN_BAGGING_THRESHOLD)
                && (Double.valueOf(random).compareTo(Double.valueOf(0.5d)) < 0);
    }

    /**
     * From Trainer, the logic is to random choose items in master dataset, but I don't want to load data twice for
     * saving memory. Use this to mock raw random repeat logic. This should be some logic difference because of data are
     * not loaded into data set, not random.
     */
    private void mockRandomRepeatData(double crossValidationRate, double random) {
        long trainingSize = this.trainingData.getRecordCount();
        long testingSize = this.testingData.getRecordCount();
        long size = trainingSize + testingSize;
        // here we used a strong cast from long to int since it's just a random choosing algorithm
        int next = RandomUtils.nextInt((int) size);
        MLDataPair dataPair = new BasicMLDataPair(new BasicMLData(new double[this.inputNodeCount]), new BasicMLData(
                new double[this.outputNodeCount]));
        if(next >= trainingSize) {
            this.testingData.getRecord(next - trainingSize, dataPair);
        } else {
            this.trainingData.getRecord(next, dataPair);
        }

        if(Double.valueOf(random).compareTo(Double.valueOf(crossValidationRate)) < 0) {
            this.testingData.add(dataPair);
        } else {
            this.trainingData.add(dataPair);
        }
    }

    /**
     * Add data pair to data set according to random number compare with crossValidationRate.
     */
    private void addDataPairToDataSet(MLDataPair pair, double crossValidationRate, double random) {
        if(Double.valueOf(random).compareTo(Double.valueOf(crossValidationRate)) < 0) {
            this.testingData.add(pair);
        } else {
            this.trainingData.add(pair);
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.worker.AbstractWorkerComputable#initRecordReader(ml.shifu.guagua.io.GuaguaFileSplit)
     */
    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        this.setRecordReader(new GuaguaLineRecordReader());
        this.getRecordReader().initialize(fileSplit);
    }

    public MLDataSet getTrainingData() {
        return trainingData;
    }

    public void setTrainingData(MLDataSet trainingData) {
        this.trainingData = trainingData;
    }

    public MLDataSet getTestingData() {
        return testingData;
    }

    public void setTestingData(MLDataSet testingData) {
        this.testingData = testingData;
    }

    public ModelConfig getModelConfig() {
        return modelConfig;
    }

    public void setModelConfig(ModelConfig modelConfig) {
        this.modelConfig = modelConfig;
    }

    public List<ColumnConfig> getColumnConfigList() {
        return columnConfigList;
    }

    public void setColumnConfigList(List<ColumnConfig> columnConfigList) {
        this.columnConfigList = columnConfigList;
    }

    public int getInputNodeCount() {
        return inputNodeCount;
    }

    public void setInputNodeCount(int inputNodeCount) {
        this.inputNodeCount = inputNodeCount;
    }

    public int getOutputNodeCount() {
        return outputNodeCount;
    }

    public void setOutputNodeCount(int outputNodeCount) {
        this.outputNodeCount = outputNodeCount;
    }

}
