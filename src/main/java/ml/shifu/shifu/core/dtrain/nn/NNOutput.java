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
package ml.shifu.shifu.core.dtrain.nn;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.GZIPOutputStream;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.encog.neural.networks.BasicNetwork;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.persist.PersistorRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.guagua.master.BasicMasterInterceptor;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.PersistBasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

/**
 * {@link NNOutput} is used to write the model output to file system.
 */
public class NNOutput extends BasicMasterInterceptor<NNParams, NNParams> {

    private static final Logger LOG = LoggerFactory.getLogger(NNOutput.class);

    private static final double EPSILON = 0.0000001;

    /**
     * Model Config read from HDFS
     */
    private ModelConfig modelConfig;

    /**
     * Column Config list read from HDFS
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * network
     */
    private BasicNetwork network;

    /**
     * Trainer id in bagging jobs
     */
    private String trainerId;

    /**
     * Save model to tmp hdfs folder
     */
    private String tmpModelsFolder;

    /**
     * Whether the training is dry training.
     */
    private boolean isDry;

    /**
     * A flag: whether params initialized.
     */
    private AtomicBoolean isInit = new AtomicBoolean(false);

    /**
     * The minimum test error during model training
     */
    private double minTestError = Double.MAX_VALUE;

    /**
     * The best weights that we meet
     */
    private double[] optimizedWeights = null;

    /**
     * Progress output stream which is used to write progress to that HDFS file. Should be closed in
     * {@link #postApplication(MasterContext)}.
     */
    private FSDataOutputStream progressOutput = null;

    /**
     * If for gs mode, write valid error to output
     */
    private GridSearch gridSearch;

    /**
     * Valid model parameters which is unified for both normal training and grid search.
     */
    private Map<String, Object> validParams;

    private boolean isKFoldCV;

    /**
     * Cache subset features with feature index for searching
     */
    protected Set<Integer> subFeatures;

    /**
     * Dropout rate which is in [0, 1], default it is 0
     */
    private double dropoutRate = 0d;

    @Override
    public void preApplication(MasterContext<NNParams, NNParams> context) {
        init(context);
    }

    @Override
    public void postIteration(final MasterContext<NNParams, NNParams> context) {
        if(this.isDry) {
            // for dry mode, we don't save models files.
            return;
        }

        double currentError = ((modelConfig.getTrain().getValidSetRate() < EPSILON) ? context.getMasterResult()
                .getTrainError() : context.getMasterResult().getTestError());

        // save the weights according the error decreasing
        if(currentError < this.minTestError) {
            this.minTestError = currentError;
            this.optimizedWeights = context.getMasterResult().getWeights();
        }

        // save tmp to hdfs according to raw trainer logic
        final int tmpModelFactor = DTrainUtils.tmpModelFactor(context.getTotalIteration());
        final int currentIteration = context.getCurrentIteration();
        final double[] weights = context.getMasterResult().getWeights();
        final int totalIteration = context.getTotalIteration();
        final boolean isHalt = context.getMasterResult().isHalt();
        // currentIteration - 1 because the first iteration is used for sync master models to workers
        if(currentIteration > 1 && (currentIteration - 1) % tmpModelFactor == 0) {
            Thread tmpNNThread = new Thread(new Runnable() {
                @Override
                public void run() {
                    saveTmpNNToHDFS(currentIteration - 1, weights);
                    // save model results for continue model training, if current job is failed, then next running
                    // we can start from this point to save time.
                    // another case for master recovery, if master is failed, read such checkpoint model
                    Path out = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));

                    // if current iteration is the last iteration, or it is halted by early stop condition, no need to
                    // save checkpoint model here as it is replicated with postApplicaiton.
                    // There is issue here if saving the same model in this thread and another thread in
                    // postApplication, sometimes this conflict will cause model writing failed.
                    if(!isHalt && currentIteration != totalIteration) {
                        writeModelWeightsToFileSystem(optimizedWeights, out);
                    }
                }
            }, "saveTmpNNToHDFS thread");
            tmpNNThread.setDaemon(true);
            tmpNNThread.start();
        }

        updateProgressLog(context);
    }

    @SuppressWarnings("deprecation")
    private void updateProgressLog(final MasterContext<NNParams, NNParams> context) {
        int currentIteration = context.getCurrentIteration();
        if(context.isFirstIteration()) {
            // first iteration is used for training preparation
            return;
        }
        String progress = new StringBuilder(200).append("    Trainer ").append(this.trainerId).append(" Epoch #")
                .append(currentIteration - 1).append(" Training Error:")
                .append(String.format("%.10f", context.getMasterResult().getTrainError())).append(" Validation Error:")
                .append(String.format("%.10f", context.getMasterResult().getTestError())).append("\n").toString();
        try {
            LOG.debug("Writing progress results to {} {}", context.getCurrentIteration(), progress.toString());
            this.progressOutput.write(progress.getBytes("UTF-8"));
            this.progressOutput.flush();
            this.progressOutput.sync();
        } catch (IOException e) {
            LOG.error("Error in write progress log:", e);
        }
    }

    @Override
    public void postApplication(MasterContext<NNParams, NNParams> context) {
        IOUtils.closeStream(this.progressOutput);

        // for dry mode, we don't save models files.
        if(this.isDry) {
            return;
        }

        if(optimizedWeights != null) {
            Path out = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
            // TODO do we need to check IOException and retry again to make sure such important model is saved
            // successfully.
            writeModelWeightsToFileSystem(optimizedWeights, out);
        }

        if(this.gridSearch.hasHyperParam() || this.isKFoldCV) {
            Path valErrOutput = new Path(context.getProps().getProperty(CommonConstants.GS_VALIDATION_ERROR));
            writeValErrorToFileSystem(context.getMasterResult().getTestError(), valErrOutput);
        }
    }

    private void writeValErrorToFileSystem(double valError, Path out) {
        FSDataOutputStream fos = null;
        try {
            fos = FileSystem.get(new Configuration()).create(out);
            LOG.info("Writing valerror to {}", out);
            fos.write((valError + "").getBytes("UTF-8"));
        } catch (IOException e) {
            LOG.error("Error in writing output.", e);
        } finally {
            IOUtils.closeStream(fos);
        }
    }

    /**
     * Save tmp nn model to HDFS.
     */
    private void saveTmpNNToHDFS(int iteration, double[] weights) {
        Path out = new Path(DTrainUtils.getTmpModelName(this.tmpModelsFolder, this.trainerId, iteration, modelConfig
                .getTrain().getAlgorithm().toLowerCase()));
        writeModelWeightsToFileSystem(weights, out);
    }

    private void init(MasterContext<NNParams, NNParams> context) {
        this.isDry = Boolean.TRUE.toString().equals(context.getProps().getProperty(CommonConstants.SHIFU_DRY_DTRAIN));

        if(this.isDry) {
            return;
        }
        if(isInit.compareAndSet(false, true)) {
            loadConfigFiles(context.getProps());
            this.trainerId = context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID);
            this.tmpModelsFolder = context.getProps().getProperty(CommonConstants.SHIFU_TMP_MODELS_FOLDER);
            gridSearch = new GridSearch(modelConfig.getTrain().getParams());
            validParams = this.modelConfig.getTrain().getParams();
            if(gridSearch.hasHyperParam()) {
                validParams = gridSearch.getParams(Integer.parseInt(trainerId));
                LOG.info("Start grid search in nn output with params: {}", validParams);
            }
            Integer kCrossValidation = this.modelConfig.getTrain().getNumKFold();
            if(kCrossValidation != null && kCrossValidation > 0) {
                isKFoldCV = true;
            }

            Object dropoutRateObj = validParams.get(CommonConstants.DROPOUT_RATE);
            if(dropoutRateObj != null) {
                this.dropoutRate = Double.valueOf(dropoutRateObj.toString());
            }
            LOG.info("'dropoutRate' in master output is :{}", this.dropoutRate);

            initNetwork(context);
        }

        try {
            Path progressLog = new Path(context.getProps().getProperty(CommonConstants.SHIFU_DTRAIN_PROGRESS_FILE));
            // if the progressLog already exists, that because the master failed, and fail-over
            // we need to append the log, so that client console can get refreshed. Or console will appear stuck.
            if(ShifuFileUtils.isFileExists(progressLog, SourceType.HDFS)) {
                this.progressOutput = FileSystem.get(new Configuration()).append(progressLog);
            } else {
                this.progressOutput = FileSystem.get(new Configuration()).create(progressLog);
            }
        } catch (IOException e) {
            LOG.error("Error in create progress log:", e);
        }
    }

    /**
     * Load all configurations for modelConfig and columnConfigList from source type. Use null check to make sure model
     * config and column config loaded once.
     */
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

    @SuppressWarnings("unchecked")
    private void initNetwork(MasterContext<NNParams, NNParams> context) {
        int[] inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(this.columnConfigList);
        @SuppressWarnings("unused")
        int inputNodeCount = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
        // if is one vs all classification, outputNodeCount is set to 1
        int outputNodeCount = modelConfig.isRegression() ? inputOutputIndex[1]
                : (modelConfig.getTrain().isOneVsAll() ? inputOutputIndex[1] : modelConfig.getTags().size());
        int numLayers = (Integer) validParams.get(CommonConstants.NUM_HIDDEN_LAYERS);
        List<String> actFunc = (List<String>) validParams.get(CommonConstants.ACTIVATION_FUNC);
        List<Integer> hiddenNodeList = (List<Integer>) validParams.get(CommonConstants.NUM_HIDDEN_NODES);

        boolean isAfterVarSelect = inputOutputIndex[0] != 0;
        // cache all feature list for sampling features
        List<Integer> allFeatures = CommonUtils.getAllFeatureList(columnConfigList, isAfterVarSelect);
        String subsetStr = context.getProps().getProperty(CommonConstants.SHIFU_NN_FEATURE_SUBSET);
        if(StringUtils.isBlank(subsetStr)) {
            this.subFeatures = new HashSet<Integer>(allFeatures);
        } else {
            String[] splits = subsetStr.split(",");
            this.subFeatures = new HashSet<Integer>();
            for(String split: splits) {
                this.subFeatures.add(Integer.parseInt(split));
            }
        }

        this.network = DTrainUtils.generateNetwork(this.subFeatures.size(), outputNodeCount, numLayers, actFunc,
                hiddenNodeList, false, this.dropoutRate);
        ((BasicFloatNetwork) this.network).setFeatureSet(this.subFeatures);

        PersistorRegistry.getInstance().add(new PersistBasicFloatNetwork());
    }

    private void writeModelWeightsToFileSystem(double[] weights, Path out) {
        double[] finalWeights = null;
        if(this.dropoutRate == 0d) {
            finalWeights = weights;
        } else {
            finalWeights = new double[weights.length];
            for(int i = 0; i < finalWeights.length; i++) {
                // do we need to norm all weights or leave last hidden layer to output layer not normed???
                // it is ok to add or not added in last iteration as such parameters only impact last final output score
                // but not change the order of scores
                finalWeights[i] = weights[i] * (1 - this.dropoutRate);
            }
        }
        FSDataOutputStream fos = null;
        try {
            fos = FileSystem.get(new Configuration()).create(out);
            LOG.info("Writing results to {}", out);
            this.network.getFlat().setWeights(finalWeights);
            if(out != null) {
                EncogDirectoryPersistence.saveObject(fos, this.network);
            }
        } catch (IOException e) {
            LOG.error("Error in writing output.", e);
        } finally {
            IOUtils.closeStream(fos);
        }
    }

    private void writeModelToFileSystem(double[] weights, Path out) {
        double[] finalWeights = null;
        if(this.dropoutRate == 0d) {
            finalWeights = weights;
        } else {
            finalWeights = new double[weights.length];
            for(int i = 0; i < finalWeights.length; i++) {
                // do we need to norm all weights or leave last hidden layer to output layer not normed???
                // it is ok to add or not added in last iteration as such parameters only impact last final output score
                // but not change the order of scores
                finalWeights[i] = weights[i] * (1 - this.dropoutRate);
            }
        }

        DataOutputStream fos = null;
        try {
            fos = new DataOutputStream(new GZIPOutputStream(FileSystem.get(new Configuration()).create(out)));
            LOG.info("Writing NN models to {}.", out);
            this.network.getFlat().setWeights(finalWeights);
            // version
            fos.writeInt(CommonConstants.NN_FORMAT_VERSION);
            fos.writeInt(this.network.getInputCount());

            Map<Integer, String> columnIndexNameMapping = new HashMap<Integer, String>();
            Map<Integer, List<String>> columnIndexCategoricalListMapping = new HashMap<Integer, List<String>>();
            Map<Integer, Double> numericalMeanMapping = new HashMap<Integer, Double>();
            for(ColumnConfig columnConfig: this.columnConfigList) {
                if(columnConfig.isFinalSelect()) {
                    columnIndexNameMapping.put(columnConfig.getColumnNum(), columnConfig.getColumnName());
                }
                if(columnConfig.isCategorical() && CollectionUtils.isNotEmpty(columnConfig.getBinCategory())) {
                    columnIndexCategoricalListMapping.put(columnConfig.getColumnNum(), columnConfig.getBinCategory());
                }

                if(columnConfig.isNumerical() && columnConfig.getMean() != null) {
                    numericalMeanMapping.put(columnConfig.getColumnNum(), columnConfig.getMean());
                }
            }

            if(columnIndexNameMapping.size() == 0) {
                for(ColumnConfig columnConfig: this.columnConfigList) {
                    if(CommonUtils.isGoodCandidate(columnConfig)) {
                        columnIndexNameMapping.put(columnConfig.getColumnNum(), columnConfig.getColumnName());
                    }
                }
            }

            // serialize numericalMeanMapping
            fos.writeInt(numericalMeanMapping.size());
            for(Entry<Integer, Double> entry: numericalMeanMapping.entrySet()) {
                fos.writeInt(entry.getKey());
                // for some feature, it is null mean value, it is not selected, just set to 0d to avoid NPE
                fos.writeDouble(entry.getValue() == null ? 0d : entry.getValue());
            }
            // serialize columnIndexNameMapping
            fos.writeInt(columnIndexNameMapping.size());
            for(Entry<Integer, String> entry: columnIndexNameMapping.entrySet()) {
                fos.writeInt(entry.getKey());
                fos.writeUTF(entry.getValue());
            }
            // serialize columnIndexCategoricalListMapping
            fos.writeInt(columnIndexCategoricalListMapping.size());
            for(Entry<Integer, List<String>> entry: columnIndexCategoricalListMapping.entrySet()) {
                List<String> categories = entry.getValue();
                if(categories != null) {
                    fos.writeInt(entry.getKey());
                    fos.writeInt(categories.size());
                    for(String category: categories) {
                        // There is 16k limitation when using writeUTF() function.
                        // if the category value is larger than 10k, then treat it as missing value
                        if(category.length() > Constants.MAX_CATEGORICAL_VAL_LEN) {
                            int pos = category.lastIndexOf(Constants.CATEGORICAL_GROUP_VAL_DELIMITER,
                                    Constants.MAX_CATEGORICAL_VAL_LEN);
                            if(pos >= 0) {
                                category = category.substring(0, pos);
                            } else {
                                category = category.substring(0, Constants.MAX_CATEGORICAL_VAL_LEN);
                            }
                        }
                        fos.writeUTF(category);
                    }
                }
            }

            Map<Integer, Map<String, Double>> categoricalWoeMappings = new HashMap<Integer, Map<String, Double>>();
            Map<Integer, Map<String, Double>> weightedCategoricalWoeMappings = new HashMap<Integer, Map<String, Double>>();
            Map<Integer, Map<String, Double>> binPosRateMappings = new HashMap<Integer, Map<String, Double>>();

            Map<Integer, List<Double>> numericalBinBoundaries = new HashMap<Integer, List<Double>>();
            Map<Integer, List<Double>> numericalWeightedWoes = new HashMap<Integer, List<Double>>();
            Map<Integer, List<Double>> numericalWoes = new HashMap<Integer, List<Double>>();

            Map<Integer, Double> numericalMeanMappings = new HashMap<Integer, Double>();
            Map<Integer, Double> numericalStddevMappings = new HashMap<Integer, Double>();
            Map<Integer, Double> woeMeanMappings = new HashMap<Integer, Double>();
            Map<Integer, Double> woeStddevMappings = new HashMap<Integer, Double>();
            Map<Integer, Double> weightedWoeMeanMappings = new HashMap<Integer, Double>();
            Map<Integer, Double> weightedWoeStddevMappings = new HashMap<Integer, Double>();

            for(ColumnConfig config: this.columnConfigList) {
                if(columnIndexNameMapping.containsKey(config.getColumnNum())) {
                    if(config.isCategorical()) {
                        Map<String, Double> woeMap = new HashMap<String, Double>();
                        Map<String, Double> weightedWoeMap = new HashMap<String, Double>();
                        Map<String, Double> posRateMap = new HashMap<String, Double>();

                        List<String> binCategory = config.getBinCategory();
                        List<Double> binCountWoe = config.getBinCountWoe();
                        List<Double> binWeightWoe = config.getBinWeightedWoe();
                        List<Double> binPosRate = config.getBinPosRate();
                        for(int i = 0; i < binCategory.size(); i++) {
                            woeMap.put(binCategory.get(i), binCountWoe.get(i));
                            weightedWoeMap.put(binCategory.get(i), binWeightWoe.get(i));
                            posRateMap.put(binCategory.get(i), binPosRate.get(i));
                        }

                        // size of binCountWoe = size of binCategory + 1, no issue for put missing woe here
                        woeMap.put(Constants.EMPTY_CATEGORY, binCountWoe.get(binCategory.size()));
                        weightedWoeMap.put(Constants.EMPTY_CATEGORY, binWeightWoe.get(binCategory.size()));
                        posRateMap.put(Constants.EMPTY_CATEGORY, binPosRate.get(binCategory.size()));

                        categoricalWoeMappings.put(config.getColumnNum(), woeMap);
                        weightedCategoricalWoeMappings.put(config.getColumnNum(), weightedWoeMap);
                        binPosRateMappings.put(config.getColumnNum(), posRateMap);
                    } else {

                    }
                }
            }

            Map<Integer, Integer> columnMapping = getColumnMapping();
            fos.writeInt(columnMapping.size());
            for(Entry<Integer, Integer> entry: columnMapping.entrySet()) {
                fos.writeInt(entry.getKey());
                fos.writeInt(entry.getValue());
            }

            // persist network
            new PersistBasicFloatNetwork().save(fos, this.network);
        } catch (IOException e) {
            LOG.error("Error in writing output.", e);
        } finally {
            IOUtils.closeStream(fos);
        }
    }

    private Map<Integer, Integer> getColumnMapping() {
        Map<Integer, Integer> columnMapping = new HashMap<Integer, Integer>(columnConfigList.size(), 1f);
        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(columnConfigList);
        boolean isAfterVarSelect = inputOutputIndex[3] == 1 ? true : false;
        int index = 0;
        for(int i = 0; i < columnConfigList.size(); i++) {
            ColumnConfig columnConfig = columnConfigList.get(i);
            if(!isAfterVarSelect) {
                if(!columnConfig.isMeta() && !columnConfig.isTarget() && CommonUtils.isGoodCandidate(columnConfig)) {
                    columnMapping.put(columnConfig.getColumnNum(), index);
                    index += 1;
                }
            } else {
                if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                        && columnConfig.isFinalSelect()) {
                    columnMapping.put(columnConfig.getColumnNum(), index);
                    index += 1;
                }
            }
        }
        return columnMapping;
    }

    public ModelConfig getModelConfig() {
        return modelConfig;
    }

}
