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
package ml.shifu.shifu.core.dtrain;

import ml.shifu.guagua.GuaguaRuntimeException;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.FloatNeuralStructure;
import ml.shifu.shifu.core.dtrain.nn.*;
import ml.shifu.shifu.core.dtrain.random.HeWeightRandomizer;
import ml.shifu.shifu.core.dtrain.random.LecunWeightRandomizer;
import ml.shifu.shifu.core.dtrain.random.XavierWeightRandomizer;
import ml.shifu.shifu.udf.norm.CategoryMissingNormType;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.HDFSUtils;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.fs.Path;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.Tuple;
import org.encog.Encog;
import org.encog.engine.network.activation.*;
import org.encog.mathutil.randomize.GaussianRandomizer;
import org.encog.mathutil.randomize.NguyenWidrowRandomizer;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.structure.NeuralStructure;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;

/**
 * Helper class for NN distributed training.
 */
public final class DTrainUtils {

    private static Logger LOG = LoggerFactory.getLogger(DTrainUtils.class);

    public static final String RESILIENTPROPAGATION = "R";
    public static final String SCALEDCONJUGATEGRADIENT = "S";
    public static final String MANHATTAN_PROPAGATION = "M";
    public static final String QUICK_PROPAGATION = "Q";
    public static final String BACK_PROPAGATION = "B";

    public static final String WGT_INIT_GAUSSIAN = "gaussian";

    public static final String WGT_INIT_DEFAULT = "default";

    public static final String WGT_INIT_XAVIER = "xavier";

    public static final String WGT_INIT_HE = "he";

    public static final String WGT_INIT_LECUN = "lecun";

    /**
     * The POSITIVE ETA value. This is specified by the resilient propagation algorithm. This is the percentage by which
     * the deltas are increased by if the partial derivative is greater than zero.
     */
    public static final double POSITIVE_ETA = 1.2;

    /**
     * The NEGATIVE ETA value. This is specified by the resilient propagation algorithm. This is the percentage by which
     * the deltas are increased by if the partial derivative is less than zero.
     */
    public static final double NEGATIVE_ETA = 0.5;

    /**
     * The minimum delta value for a weight matrix value.
     */
    public static final double DELTA_MIN = 1e-6;

    /**
     * The starting update for a delta.
     */
    public static final double DEFAULT_INITIAL_UPDATE = 0.1;

    private DTrainUtils() {
    }

    /**
     * Check tmp dir for data set to store. If not exist, create it firstly.
     */
    private static Path getTmpDir() throws IOException {
        // If the Constants.TMP is absolute folder, there may be some conflicts for two jobs.
        Path path = new Path(Constants.TMP);
        if(!HDFSUtils.getLocalFS().exists(path)) {
            if(!HDFSUtils.getLocalFS().mkdirs(path)) {
                throw new RuntimeException("Error in creating tmp folder.");
            }
        }
        return path;
    }

    /**
     * Return testing file to store training data, if exists, delete it firstly.
     * 
     * @return the testing file path
     * @throws IOException
     *             if any exception on local fs operations.
     * @throws RuntimeException
     *             if error on deleting testing file.
     */
    public static Path getTestingFile() throws IOException {
        Path testingFile = new Path(getTmpDir(), NNConstants.TESTING_EGB);
        if(HDFSUtils.getLocalFS().exists(testingFile)) {
            if(!HDFSUtils.getLocalFS().delete(testingFile, true)) {
                throw new RuntimeException("error in deleting testing file");
            }
        }

        return testingFile;
    }

    /**
     * Return training file to store training data, if exists, delete it firstly.
     * 
     * @return the training file path
     * @throws IOException
     *             if any exception on local fs operations.
     * @throws RuntimeException
     *             if error on deleting training file.
     */
    public static Path getTrainingFile() throws IOException {
        Path trainingFile = new Path(getTmpDir(), NNConstants.TRAINING_EGB);
        if(HDFSUtils.getLocalFS().exists(trainingFile)) {
            if(!HDFSUtils.getLocalFS().delete(trainingFile, true)) {
                throw new RuntimeException("error in deleting traing file");
            }
        }
        return trainingFile;
    }

    /**
     * Get input nodes number (final select) and output nodes number from column config, and candidate input node
     * number.
     * 
     * <p>
     * If number of column in final-select is 0, which means to select all non meta and non target columns. So the input
     * number is set to all candidates.
     * 
     * @param normType
     *            normalization type
     * @param columnConfigList
     *            the column config list
     * @return [input, output, candidate]
     * @throws NullPointerException
     *             if columnConfigList or ColumnConfig object in columnConfigList is null.
     */
    public static int[] getInputOutputCandidateCounts(ModelNormalizeConf.NormType normType,
            List<ColumnConfig> columnConfigList) {
        int input = 0, output = 0, goodCandidate = 0;
        boolean hasCandidate = CommonUtils.hasCandidateColumns(columnConfigList);
        for(ColumnConfig config: columnConfigList) {
            if(!config.isTarget() && !config.isMeta()) {
                if(CommonUtils.isGoodCandidate(config, hasCandidate)) {
                    goodCandidate += 1;
                }
            }
            if(config.isFinalSelect() && !config.isTarget() && !config.isMeta()) {
                if(normType.equals(ModelNormalizeConf.NormType.ONEHOT)) {
                    if(config.isCategorical()) {
                        input += config.getBinCategory().size() + 1;
                    } else {
                        input += config.getBinBoundary().size() + 1;
                    }
                } else if(normType.equals(ModelNormalizeConf.NormType.ZSCALE_ONEHOT) && config.isCategorical()) {
                    input += config.getBinCategory().size() + 1;
                } else {
                    input += 1;
                }
            }
            if(config.isTarget()) {
                output += 1;
            }
        }
        return new int[] { input, output, goodCandidate };
    }

    /**
     * Get the model output dimension - usually it will be 1
     * 
     * @param columnConfigList
     *            the column config list
     * @return the output count
     */
    public static int getModelOutputCnt(List<ColumnConfig> columnConfigList) {
        int output = 0;
        for(ColumnConfig config: columnConfigList) {
            if(config.isTarget()) {
                output += 1;
            }
        }
        return output;
    }

    /**
     * Get numeric and categorical input nodes number (final select) and output nodes number from column config, and
     * candidate input node number.
     * 
     * <p>
     * If number of column in final-select is 0, which means to select all non meta and non target columns. So the input
     * number is set to all candidates.
     * 
     * @param columnConfigList
     *            the column config list
     * @return [input, output, candidate]
     * @throws NullPointerException
     *             if columnConfigList or ColumnConfig object in columnConfigList is null.
     */
    public static int[] getNumericAndCategoricalInputAndOutputCounts(List<ColumnConfig> columnConfigList) {
        int numericInput = 0, categoricalInput = 0, output = 0, numericCandidateInput = 0,
                categoricalCandidateInput = 0;
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);

        for(ColumnConfig config: columnConfigList) {
            if(!config.isTarget() && !config.isMeta() && CommonUtils.isGoodCandidate(config, hasCandidates)) {
                if(config.isNumerical()) {
                    numericCandidateInput += 1;
                }
                if(config.isCategorical()) {
                    categoricalCandidateInput += 1;
                }
            }
            if(config.isFinalSelect() && !config.isTarget() && !config.isMeta()) {
                if(config.isNumerical()) {
                    numericInput += 1;
                }
                if(config.isCategorical()) {
                    categoricalInput += 1;
                }
            }
            if(config.isTarget()) {
                output += 1;
            }
        }

        // check if it is after varselect, if not, no variable is set to finalSelect which means, all good variable
        // should be set as finalSelect TODO, bad practice, refactor me
        int isVarSelect = 1;
        if(numericInput == 0 && categoricalInput == 0) {
            numericInput = numericCandidateInput;
            categoricalInput = categoricalCandidateInput;
            isVarSelect = 0;
        }

        return new int[] { numericInput, categoricalInput, output, isVarSelect };
    }

    public static Map<Integer, Integer> getColumnMapping(List<ColumnConfig> columnConfigList) {
        Map<Integer, Integer> columnMapping = new HashMap<Integer, Integer>(columnConfigList.size(), 1f);
        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(columnConfigList);
        boolean isAfterVarSelect = inputOutputIndex[3] == 1 ? true : false;
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
        int index = 0;
        for(int i = 0; i < columnConfigList.size(); i++) {
            ColumnConfig columnConfig = columnConfigList.get(i);
            if(!isAfterVarSelect) {
                if(!columnConfig.isMeta() && !columnConfig.isTarget()
                        && CommonUtils.isGoodCandidate(columnConfig, hasCandidates)) {
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

    public static List<Integer> getNumericalIds(List<ColumnConfig> columnConfigList, boolean isAfterVarSelect) {
        List<Integer> numericalIds = new ArrayList<>();
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);

        for(ColumnConfig config: columnConfigList) {
            if(isAfterVarSelect) {
                if(config.isNumerical() && config.isFinalSelect() && !config.isTarget() && !config.isMeta()) {
                    numericalIds.add(config.getColumnNum());
                }
            } else {
                if(config.isNumerical() && !config.isTarget() && !config.isMeta()
                        && CommonUtils.isGoodCandidate(config, hasCandidates)) {
                    numericalIds.add(config.getColumnNum());
                }
            }
        }
        return numericalIds;
    }

    public static List<Integer> getCategoricalIds(List<ColumnConfig> columnConfigList, boolean isAfterVarSelect) {
        List<Integer> results = new ArrayList<>();
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);

        for(ColumnConfig config: columnConfigList) {
            if(isAfterVarSelect) {
                if(config.isFinalSelect() && !config.isTarget() && !config.isMeta() && config.isCategorical()) {
                    results.add(config.getColumnNum());
                }
            } else {
                if(!config.isTarget() && !config.isMeta() && CommonUtils.isGoodCandidate(config, hasCandidates)
                        && config.isCategorical()) {
                    results.add(config.getColumnNum());
                }
            }
        }

        return results;
    }

    public static String getTmpModelName(String tmpModelsFolder, String trainerId, int iteration, String modelPost) {
        return new StringBuilder(200).append(tmpModelsFolder).append(Path.SEPARATOR_CHAR).append("model")
                .append(trainerId).append('-').append(iteration).append(".").append(modelPost).toString();
    }

    public static int tmpModelFactor(int epochs) {
        return Math.max(epochs / 25, 20);
    }

    // public static BasicNetwork generateNetwork(int in, int out, int numLayers, List<String> actFunc,
    // List<Integer> hiddenNodeList, boolean isRandomizeWeights, double dropoutRate) {
    // return generateNetwork(in, out, numLayers, actFunc, hiddenNodeList, isRandomizeWeights, dropoutRate,
    // WGT_INIT_DEFAULT);
    // }

    public static BasicNetwork generateNetwork(int in, int out, int numLayers, List<String> actFunc,
            List<Integer> hiddenNodeList, boolean isRandomizeWeights, double dropoutRate, String wgtInit,
            boolean isLinearTarget, String outputActivationFunc) {
        final BasicFloatNetwork network = new BasicFloatNetwork();

        // in shifuconfig, we have a switch to control enable input layer dropout
        if(Boolean.valueOf(Environment.getProperty(CommonConstants.SHIFU_TRAIN_NN_INPUTLAYERDROPOUT_ENABLE, "true"))) {
            // we need to guarantee that input layer dropout rate is 40% of hiddenlayer dropout rate
            network.addLayer(new BasicDropoutLayer(new ActivationLinear(), true, in, dropoutRate * 0.4d));
        } else {
            network.addLayer(new BasicDropoutLayer(new ActivationLinear(), true, in, 0d));
        }

        // int hiddenNodes = 0;
        for(int i = 0; i < numLayers; i++) {
            String func = actFunc.get(i);
            Integer numHiddenNode = hiddenNodeList.get(i);
            // hiddenNodes += numHiddenNode;
            if(func.equalsIgnoreCase(NNConstants.NN_LINEAR)) {
                network.addLayer(new BasicDropoutLayer(new ActivationLinear(), true, numHiddenNode, dropoutRate));
            } else if(func.equalsIgnoreCase(NNConstants.NN_SIGMOID)) {
                network.addLayer(new BasicDropoutLayer(new ActivationSigmoid(), true, numHiddenNode, dropoutRate));
            } else if(func.equalsIgnoreCase(NNConstants.NN_TANH)) {
                network.addLayer(new BasicDropoutLayer(new ActivationTANH(), true, numHiddenNode, dropoutRate));
            } else if(func.equalsIgnoreCase(NNConstants.NN_LOG)) {
                network.addLayer(new BasicDropoutLayer(new ActivationLOG(), true, numHiddenNode, dropoutRate));
            } else if(func.equalsIgnoreCase(NNConstants.NN_SIN)) {
                network.addLayer(new BasicDropoutLayer(new ActivationSIN(), true, numHiddenNode, dropoutRate));
            } else if(func.equalsIgnoreCase(NNConstants.NN_RELU)) {
                network.addLayer(new BasicDropoutLayer(new ActivationReLU(), true, numHiddenNode, dropoutRate));
            } else if(func.equalsIgnoreCase(NNConstants.NN_LEAKY_RELU)) {
                network.addLayer(new BasicDropoutLayer(new ActivationLeakyReLU(), true, numHiddenNode, dropoutRate));
            } else if(func.equalsIgnoreCase(NNConstants.NN_SWISH)) {
                network.addLayer(new BasicDropoutLayer(new ActivationSwish(), true, numHiddenNode, dropoutRate));
            } else if(func.equalsIgnoreCase(NNConstants.NN_PTANH)) {
                network.addLayer(new BasicDropoutLayer(new ActivationPTANH(), true, numHiddenNode, dropoutRate));
            } else {
                network.addLayer(new BasicDropoutLayer(new ActivationSigmoid(), true, numHiddenNode, dropoutRate));
            }
        }

        if(isLinearTarget) {
            if(NNConstants.NN_RELU.equalsIgnoreCase(outputActivationFunc)) {
                network.addLayer(new BasicLayer(new ActivationReLU(), true, out));
            } else if(NNConstants.NN_LEAKY_RELU.equalsIgnoreCase(outputActivationFunc)) {
                network.addLayer(new BasicLayer(new ActivationLeakyReLU(), true, out));
            } else if(NNConstants.NN_SWISH.equalsIgnoreCase(outputActivationFunc)) {
                network.addLayer(new BasicLayer(new ActivationSwish(), true, out));
            } else {
                network.addLayer(new BasicLayer(new ActivationLinear(), true, out));
            }
        } else {
            network.addLayer(new BasicLayer(new ActivationSigmoid(), false, out));
        }

        NeuralStructure structure = network.getStructure();
        if(network.getStructure() instanceof FloatNeuralStructure) {
            ((FloatNeuralStructure) structure).finalizeStruct();
        } else {
            structure.finalizeStructure();
        }
        if(isRandomizeWeights) {
            if(wgtInit == null || wgtInit.length() == 0) {
                // default randomization
                network.reset();
            } else if(wgtInit.equalsIgnoreCase(WGT_INIT_GAUSSIAN)) {
                new GaussianRandomizer(0, 1).randomize(network);
            } else if(wgtInit.equalsIgnoreCase(WGT_INIT_XAVIER)) {
                new XavierWeightRandomizer().randomize(network);
            } else if(wgtInit.equalsIgnoreCase(WGT_INIT_HE)) {
                new HeWeightRandomizer().randomize(network);
            } else if(wgtInit.equalsIgnoreCase(WGT_INIT_LECUN)) {
                new LecunWeightRandomizer().randomize(network);
            } else if(wgtInit.equalsIgnoreCase(WGT_INIT_DEFAULT)) {
                // default randomization
                network.reset();
            } else {
                // default randomization
                network.reset();
            }
        }

        return network;
    }

    public static boolean isExtremeLearningMachinePropagation(String propagation) {
        return propagation != null && "E".equals(propagation);
    }

    /**
     * Determine the sign of the value.
     * 
     * @param value
     *            The value to check.
     * @return -1 if less than zero, 1 if greater, or 0 if zero.
     */
    public static int sign(final double value) {
        if(Math.abs(value) < Encog.DEFAULT_DOUBLE_EQUAL) {
            return 0;
        } else if(value > 0) {
            return 1;
        } else {
            return -1;
        }
    }

    public static void randomize(int seed, double[] weights) {
        NguyenWidrowRandomizer randomizer = new NguyenWidrowRandomizer(-1, 1);
        randomizer.randomize(weights);
    }

    /**
     * Generate random instance according to sample seed.
     * 
     * @param sampleSeed
     *            sample seed to generate Random instance
     * @param fallbackValue
     *            sample seed fall back value
     * @return Random instance according to the sample seed value
     *         If the sample seed value not equal to fallbackValue, then will use it to generate Random instance.
     *         Else take fallback measure: generate Random instance without given seed.
     */
    public static Random generateRandomBySampleSeed(long sampleSeed, long fallbackValue) {
        if(sampleSeed != fallbackValue) {
            return new Random(sampleSeed);
        }
        return new Random();
    }

    public static int getFeatureInputsCnt(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            Set<Integer> featureSet) {
        if(modelConfig.getNormalizeType().equals(ModelNormalizeConf.NormType.ONEHOT)) {
            int inputCount = 0;
            for(ColumnConfig columnConfig: columnConfigList) {
                if(featureSet.contains(columnConfig.getColumnNum())) {
                    if(columnConfig.isNumerical()) {
                        inputCount += (columnConfig.getBinBoundary().size() + 1);
                    } else {
                        inputCount += (columnConfig.getBinCategory().size() + 1);
                    }
                }
            }
            return inputCount;
        } else if(modelConfig.getNormalizeType().equals(ModelNormalizeConf.NormType.ZSCALE_ONEHOT)) {
            int inputCount = 0;
            for(ColumnConfig columnConfig: columnConfigList) {
                if(featureSet.contains(columnConfig.getColumnNum())) {
                    if(columnConfig.isNumerical()) {
                        inputCount += 1;
                    } else {
                        inputCount += (columnConfig.getBinCategory().size() + 1);
                    }
                }
            }
            return inputCount;
        } else {
            return featureSet.size();
        }
    }

    /**
     * Get Double property value from map.
     * If the value doesn't exist in the Map or the format is incorrect, use @defval as default
     * 
     * @param params
     *            input Map
     * @param key
     *            the key to look up
     * @param defval
     *            default value, if the key is not in the map or the value format is illegal
     * @return
     *         Double value if the key exists and value format is correct
     *         or defval
     */
    public static Double getDouble(Map<?, ?> params, String key, Double defval) {
        Double val = defval;
        if(MapUtils.isNotEmpty(params) && params.containsKey(key)) {
            Object obj = params.get(key);
            if(obj != null) {
                try {
                    val = Double.valueOf(org.apache.commons.lang3.StringUtils.trimToEmpty(obj.toString()));
                } catch (Exception e) {
                    LOG.warn("Export double value for {} in params, but got {}", key, obj, e);
                }
            }
        }
        return val;
    }

    /**
     * Get Boolean property value from map.
     * If the value doesn't exist in the Map or the format is incorrect, use @defval as default
     * 
     * @param params
     *            input Map
     * @param key
     *            the key to look up
     * @param defval
     *            default value, if the key is not in the map or the value format is illegal
     * @return
     *         Boolean value if the key exists and value format is correct
     *         or defval
     */
    public static Boolean getBoolean(Map<?, ?> params, String key, Boolean defval) {
        Boolean val = defval;
        if(MapUtils.isNotEmpty(params) && params.containsKey(key)) {
            Object obj = params.get(key);
            if(obj != null) {
                try {
                    val = Boolean.valueOf(StringUtils.trimToEmpty(obj.toString()));
                } catch (Exception e) {
                    LOG.warn("Export boolean value for {} in params, but got {}", key, obj, e);
                }
            }
        }
        return val;
    }

    /**
     * Get Integer property value from map.
     * If the value doesn't exist in the Map or the format is incorrect, use @defval as default
     * 
     * @param params
     *            input Map
     * @param key
     *            the key to look up
     * @param defval
     *            default value, if the key is not in the map or the value format is illegal
     * @return
     *         Integer value if the key exists and value format is correct
     *         or defval
     */
    @SuppressWarnings("rawtypes")
    public static Integer getInt(Map params, String key, Integer defval) {
        Integer val = defval;
        if(MapUtils.isNotEmpty(params) && params.containsKey(key)) {
            Object obj = params.get(key);
            if(obj != null) {
                try {
                    val = Integer.valueOf(StringUtils.trimToEmpty(obj.toString()));
                } catch (Exception e) {
                    LOG.warn("Export int value for {} in params, but got {}", key, obj, e);
                }
            }
        }
        return val;
    }

    /**
     * Bin size of all variables, categorical means number of all categories, numerical variable means number of bins.
     * 
     * @param columnConfigList
     *            the column config list of the model
     * @return the map mapping from column Id to bin category list size
     */
    public static Map<Integer, Integer> getIdBinCategorySizeMap(List<ColumnConfig> columnConfigList) {
        Map<Integer, Integer> idBinCategoryMap = new HashMap<>(columnConfigList.size());
        for(ColumnConfig columnConfig: columnConfigList) {
            if(columnConfig.getBinCategory() != null) {
                idBinCategoryMap.put(columnConfig.getColumnNum(), columnConfig.getBinCategory().size());
            } else if(columnConfig.getBinBoundary() != null) {
                idBinCategoryMap.put(columnConfig.getColumnNum(), columnConfig.getBinBoundary().size());
            } else {
                idBinCategoryMap.put(columnConfig.getColumnNum(), 0);
            }
        }
        return idBinCategoryMap;
    }

    /**
     * Whether there is any final select variables or not
     * 
     * @param columnConfigList
     *            - the model column config list
     * @return true - has final selected variables, or false
     */
    public static boolean hasFinalSelectedVars(List<ColumnConfig> columnConfigList) {
        boolean hasFinalSelectedVars = false;
        for(ColumnConfig columnConfig: columnConfigList) {
            if(columnConfig.isFinalSelect()) {
                hasFinalSelectedVars = true;
                break;
            }
        }
        return hasFinalSelectedVars;
    }

    /**
     * Get all the feature IDs that could be used to train model
     * 
     * @param columnConfigList
     *            - the model column config list
     * @param hasCandidates
     *            - user specify candidate variables or not
     * @return - feature IDs that could be used to train model
     */
    public static Set<Integer> getModelFeatureSet(List<ColumnConfig> columnConfigList, boolean hasCandidates) {
        return getModelFeatureSet(columnConfigList, hasFinalSelectedVars(columnConfigList), hasCandidates);
    }

    /**
     * Get all the feature IDs that could be used to train model
     * 
     * @param columnConfigList
     *            - the model column config list
     * @param hasSelectedVars
     *            - there is selected variables in ColumnConfig.json or not
     * @param hasCandidates
     *            - user specify candidate variables or not
     * @return - feature IDs that could be used to train model
     */
    public static Set<Integer> getModelFeatureSet(List<ColumnConfig> columnConfigList, boolean hasSelectedVars,
            boolean hasCandidates) {
        Set<Integer> featureSet = new HashSet<>();
        for(ColumnConfig columnConfig: columnConfigList) {
            if(hasSelectedVars) {
                if(columnConfig.isFinalSelect()) {
                    featureSet.add(columnConfig.getColumnNum());
                }
            } else {
                // should we call CommonUtils.isGoodCandidate(columnConfig, hasCandidates, isBinaryClassification) ?
                if(CommonUtils.isGoodCandidate(columnConfig, hasCandidates)) {
                    featureSet.add(columnConfig.getColumnNum());
                }
            }
        }
        return featureSet;
    }

    /**
     * Parse the field of normalized data
     * 
     * @param fields
     *            - normalized data array
     * @param dataPos
     *            - the position of data element
     * @param defVal
     *            - the default value, if the data element couldn't be parsed
     * @return the float value of data element
     */
    public static float parseRawNormValue(String[] fields, int dataPos, float defVal) {
        if(dataPos >= fields.length) { // out of range, when fetching normalization data element
            LOG.error("Normalization data set doesn't match. Out of Range {}/{}", dataPos, fields.length);
            throw new RuntimeException("Out of range Normalization data doesn't match with ColumnConfig.json.");
        }

        String input = fields[dataPos];
        // check here to avoid bad performance in failed NumberFormatUtils.getFloat(input, 0f)
        float fval = ((input.length() == 0) ? defVal : NumberFormatUtils.getFloat(input, defVal));
        // no idea about why NaN in input data, we should process it as missing value
        // TODO , according to norm type
        fval = (Float.isNaN(fval) || Double.isNaN(fval)) ? defVal : fval;
        return fval;
    }

    /**
     * Parse the field of normalized data
     * 
     * @param tuple
     *            - data tuple of normalized data
     * @param dataPos
     *            - the position of data element
     * @param defVal
     *            - the default value, if the data element couldn't be parsed
     * @return the float value of data element
     */
    public static float parseRawNormValue(Tuple tuple, int dataPos, float defVal) {
        if(dataPos >= tuple.size()) { // out of range, when fetching normalization data element
            LOG.error("Normalization data set doesn't match. Out of Range {}/{}", dataPos, tuple.size());
            throw new RuntimeException("Out of range Normalization data doesn't match with ColumnConfig.json.");
        }

        Object element = null;
        try {
            element = tuple.get(dataPos);
        } catch (ExecException e) {
            throw new GuaguaRuntimeException(e);
        }
        float fval = 0.0f;
        if(element != null) {
            if(element instanceof Float) {
                fval = (Float) element;
            } else {
                // check here to avoid bad performance in failed NumberFormatUtils.getFloat(input, 0f)
                fval = (element.toString().length() == 0) ? defVal
                        : NumberFormatUtils.getFloat(element.toString(), defVal);
            }
        }

        // no idea about why NaN in input data, we should process it as missing value
        // TODO , according to norm type
        fval = (Float.isNaN(fval) || Double.isNaN(fval)) ? defVal : fval;
        return fval;
    }

    public static double loadDataIntoInputs(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            Set<Integer> featureSet, boolean isLinearTarget, boolean hasCandidates, double[] inputs, double outputs[],
            String[] rawInputs) {
        int dataPos = 0, inputsIndex = 0, outputIndex = 0;
        for(ColumnConfig columnConfig: columnConfigList) {
            float fval = DTrainUtils.parseRawNormValue(rawInputs, dataPos, 0.0f);
            if(columnConfig.isTarget()) { // target
                if(isLinearTarget || modelConfig.isRegression()) {
                    outputs[outputIndex++] = fval;
                } else {
                    // for multi-classification
                }
                dataPos++;
            } else { // other variables
                if(featureSet.contains(columnConfig.getColumnNum())) {
                    if(columnConfig.isMeta() || columnConfig.isForceRemove()) {
                        // it shouldn't happen here
                        dataPos += 1;
                    } else if(columnConfig != null && columnConfig.isNumerical()
                            && modelConfig.getNormalizeType().equals(ModelNormalizeConf.NormType.ONEHOT)) {
                        for(int k = 0; k < columnConfig.getBinBoundary().size() + 1; k++) {
                            float tval = DTrainUtils.parseRawNormValue(rawInputs, dataPos, 0.0f);
                            inputs[inputsIndex++] = tval;
                            dataPos++;
                        }
                    } else if(columnConfig != null && columnConfig.isCategorical()
                            && (modelConfig.getNormalizeType().equals(ModelNormalizeConf.NormType.ZSCALE_ONEHOT)
                                    || modelConfig.getNormalizeType().equals(ModelNormalizeConf.NormType.ONEHOT))) {
                        for(int k = 0; k < columnConfig.getBinCategory().size() + 1; k++) {
                            float tval = DTrainUtils.parseRawNormValue(rawInputs, dataPos, 0.0f);
                            inputs[inputsIndex++] = tval;
                            dataPos++;
                        }
                    } else {
                        inputs[inputsIndex++] = fval;
                        dataPos++;
                    }
                } else { // just skip unused data in normalized data
                    if(!CommonUtils.isToNormVariable(columnConfig, hasCandidates, modelConfig.isRegression())) {
                        dataPos += 1;
                    } else if(columnConfig.isNumerical()
                            && modelConfig.getNormalizeType().equals(ModelNormalizeConf.NormType.ONEHOT)
                            && columnConfig.getBinBoundary() != null && columnConfig.getBinBoundary().size() > 0) {
                        dataPos += (columnConfig.getBinBoundary().size() + 1);
                    } else if(columnConfig.isCategorical()
                            && (modelConfig.getNormalizeType().equals(ModelNormalizeConf.NormType.ZSCALE_ONEHOT)
                                    || modelConfig.getNormalizeType().equals(ModelNormalizeConf.NormType.ONEHOT))
                            && columnConfig.getBinCategory().size() > 0) {
                        dataPos += (columnConfig.getBinCategory().size() + 1);
                    } else {
                        dataPos += 1;
                    }
                }
            }
        }

        return DTrainUtils.parseRawNormValue(rawInputs, rawInputs.length - 1, 1.0f);
    }

    public static Set<Integer> generateModelFeatureSet(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        Set<Integer> columnIdSet = new HashSet<>();
        boolean hasFinalSelectedVars = DTrainUtils.hasFinalSelectedVars(columnConfigList);
        if(hasFinalSelectedVars) {
            columnConfigList.stream().forEach(columnConfig -> {
                if(columnConfig.isFinalSelect()) {
                    columnIdSet.add(columnConfig.getColumnNum());
                }
            });
        } else {
            boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);
            columnConfigList.stream().forEach(columnConfig -> {
                if(CommonUtils.isGoodCandidate(columnConfig, hasCandidates, modelConfig.isRegression())) {
                    columnIdSet.add(columnConfig.getColumnNum());
                }
            });
        }
        return columnIdSet;
    }

    public static int generateFeatureInputInfo(ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            Set<Integer> featureSet, Map<Integer, List<Double>> columnMissingInputValues,
            Map<Integer, Integer> columnNormDataPosMapping) {
        int vectorLen = 0;
        for(ColumnConfig columnConfig: columnConfigList) {
            if(featureSet.contains(columnConfig.getColumnNum())) {
                columnNormDataPosMapping.put(columnConfig.getColumnNum(), vectorLen);
                List<Double> normValues = Normalizer.normalize(columnConfig, null,
                        modelConfig.getNormalizeStdDevCutOff(), modelConfig.getNormalizeType(),
                        CategoryMissingNormType.MEAN);
                if(CollectionUtils.isNotEmpty(normValues)) { // usually, the normValues won't be empty
                    columnMissingInputValues.put(columnConfig.getColumnNum(), normValues);
                    vectorLen += normValues.size();
                }
            }
        }
        return vectorLen;
    }
}
