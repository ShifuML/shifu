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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.dataset.FloatNeuralStructure;
import ml.shifu.shifu.core.dtrain.nn.*;
import ml.shifu.shifu.core.dtrain.random.HeWeightRandomizer;
import ml.shifu.shifu.core.dtrain.random.LecunWeightRandomizer;
import ml.shifu.shifu.core.dtrain.random.XavierWeightRandomizer;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.HDFSUtils;

import org.apache.commons.collections.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.fs.Path;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationLOG;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSIN;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.mathutil.randomize.GaussianRandomizer;
import org.encog.mathutil.randomize.NguyenWidrowRandomizer;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.structure.NeuralStructure;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
        @SuppressWarnings("unused")
        int input = 0, output = 0, totalCandidate = 0, goodCandidate = 0;
        boolean hasCandidate = CommonUtils.hasCandidateColumns(columnConfigList);
        for(ColumnConfig config: columnConfigList) {
            if(!config.isTarget() && !config.isMeta()) {
                totalCandidate += 1;
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

    public static List<Integer> getNumericalIds(List<ColumnConfig> columnConfigList, boolean isAfterVarSelect){
        List<Integer> numericalIds = new ArrayList<>();
        boolean hasCandidates = CommonUtils.hasCandidateColumns(columnConfigList);

        for(ColumnConfig config: columnConfigList) {
            if(isAfterVarSelect) {
                if(config.isNumerical() && config.isFinalSelect() && !config.isTarget() && !config.isMeta()) {
                    numericalIds.add(config.getColumnNum());
                }
            } else {
                if(config.isNumerical() && !config.isTarget() && !config.isMeta() &&
                        CommonUtils.isGoodCandidate(config, hasCandidates)) {
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
            } else if (func.equalsIgnoreCase(NNConstants.NN_PTANH)) {
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
}
