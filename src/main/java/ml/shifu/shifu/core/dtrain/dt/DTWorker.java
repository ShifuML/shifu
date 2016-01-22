/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.dt;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Random;

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
import ml.shifu.shifu.core.dtrain.dt.DTWorkerParams.NodeStats;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.math3.distribution.PoissonDistribution;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Splitter;

/**
 * TODO
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class DTWorker
        extends
        AbstractWorkerComputable<DTMasterParams, DTWorkerParams, GuaguaWritableAdapter<LongWritable>, GuaguaWritableAdapter<Text>> {

    protected static final Logger LOG = LoggerFactory.getLogger(DTWorker.class);

    /**
     * Model configuration loaded from configuration file.
     */
    private ModelConfig modelConfig;

    /**
     * Column configuration loaded from configuration file.
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * Total tree numbers
     */
    private int treeNum;

    /**
     * Basic numeric input count
     */
    protected int numericInputCount;

    /**
     * Basic categorical input count
     */
    protected int categoricalInputCount;

    /**
     * Means if do variable selection, if done, many variables will be set to finalSelect = true; if not, no variables
     * are selected and should be set to all good candidate variables.
     */
    private boolean isAfterVarSelect = true;

    /**
     * Basic output node count for NN model
     */
    protected int outputNodeCount;

    /**
     * {@link #candidateCount} is used to check if no variable is selected. If {@link #inputNodeCount} equals
     * {@link #candidateCount}, which means no column is selected or all columns are selected.
     */
    protected int candidateCount;

    /**
     * input record size, inc one by one.
     */
    protected long count;

    /**
     * sampled input record size.
     */
    protected long sampleCount;

    /**
     * Training data set.
     */
    private BytableMemoryDiskList<Data> trainingData;

    /**
     * PoissonDistribution which is used for possion sampling for bagging with replacement.
     */
    protected PoissonDistribution[] rng = null;

    /**
     * If tree number = 1, no need bagging with replacement.
     */
    private Random random = new Random();

    /**
     * Default splitter used to split input record. Use one instance to prevent more news in Splitter.on.
     */
    protected static final Splitter DEFAULT_SPLITTER = Splitter.on(CommonConstants.DEFAULT_COLUMN_SEPARATOR)
            .trimResults();

    /**
     * Index map in which column index and numeric input array index for fast location.
     */
    private Map<Integer, Integer> numericInputIndexMap = new HashMap<Integer, Integer>();

    /**
     * Index map in which column index and categorical input array index for fast location.
     */
    private Map<Integer, Integer> categoricalInputIndexMap = new HashMap<Integer, Integer>();

    private Map<Integer, Map<String, Integer>> categoryIndexMap = new HashMap<Integer, Map<String, Integer>>();

    private Impurity impurity;

    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        super.setRecordReader(new GuaguaLineRecordReader(fileSplit));
    }

    @Override
    public void init(WorkerContext<DTMasterParams, DTWorkerParams> context) {
        Properties props = context.getProps();
        try {
            SourceType sourceType = SourceType.valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE,
                    SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);
            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    props.getProperty(CommonConstants.SHIFU_COLUMN_CONFIG), sourceType);
            for(ColumnConfig config: this.columnConfigList) {
                if(config.isCategorical()) {
                    Map<String, Integer> categoryMap = new HashMap<String, Integer>();
                    for(int i = 0; i < config.getBinCategory().size(); i++) {
                        categoryMap.put(config.getBinCategory().get(i), i);
                    }
                    this.categoryIndexMap.put(config.getColumnNum(), categoryMap);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        this.treeNum = this.modelConfig.getTrain().getBaggingNum();

        double memoryFraction = Double.valueOf(context.getProps().getProperty("guagua.data.memoryFraction", "0.6"));
        LOG.info("Max heap memory: {}, fraction: {}", Runtime.getRuntime().maxMemory(), memoryFraction);
        String tmpFolder = context.getProps().getProperty("guagua.data.tmpfolder", "tmp");
        this.trainingData = new BytableMemoryDiskList<Data>((long) (Runtime.getRuntime().maxMemory() * memoryFraction),
                tmpFolder + File.separator + "train-" + System.currentTimeMillis());

        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
        this.numericInputCount = inputOutputIndex[0];
        this.categoricalInputCount = inputOutputIndex[1];
        this.outputNodeCount = modelConfig.isBinaryClassification() ? inputOutputIndex[2] : modelConfig.getTags()
                .size();
        this.isAfterVarSelect = inputOutputIndex[3] == 1 ? true : false;

        this.rng = new PoissonDistribution[treeNum];
        for(int i = 0; i < treeNum; i++) {
            this.rng[i] = new PoissonDistribution(this.modelConfig.getTrain().getBaggingSampleRate());
        }

        int numClasses = this.modelConfig.isMultiClassification() ? this.modelConfig.getFlattenTags().size() : 2;
        String imStr = this.modelConfig.getTrain().getParams().get("Impurity").toString();
        if(imStr.equalsIgnoreCase("entropy")) {
            impurity = new Entropy(numClasses);
        } else if(imStr.equalsIgnoreCase("gini")) {
            impurity = new Gini(numClasses);
        } else {
            impurity = new Variance();
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.worker.AbstractWorkerComputable#doCompute(ml.shifu.guagua.worker.WorkerContext)
     */
    @Override
    public DTWorkerParams doCompute(WorkerContext<DTMasterParams, DTWorkerParams> context) {
        if(context.isFirstIteration()) {
            return new DTWorkerParams();
        }
        DTMasterParams lastMasterResult = context.getLastMasterResult();
        List<TreeNode> trees = lastMasterResult.getTrees();
        Map<Integer, TreeNode> todoNodes = lastMasterResult.getTodoNodes();

        Map<Integer, NodeStats> statistics = new HashMap<Integer, NodeStats>(todoNodes.size(), 1f);
        for(Map.Entry<Integer, TreeNode> entry: todoNodes.entrySet()) {
            List<Integer> features = entry.getValue().getFeatures();
            if(features.isEmpty()) {
                features = getAllValidFeatures();
            }
            Map<Integer, double[]> featureStatistics = new HashMap<Integer, double[]>(features.size(), 1f);
            for(Integer columnNum: features) {
                ColumnConfig columnConfig = this.columnConfigList.get(columnNum);
                if(columnConfig.isNumerical()) {
                    // TODO, how to process null bin
                    int featureStatsSize = columnConfig.getBinBoundary().size() * this.impurity.getStatsSize();
                    featureStatistics.put(columnNum, new double[featureStatsSize]);
                } else if(columnConfig.isCategorical()) {
                    int featureStatsSize = (columnConfig.getBinCategory().size() + 1) * this.impurity.getStatsSize();
                    featureStatistics.put(columnNum, new double[featureStatsSize]);
                }
            }
            statistics.put(entry.getKey(), new NodeStats(entry.getValue().getTreeId(), entry.getValue().getNode()
                    .getId(), featureStatistics));
        }
        // reopen for iteration
        this.trainingData.reOpen();
        double squareError = 0d;
        for(Data data: this.trainingData) {
            List<Integer> nodeIndexes = new ArrayList<Integer>(trees.size());
            for(TreeNode treeNode: trees) {
                Node predictNode = predictNodeIndex(treeNode.getNode(), data);
                if(predictNode.getPredict() != null) {
                    // only update when not in first node, for treeNode, no predict statistics at that time
                    double error = data.outputs[0] - predictNode.getPredict().getPredict();
                    squareError += error * error;
                }
                int predictNodeIndex = predictNode.getId();
                nodeIndexes.add(predictNodeIndex);
            }
            for(Map.Entry<Integer, TreeNode> entry: todoNodes.entrySet()) {
                // only do statistics on effective data
                Node todoNode = entry.getValue().getNode();
                int treeId = entry.getValue().getTreeId();
                if(todoNode.getId() == nodeIndexes.get(entry.getValue().getTreeId())) {
                    List<Integer> features = entry.getValue().getFeatures();
                    if(features.isEmpty()) {
                        features = getAllValidFeatures();
                    }
                    for(Integer columnNum: features) {
                        ColumnConfig config = this.columnConfigList.get(columnNum);
                        double[] featuerStatistic = statistics.get(entry.getKey()).getFeatureStatistics()
                                .get(columnNum);
                        float weight = data.subsampleWeights[treeId];
                        if(config.isNumerical()) {
                            float value = data.numericInputs[this.numericInputIndexMap.get(columnNum)];
                            int binIndex = getBinIndex(value, config.getBinBoundary());
                            this.impurity.featureUpdate(featuerStatistic, binIndex, data.outputs[0], data.significance,
                                    weight);
                        } else if(config.isCategorical()) {
                            String category = data.categoricalInputs[this.categoricalInputIndexMap.get(columnNum)];
                            Integer binIndex = this.categoryIndexMap.get(columnNum).get(category);
                            this.impurity.featureUpdate(featuerStatistic, binIndex, data.outputs[0], data.significance,
                                    weight);
                        } else {
                            throw new IllegalStateException("Only numerical and categorical columns supported. ");
                        }
                    }
                }
            }
        }
        for(Map.Entry<Integer, NodeStats> entry: statistics.entrySet()) {
            NodeStats nodeStats = entry.getValue();
            LOG.info("Node index {}, node id {}, tree id{}", entry.getKey(), nodeStats.getNodeId(),
                    nodeStats.getTreeId());
            Map<Integer, double[]> featureStatistics = nodeStats.getFeatureStatistics();
            for(Entry<Integer, double[]> feaEntry: featureStatistics.entrySet()) {
                LOG.info("ColumnNum {} statistics {}", feaEntry.getKey(), Arrays.toString(feaEntry.getValue()));
            }
        }
        LOG.debug("Worker statistics is {}", statistics);
        return new DTWorkerParams(count, squareError, statistics);
    }

    @Override
    protected void postLoad(WorkerContext<DTMasterParams, DTWorkerParams> context) {
        this.trainingData.switchState();
    }

    private List<Integer> getAllValidFeatures() {
        List<Integer> features = new ArrayList<Integer>();
        for(ColumnConfig config: columnConfigList) {
            if(isAfterVarSelect) {
                if(config.isFinalSelect() && !config.isTarget() && !config.isMeta()) {
                    features.add(config.getColumnNum());
                }
            } else {
                if(!config.isMeta() && !config.isTarget() && CommonUtils.isGoodCandidate(config)) {
                    features.add(config.getColumnNum());
                }
            }
        }
        return features;
    }

    /**
     * 'binBoundary' is ArrayList in fact, so we can use get method. ["-Infinity", 1d, 4d, ....]
     */
    private int getBinIndex(float value, List<Double> binBoundary) {
        if(binBoundary.size() <= 1) {
            throw new IllegalArgumentException();
        }

        int low = 0, high = binBoundary.size() - 1;
        while(low <= high) {
            int mid = (low + high) >>> 1;
            double lowThreshold = binBoundary.get(mid);
            double highThreshold = mid == binBoundary.size() - 1 ? Double.MAX_VALUE : binBoundary.get(mid + 1);
            if(value >= lowThreshold && value < highThreshold) {
                return mid;
            }
            if(value >= highThreshold) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return -1;
    }

    private Node predictNodeIndex(Node node, Data data) {
        Node currNode = node;
        Split split = currNode.getSplit();
        if(split == null || currNode.isLeaf()) {
            return currNode;
        }

        ColumnConfig columnConfig = this.columnConfigList.get(split.getColumnNum());

        Node nextNode = null;
        if(columnConfig.isNumerical()) {
            float value = data.numericInputs[this.numericInputIndexMap.get(split.getColumnNum())];
            if(value <= split.getThreshold()) {
                nextNode = currNode.getLeft();
            } else {
                nextNode = currNode.getRight();
            }
        } else if(columnConfig.isCategorical()) {
            String value = data.categoricalInputs[this.categoricalInputIndexMap.get(split.getColumnNum())];
            if(split.getLeftCategories().contains(value)) {
                nextNode = currNode.getLeft();
            } else {
                nextNode = currNode.getRight();
            }
        }

        assert nextNode != null;
        return predictNodeIndex(nextNode, data);
    }

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<DTMasterParams, DTWorkerParams> context) {
        this.count += 1;
        if((this.count) % 100000 == 0) {
            LOG.info("Read {} records.", this.count);
        }

        float[] numericInputs = new float[this.numericInputCount];
        String[] categoricalInputs = new String[this.categoricalInputCount];
        float[] ideal = new float[this.outputNodeCount];

        float significance = 1f;
        // use guava Splitter to iterate only once
        // use NNConstants.NN_DEFAULT_COLUMN_SEPARATOR to replace getModelConfig().getDataSetDelimiter(), super follows
        // the function in akka mode.
        int index = 0, numericInputsIndex = 0, outputIndex = 0, categoricalInputsIndex = 0;
        for(String input: DEFAULT_SPLITTER.split(currentValue.getWritable().toString())) {
            float floatValue = NumberFormatUtils.getFloat(input, 0f);
            // no idea about why NaN in input data, we should process it as missing value TODO, according to norm type
            if(Float.isNaN(floatValue) || Double.isNaN(floatValue)) {
                floatValue = 0f;
            }
            if(index == this.columnConfigList.size()) {
                significance = NumberFormatUtils.getFloat(input, 1f);
                // the last field is significance, break here
                break;
            } else {
                ColumnConfig columnConfig = this.columnConfigList.get(index);
                if(columnConfig != null && columnConfig.isTarget()) {
                    if(modelConfig.isBinaryClassification()) {
                        ideal[outputIndex++] = floatValue;
                    } else {
                        int ideaIndex = (int) floatValue;
                        ideal[ideaIndex] = 1f;
                    }
                } else {
                    if(!isAfterVarSelect) {
                        // no variable selected, good candidate but not meta and not target chose
                        if(!columnConfig.isMeta() && !columnConfig.isTarget()
                                && CommonUtils.isGoodCandidate(columnConfig)) {
                            if(columnConfig.isNumerical()) {
                                numericInputs[numericInputsIndex] = floatValue;
                                this.numericInputIndexMap.put(columnConfig.getColumnNum(), numericInputsIndex);
                                numericInputsIndex += 1;
                            } else if(columnConfig.isCategorical()) {
                                if(input == null) {
                                    // use empty to replace null categories
                                    categoricalInputs[categoricalInputsIndex] = "";
                                } else {
                                    categoricalInputs[categoricalInputsIndex] = input;
                                }
                                this.categoricalInputIndexMap.put(columnConfig.getColumnNum(), categoricalInputsIndex);
                                categoricalInputsIndex += 1;
                            }
                        }
                    } else {
                        // final select some variables but meta and target are not included
                        if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                                && columnConfig.isFinalSelect()) {
                            if(columnConfig.isNumerical()) {
                                numericInputs[numericInputsIndex] = floatValue;
                                this.numericInputIndexMap.put(columnConfig.getColumnNum(), numericInputsIndex);
                                numericInputsIndex += 1;
                            } else if(columnConfig.isCategorical()) {
                                if(input == null) {
                                    // use empty to replace null categories
                                    categoricalInputs[categoricalInputsIndex] = "";
                                } else {
                                    categoricalInputs[categoricalInputsIndex] = input;
                                }
                                this.categoricalInputIndexMap.put(columnConfig.getColumnNum(), categoricalInputsIndex);
                                categoricalInputsIndex += 1;
                            }
                        }
                    }
                }
            }
            index += 1;
        }

        float[] sampleWeights;
        if(this.treeNum == 1) {
            if(random.nextDouble() <= modelConfig.getTrain().getBaggingSampleRate()) {
                sampleWeights = new float[] { 1.0f };
            } else {
                sampleWeights = new float[] { 0.0f };
            }
        } else {
            sampleWeights = new float[this.treeNum];
            for(int i = 0; i < sampleWeights.length; i++) {
                sampleWeights[i] = this.rng[i].sample();
            }
        }
        this.trainingData.append(new Data(numericInputs, categoricalInputs, ideal, significance, sampleWeights));
    }

    private static class Data implements Serializable, Bytable {

        private static final long serialVersionUID = 903201066309036170L;

        private float[] numericInputs;
        private String[] categoricalInputs;
        private float[] outputs;
        private float significance;
        private float[] subsampleWeights = new float[] { 1.0f };

        public Data(float[] numericInputs, String[] categoricalInputs, float[] outputs, float significance,
                float[] subsampleWeights) {
            this.numericInputs = numericInputs;
            this.categoricalInputs = categoricalInputs;
            this.outputs = outputs;
            this.significance = significance;
            this.subsampleWeights = subsampleWeights;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeInt(numericInputs.length);
            for(float input: numericInputs) {
                out.writeFloat(input);
            }

            out.writeInt(categoricalInputs.length);
            for(String input: categoricalInputs) {
                // TODO if writeUTF, readUTF is right???
                out.writeUTF(input);
            }

            out.writeInt(outputs.length);
            for(float output: outputs) {
                out.writeFloat(output);
            }

            out.writeFloat(significance);

            out.writeInt(subsampleWeights.length);
            for(float sample: subsampleWeights) {
                out.writeFloat(sample);
            }
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            int iLen = in.readInt();
            this.numericInputs = new float[iLen];
            for(int i = 0; i < iLen; i++) {
                this.numericInputs[i] = in.readFloat();
            }

            int cLen = in.readInt();
            this.categoricalInputs = new String[cLen];
            for(int i = 0; i < cLen; i++) {
                this.categoricalInputs[i] = in.readUTF();
            }

            int oLen = in.readInt();
            this.outputs = new float[oLen];
            for(int i = 0; i < oLen; i++) {
                this.outputs[i] = in.readFloat();
            }

            this.significance = in.readFloat();

            int sLen = in.readInt();
            this.subsampleWeights = new float[sLen];
            for(int i = 0; i < sLen; i++) {
                this.subsampleWeights[i] = in.readFloat();
            }
        }

    }

}
