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
import java.util.List;
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
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
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

    private int treeNum;

    /**
     * Basic input node count for NN model
     */
    protected int inputNodeCount;

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
    protected static final Splitter DEFAULT_SPLITTER = Splitter.on(NNConstants.NN_DEFAULT_COLUMN_SEPARATOR);

    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        super.setRecordReader(new GuaguaLineRecordReader(fileSplit));
    }

    @Override
    public void init(WorkerContext<DTMasterParams, DTWorkerParams> context) {
        Properties props = context.getProps();
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

        this.treeNum = this.modelConfig.getTrain().getBaggingNum();

        double memoryFraction = Double.valueOf(context.getProps().getProperty("guagua.data.memoryFraction", "0.6"));
        LOG.info("Max heap memory: {}, fraction: {}", Runtime.getRuntime().maxMemory(), memoryFraction);
        String tmpFolder = context.getProps().getProperty("guagua.data.tmpfolder", "tmp");
        this.trainingData = new BytableMemoryDiskList<Data>((long) (Runtime.getRuntime().maxMemory() * memoryFraction),
                tmpFolder + File.separator + "train-" + System.currentTimeMillis());

        int[] inputOutputIndex = DTrainUtils.getInputOutputCandidateCounts(this.columnConfigList);
        this.inputNodeCount = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
        this.outputNodeCount = modelConfig.isBinaryClassification() ? inputOutputIndex[1] : modelConfig.getTags()
                .size();
        this.candidateCount = inputOutputIndex[2];

        this.rng = new PoissonDistribution[treeNum];
        for(int i = 0; i < treeNum; i++) {
            this.rng[i] = new PoissonDistribution(this.modelConfig.getTrain().getBaggingSampleRate());
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.worker.AbstractWorkerComputable#doCompute(ml.shifu.guagua.worker.WorkerContext)
     */
    @Override
    public DTWorkerParams doCompute(WorkerContext<DTMasterParams, DTWorkerParams> context) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<DTMasterParams, DTWorkerParams> context) {
        this.count += 1;
        if((this.count) % 100000 == 0) {
            LOG.info("Read {} records.", this.count);
        }

        float[] inputs = new float[this.inputNodeCount];
        float[] ideal = new float[this.outputNodeCount];

        float significance = 1f;
        // use guava Splitter to iterate only once
        // use NNConstants.NN_DEFAULT_COLUMN_SEPARATOR to replace getModelConfig().getDataSetDelimiter(), super follows
        // the function in akka mode.
        int index = 0, inputsIndex = 0, outputIndex = 0;
        for(String input: DEFAULT_SPLITTER.split(currentValue.getWritable().toString())) {
            float floatValue = NumberFormatUtils.getFloat(input.trim(), 0f);
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
                    if(this.inputNodeCount == this.candidateCount) {
                        // no variable selected, good candidate but not meta and not target choosed
                        if(!columnConfig.isMeta() && !columnConfig.isTarget()
                                && CommonUtils.isGoodCandidate(columnConfig)) {
                            inputs[inputsIndex++] = floatValue;
                        }
                    } else {
                        // final select some variables but meta and target are not included
                        if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                                && columnConfig.isFinalSelect()) {
                            inputs[inputsIndex++] = floatValue;
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
        this.trainingData.append(new Data(inputs, ideal, significance, sampleWeights));
    }

    private static class Data implements Serializable, Bytable {

        private static final long serialVersionUID = 903201066309036170L;

        private float[] inputs;
        private float[] outputs;
        private float significance;
        private float[] subsampleWeights = new float[] { 1.0f };

        public Data(float[] inputs, float[] outputs, float significance, float[] subsampleWeights) {
            this.inputs = inputs;
            this.outputs = outputs;
            this.significance = significance;
            this.subsampleWeights = subsampleWeights;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeInt(inputs.length);
            for(float input: inputs) {
                out.writeFloat(input);
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
            this.inputs = new float[iLen];
            for(int i = 0; i < iLen; i++) {
                this.inputs[i] = in.readFloat();
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
