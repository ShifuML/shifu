/**
 * Copyright [2012-2014] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use super file except in compliance with the License.
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
import java.util.Arrays;

import ml.shifu.guagua.hadoop.io.GuaguaLineRecordReader;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.guagua.worker.WorkerContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;

/**
 * {@link NNWorker} is used to compute NN model according to splits assigned. The result will be sent to master for
 * accumulation.
 * 
 * <p>
 * Gradients in each worker will be sent to master to update weights of model in worker, which follows Encog's
 * multi-core implementation.
 * 
 * <p>
 * {@link NNWorker} is to load data with text format.
 */
public class NNWorker extends AbstractNNWorker<Text> {

    boolean isPrint = false;

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<NNParams, NNParams> workerContext) {
        super.count += 1;
        if((super.count) % 100000 == 0) {
            LOG.info("Read {} records.", super.count);
        }

        double baggingSampleRate = super.modelConfig.getBaggingSampleRate();
        // if fixInitialInput = false, we only compare random value with baggingSampleRate to avoid parsing data.
        // if fixInitialInput = true, we should use hashcode after parsing.
        if(!super.modelConfig.isFixInitialInput() && Double.compare(Math.random(), baggingSampleRate) >= 0) {
            return;
        }

        double[] inputs = new double[super.inputNodeCount];
        double[] ideal = new double[super.outputNodeCount];

        if(super.isDry) {
            // dry train, use empty data.
            addDataPairToDataSet(0, new BasicMLDataPair(new BasicMLData(inputs), new BasicMLData(ideal)));
            return;
        }

        long hashcode = 0;
        double significance = CommonConstants.DEFAULT_SIGNIFICANCE_VALUE;
        // use guava Splitter to iterate only once
        // use NNConstants.NN_DEFAULT_COLUMN_SEPARATOR to replace getModelConfig().getDataSetDelimiter(), super follows
        // the function in akka mode.
        int index = 0, inputsIndex = 0, outputIndex = 0;
        for(String input: DEFAULT_SPLITTER.split(currentValue.getWritable().toString())) {
            double doubleValue = NumberFormatUtils.getDouble(input.trim(), 0.0d);
            // no idea about why NaN in input data, we should process it as missing value TODO , according to norm type
            if(Double.isNaN(doubleValue)) {
                doubleValue = 0d;
            }
            if(index == super.columnConfigList.size()) {
                significance = NumberFormatUtils.getDouble(input, CommonConstants.DEFAULT_SIGNIFICANCE_VALUE);
                break;
            } else {
                ColumnConfig columnConfig = super.columnConfigList.get(index);

                if(columnConfig != null && columnConfig.isTarget()) {
                    if(modelConfig.isBinaryClassification()) {
                        ideal[outputIndex++] = doubleValue;
                    } else {
                        int ideaIndex = (int) doubleValue;
                        ideal[ideaIndex] = 1d;
                    }
                } else {
                    if(super.inputNodeCount == super.candidateCount) {
                        // no variable selected, good candidate but not meta and not target choosed
                        if(!columnConfig.isMeta() && !columnConfig.isTarget()
                                && CommonUtils.isGoodCandidate(columnConfig)) {
                            inputs[inputsIndex++] = doubleValue;
                            hashcode = hashcode * 31 + Double.valueOf(doubleValue).hashCode();
                        }
                    } else {
                        // final select some variables but meta and target are not included
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
            index += 1;
        }

        if(!isPrint) {
            LOG.info("data: input: {}, output: {}", Arrays.toString(inputs), Arrays.toString(ideal));
            isPrint = true;
        }

        // if fixInitialInput = true, we should use hashcode to sample.
        long longBaggingSampleRate = Double.valueOf(baggingSampleRate * 100).longValue();
        if(super.modelConfig.isFixInitialInput() && hashcode % 100 >= longBaggingSampleRate) {
            return;
        }

        super.sampleCount += 1;

        MLDataPair pair = new BasicMLDataPair(new BasicMLData(inputs), new BasicMLData(ideal));
        pair.setSignificance(significance);

        addDataPairToDataSet(hashcode, pair);
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.worker.AbstractWorkerComputable#initRecordReader(ml.shifu.guagua.io.GuaguaFileSplit)
     */
    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        super.setRecordReader(new GuaguaLineRecordReader(fileSplit));
    }

}
