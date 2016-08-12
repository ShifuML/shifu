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
package ml.shifu.shifu.core.dtrain.nn;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

import ml.shifu.guagua.ComputableMonitor;
import ml.shifu.guagua.hadoop.io.GuaguaLineRecordReader;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.guagua.worker.WorkerContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatMLData;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatMLDataPair;
import ml.shifu.shifu.core.dtrain.dataset.FloatMLDataPair;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

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
@ComputableMonitor(timeUnit = TimeUnit.SECONDS, duration = 240)
public class NNWorker extends AbstractNNWorker<Text> {

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<NNParams, NNParams> workerContext) {
        super.count += 1;
        if((super.count) % 5000 == 0) {
            LOG.info("Read {} records.", super.count);
        }

        double baggingSampleRate = super.modelConfig.getBaggingSampleRate();
        // if fixInitialInput = false, we only compare random value with baggingSampleRate to avoid parsing data.
        // if fixInitialInput = true, we should use hashcode after parsing.
        if(!super.modelConfig.isFixInitialInput() && Double.compare(Math.random(), baggingSampleRate) >= 0) {
            return;
        }

        float[] inputs = new float[super.inputNodeCount];
        float[] ideal = new float[super.outputNodeCount];

        if(super.isDry) {
            // dry train, use empty data.
            addDataPairToDataSet(0, new BasicFloatMLDataPair(new BasicFloatMLData(inputs), new BasicFloatMLData(ideal)));
            return;
        }

        long hashcode = 0;
        float significance = 1f;
        // use guava Splitter to iterate only once
        // use NNConstants.NN_DEFAULT_COLUMN_SEPARATOR to replace getModelConfig().getDataSetDelimiter(), super follows
        // the function in akka mode.
        int index = 0, inputsIndex = 0, outputIndex = 0;
        for(String input: DEFAULT_SPLITTER.split(currentValue.getWritable().toString())) {
            // check here to avoid bad performance in failed NumberFormatUtils.getFloat(input, 0f)
            float floatValue = input.length() == 0 ? 0f : NumberFormatUtils.getFloat(input, 0f);
            // no idea about why NaN in input data, we should process it as missing value TODO , according to norm type
            floatValue = (Float.isNaN(floatValue) || Double.isNaN(floatValue)) ? 0f : floatValue;

            if(index == super.columnConfigList.size()) {
                // check here to avoid bad performance in failed NumberFormatUtils.getFloat(input, 1f)
                significance = input.length() == 0 ? 1f : NumberFormatUtils.getFloat(input, 1f);
                // the last field is significance, break here
                break;
            } else {
                ColumnConfig columnConfig = super.columnConfigList.get(index);
                if(columnConfig != null && columnConfig.isTarget()) {
                    if(modelConfig.isRegression()) {
                        ideal[outputIndex++] = floatValue;
                    } else {
                        if(modelConfig.getTrain().isOneVsAll()) {
                            // if one vs all, set correlated idea value according to trainerId which means in trainer
                            // with id 0, target 0 is treated with 1, other are 0. Such target value are set to index of
                            // tags like [0, 1, 2, 3] compared with ["a", "b", "c", "d"]
                            ideal[outputIndex++] = Float.compare(floatValue, trainerId) == 0 ? 1f : 0f;
                        } else {
                            int ideaIndex = (int) floatValue;
                            ideal[ideaIndex] = 1f;
                        }
                    }
                } else {
                    if(super.inputNodeCount == super.candidateCount) {
                        // no variable selected, good candidate but not meta and not target choosed
                        if(!columnConfig.isMeta() && !columnConfig.isTarget()
                                && CommonUtils.isGoodCandidate(columnConfig)) {
                            inputs[inputsIndex++] = floatValue;
                            hashcode = hashcode * 31 + Double.valueOf(floatValue).hashCode();
                        }
                    } else {
                        // final select some variables but meta and target are not included
                        if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                                && columnConfig.isFinalSelect()) {
                            inputs[inputsIndex++] = floatValue;
                            // only fixInitialInput=true, hashcode is effective. Remove Arrays.hashcode to avoid one
                            // iteration for the input columns. Last weight column should be excluded.
                            hashcode = hashcode * 31 + Double.valueOf(floatValue).hashCode();
                        }
                    }
                }
            }
            index += 1;
        }

        // if fixInitialInput = true, we should use hashcode to sample.
        long longBaggingSampleRate = Double.valueOf(baggingSampleRate * 100).longValue();
        if(super.modelConfig.isFixInitialInput() && hashcode % 100 >= longBaggingSampleRate) {
            return;
        }

        super.sampleCount += 1;

        FloatMLDataPair pair = new BasicFloatMLDataPair(new BasicFloatMLData(inputs), new BasicFloatMLData(ideal));

        if(modelConfig.isRegression() && isUpSampleEnabled() && Double.compare(ideal[0], 1d) == 0) {
            // Double.compare(ideal[0], 1d) == 0 means positive tags; sample + 1 to avoid sample count to 0
            pair.setSignificance(significance * (super.upSampleRng.sample() + 1));
        } else {
            pair.setSignificance(significance);
        }
        boolean isTesting = false;
        if(workerContext.getAttachment()!=null&&workerContext.getAttachment() instanceof Boolean){
            isTesting = (Boolean)workerContext.getAttachment();
        }
        addDataPairToDataSet(hashcode, pair,isTesting);
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
