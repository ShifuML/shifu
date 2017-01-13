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

        // if only sample negative, no matter bagging or replacement, do sampling here.
        if(modelConfig.getTrain().getSampleNegOnly() // sample negative enabled
                && (modelConfig.isRegression() || (modelConfig.isClassification() && modelConfig.getTrain()
                        .isOneVsAll())) // regression or onevsall
                && Double.compare(ideal[0] + 0.01d, 0d) == 0 // negative record
                && (!this.modelConfig.isFixInitialInput() && Double.compare(Math.random(),
                        this.modelConfig.getBaggingSampleRate()) >= 0)) {
            return;
        }
        if(modelConfig.getTrain().getSampleNegOnly()// sample negative enabled
                && (modelConfig.isRegression() || (modelConfig.isClassification() && modelConfig.getTrain()
                        .isOneVsAll()))// regression or onevsall
                && (Double.compare(ideal[0] + 0.01d, 0d) == 0 // negative record
                        && this.modelConfig.isFixInitialInput() && hashcode % 100 >= Double.valueOf(
                        this.modelConfig.getBaggingSampleRate() * 100).longValue())) {
            return;
        }

        FloatMLDataPair pair = new BasicFloatMLDataPair(new BasicFloatMLData(inputs), new BasicFloatMLData(ideal));

        // up sampling logic, just add more weights while bagging sampling rate is still not changed
        if(modelConfig.isRegression() && isUpSampleEnabled() && Double.compare(ideal[0], 1d) == 0) {
            // Double.compare(ideal[0], 1d) == 0 means positive tags; sample + 1 to avoid sample count to 0
            pair.setSignificance(significance * (super.upSampleRng.sample() + 1));
        } else {
            pair.setSignificance(significance);
        }

        boolean isValidation = false;
        if(workerContext.getAttachment() != null && workerContext.getAttachment() instanceof Boolean) {
            isValidation = (Boolean) workerContext.getAttachment();
        }

        boolean isInTraining = addDataPairToDataSet(hashcode, pair, isValidation);

        // do bagging sampling only for training data，
        if(isInTraining) {
            float subsampleWeights = sampleWeights(pair.getIdealArray()[0]);
            if(isPositive(pair.getIdealArray()[0])) {
                this.positiveSelectedTrainCount += subsampleWeights * 1L;
            } else {
                this.negativeSelectedTrainCount += subsampleWeights * 1L;
            }
            // set weights to significance, if 0, significance will be 0, that is bagging sampling
            pair.setSignificance(pair.getSignificance() * subsampleWeights);
        } else {
            // for validation data, according bagging sampling logic, we may need to sampling validation data set, while
            // validation data set are only used to compute validation error, not to do real sampling is ok.
        }
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
