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

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import com.google.common.collect.Lists;

import ml.shifu.guagua.ComputableMonitor;
import ml.shifu.guagua.hadoop.io.GuaguaLineRecordReader;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.worker.WorkerContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatMLData;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatMLDataPair;
import ml.shifu.shifu.core.dtrain.dataset.FloatMLDataPair;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

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
@ComputableMonitor(timeUnit = TimeUnit.SECONDS, duration = 3600)
public class NNWorker extends AbstractNNWorker<Text> {

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<NNParams, NNParams> workerContext) {
        super.count += 1;
        if((super.count) % 5000 == 0) {
            LOG.info("Read {} records.", super.count);
        }

        float[] inputs = new float[super.featureInputsCnt];
        float[] ideal = new float[super.outputNodeCount];

        if(super.isDry) {
            // dry train, use empty data.
            addDataPairToDataSet(0,
                    new BasicFloatMLDataPair(new BasicFloatMLData(inputs), new BasicFloatMLData(ideal)));
            return;
        }

        long hashcode = 0;
        float significance = 1.0f;
        // use guava Splitter to iterate only once
        // use NNConstants.NN_DEFAULT_COLUMN_SEPARATOR to replace getModelConfig().getDataSetDelimiter(), super follows
        // the function in akka mode.
        int dataPos = 0, inputsIndex = 0, outputIndex = 0;

        String[] fields = Lists.newArrayList(this.splitter.split(currentValue.getWritable().toString()))
                .toArray(new String[0]);

        if (super.count == 1) {
            // When reading the first line, we check if it is the compact mode.
            configureCompactMode(fields.length);
        }

        for (ColumnConfig columnConfig : this.columnConfigList) {
            float fval;

            if (columnConfig.isTarget()) {
                fval = DTrainUtils.parseRawNormValue(fields, dataPos, 0.0f);
                if(isLinearTarget || modelConfig.isRegression()) {
                    ideal[outputIndex++] = fval;
                } else {
                    if(modelConfig.getTrain().isOneVsAll()) {
                        // if one vs all, set correlated idea value according to trainerId which means in trainer
                        // with id 0, target 0 is treated with 1, other are 0. Such target value are set to index of
                        // tags like [0, 1, 2, 3] compared with ["a", "b", "c", "d"]
                        ideal[outputIndex++] = Float.compare(fval, trainerId) == 0 ? 1f : 0f;
                    } else {
                        if(modelConfig.getTags().size() == 2) {
                            // if only 2 classes, output node is 1 node. if target = 0 means 0 is the index for
                            // positive prediction, set positive to 1 and negative to 0
                            int ideaIndex = (int) fval;
                            ideal[0] = ideaIndex == 0 ? 1f : 0f;
                        } else {
                            // for multiple classification
                            int ideaIndex = (int) fval;
                            ideal[ideaIndex] = 1f;
                        }
                    }
                }
                dataPos ++;
            } else if (this.weightColumnId > 0 // user set weight column
                    && this.weightColumnId == columnConfig.getColumnNum() // the weight column is current
                    && this.isWeightColumnMeta) { // the weight column is Meta
                significance = DTrainUtils.parseRawNormValue(fields, dataPos, 1.0f);
                // if invalid weight, set it to 1f and warning in log
                if(Float.compare(significance, 0f) < 0) {
                    LOG.warn("The {} record in current worker weight {} is less than 0f, it is invalid, set it to 1.",
                            count, significance);
                    significance = 1f;
                }
                dataPos ++;
            } else { // other variables
                if(subFeatureSet.contains(columnConfig.getColumnNum())) {
                    fval = DTrainUtils.parseRawNormValue(fields, dataPos, 0.0f);
                    if(columnConfig.isMeta() || columnConfig.isForceRemove()) {
                        // it shouldn't happen here
                        dataPos += 1;
                    } else if(columnConfig != null && columnConfig.isNumerical()
                            && modelConfig.getNormalizeType().equals(ModelNormalizeConf.NormType.ONEHOT)) {
                        for(int k = 0; k < columnConfig.getBinBoundary().size() + 1; k++) {
                            float tval = DTrainUtils.parseRawNormValue(fields, dataPos, 0.0f);
                            inputs[inputsIndex++] = tval;
                            dataPos++;
                        }
                    } else if(columnConfig != null && columnConfig.isCategorical()
                            && (modelConfig.getNormalizeType().equals(ModelNormalizeConf.NormType.ZSCALE_ONEHOT)
                            || modelConfig.getNormalizeType().equals(ModelNormalizeConf.NormType.ONEHOT))) {
                        for(int k = 0; k < columnConfig.getBinCategory().size() + 1; k++) {
                            float tval = DTrainUtils.parseRawNormValue(fields, dataPos, 0.0f);
                            inputs[inputsIndex++] = tval;
                            dataPos++;
                        }
                    } else {
                        inputs[inputsIndex++] = fval;
                        dataPos++;
                    }
                    hashcode = hashcode * 31 + Double.valueOf(fval).hashCode();
                } else if (!isCompactMode){ // It is not compact mode, just skip unused data in normalized data. No unused data will exist in compact mode.
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

        // if (dataPos == fields.length -1), the last column is weight column
        // if (dataPos == fields.length), normalized data doesn't have weight column
        if (dataPos != fields.length -1 && dataPos != fields.length) {
            LOG.error("Normalization data has extra data. Expect {} or {}, actual is {}.", dataPos, dataPos + 1, fields.length);
            throw new RuntimeException("Out of range Normalization data doesn't match with ColumnConfig.json.");
        }

        if (this.weightColumnId > 0 && !this.isWeightColumnMeta && dataPos == fields.length - 1) {
            // user specified the weight column, it is not meta column and now point to last column of data
            significance = DTrainUtils.parseRawNormValue(fields, dataPos, 1.0f);
            // if invalid weight, set it to 1f and warning in log
            if(Float.compare(significance, 0f) < 0) {
                LOG.warn("The {} record in current worker weight {} is less than 0f, it is invalid, set it to 1.",
                        count, significance);
                significance = 1f;
            }
        } else if(this.weightColumnId > 0 && !this.isWeightColumnMeta && dataPos == fields.length) {
            // user specified the weight column, and it is not meta column
            // but it doesn't exist in normalized data set, throw error or use default?
            // OK, use default currently
            significance = 1f;
        }

        // output delimiter in norm can be set by user now and if user set a special one later changed, this exception
        // is helped to quick find such issue.
        if(inputsIndex != inputs.length) {
            String delimiter = workerContext.getProps().getProperty(Constants.SHIFU_OUTPUT_DATA_DELIMITER,
                    Constants.DEFAULT_DELIMITER);
            throw new RuntimeException("Input length is inconsistent with parsing size. Input original size: "
                    + inputs.length + ", parsing size:" + inputsIndex + ", delimiter:" + delimiter + ".");
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
                if((modelConfig.isRegression()
                        || (modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll())) // regression or
                                                                                                    // onevsall
                        && (int) (ideal[0] + 0.01d) == 0 // negative record
                        && isInRange(hashcode, startHashCode, endHashCode)) {
                    return;
                }
            } else {
                // if not fixed initial input, and for regression or onevsall multiple classification (regression also).
                // if negative record
                if((modelConfig.isRegression()
                        || (modelConfig.isClassification() && modelConfig.getTrain().isOneVsAll())) // regression or
                                                                                                    // onevsall
                        && (int) (ideal[0] + 0.01d) == 0 // negative record
                        && Double.compare(super.sampelNegOnlyRandom.nextDouble(),
                                this.modelConfig.getBaggingSampleRate()) >= 0) {
                    return;
                }
            }
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

        // do bagging sampling only for training data
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
