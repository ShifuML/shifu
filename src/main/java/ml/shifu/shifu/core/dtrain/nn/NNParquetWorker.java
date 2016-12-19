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

import ml.shifu.guagua.GuaguaRuntimeException;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.guagua.worker.WorkerContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatMLData;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatMLDataPair;
import ml.shifu.shifu.core.dtrain.dataset.FloatMLDataPair;
import ml.shifu.shifu.guagua.GuaguaParquetRecordReader;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.pig.LoadPushDown.RequiredFieldList;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.Tuple;
import org.apache.pig.impl.util.ObjectSerializer;

/**
 * {@link NNParquetWorker} is used to compute NN model according to splits assigned. The result will be sent to master
 * for accumulation.
 * 
 * <p>
 * Gradients in each worker will be sent to master to update weights of model in worker, which follows Encog's
 * multi-core implementation.
 * 
 * <p>
 * {@link NNParquetWorker} is to load data with parquet format. Only selected columns are loaded into value of Tuple.
 * Original index is already compacted so we should recover original index list and then use it to get real column
 * configuration object.
 */
public class NNParquetWorker extends AbstractNNWorker<Tuple> {

    private RequiredFieldList requiredFieldList = null;

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Tuple> currentValue,
            WorkerContext<NNParams, NNParams> workerContext) {
        // init field list for later read
        this.initFieldList();

        super.count += 1;
        if((super.count) % 2000 == 0) {
            LOG.info("Read {} records.", super.count);
        }

        double baggingSampleRate = super.modelConfig.getBaggingSampleRate();
        // if fixInitialInput = false, we only compare random value with baggingSampleRate to avoid parsing data.
        // if fixInitialInput = true, we should use hash code after parsing.
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

        Tuple tuple = currentValue.getWritable();

        // back from foreach to for loop because of in earlier version, tuple cannot be iterable.
        for(int i = 0; i < tuple.size(); i++) {
            Object element = null;
            try {
                element = tuple.get(i);
            } catch (ExecException e) {
                throw new GuaguaRuntimeException(e);
            }
            float floatValue = 0f;
            if(element != null) {
                if(element instanceof Float) {
                    floatValue = (Float) element;
                } else {
                    // check here to avoid bad performance in failed NumberFormatUtils.getFloat(input, 0f)
                    floatValue = element.toString().length() == 0 ? 0f : NumberFormatUtils.getFloat(element.toString(),
                            0f);
                }
            }
            // no idea about why NaN in input data, we should process it as missing value TODO , according to norm type
            floatValue = (Float.isNaN(floatValue) || Double.isNaN(floatValue)) ? 0f : floatValue;

            if(index == (super.inputNodeCount + super.outputNodeCount)) {
                assert element != null;
                if(element != null && element instanceof Float) {
                    significance = (Float) element;
                } else {
                    // check here to avoid bad performance in failed NumberFormatUtils.getFloat(input, 0f)
                    significance = element.toString().length() == 0 ? 1f : NumberFormatUtils.getFloat(
                            element.toString(), 1f);
                }
                // break here if we reach weight column which is last column
                break;
            } else {
                int columnIndex = requiredFieldList.getFields().get(index).getIndex();
                if(columnIndex >= super.columnConfigList.size()) {
                    assert element != null;
                    if(element != null && element instanceof Float) {
                        significance = (Float) element;
                    } else {
                        // check here to avoid bad performance in failed NumberFormatUtils.getFloat(input, 0f)
                        significance = element.toString().length() == 0 ? 1f : NumberFormatUtils.getFloat(
                                element.toString(), 1f);
                    }
                    break;
                } else {
                    ColumnConfig columnConfig = super.columnConfigList.get(columnIndex);
                    if(columnConfig != null && columnConfig.isTarget()) {
                        if(modelConfig.isRegression()) {
                            ideal[outputIndex++] = floatValue;
                        } else {
                            if(modelConfig.getTrain().isOneVsAll()) {
                                // if one vs all, set correlated idea value according to trainerId which means in
                                // trainer
                                // with id 0, target 0 is treated with 1, other are 0. Such target value are set to
                                // index of
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
                            if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                                    && CommonUtils.isGoodCandidate(columnConfig)) {
                                inputs[inputsIndex++] = floatValue;
                                hashcode = hashcode * 31 + Double.valueOf(floatValue).hashCode();
                            }
                        } else {
                            // only choose variable final select and not meta, not target
                            if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                                    && columnConfig.isFinalSelect()) {
                                inputs[inputsIndex++] = floatValue;
                                hashcode = hashcode * 31 + Double.valueOf(floatValue).hashCode();
                            }
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

        addDataPairToDataSet(hashcode, pair);
    }

    private void initFieldList() {
        if(requiredFieldList == null) {
            try {
                requiredFieldList = (RequiredFieldList) ObjectSerializer.deserialize(super.props
                        .getProperty("parquet.private.pig.required.fields"));
                LOG.debug("required list: {}", requiredFieldList);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.worker.AbstractWorkerComputable#initRecordReader(ml.shifu.guagua.io.GuaguaFileSplit)
     */
    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        Configuration conf = new Configuration();
        // new configuration don't have parquet.pig.schema, we need to add it manually.
        String pigSchema = super.props.getProperty("parquet.pig.schema");
        LOG.debug("pig schema: {}", pigSchema);
        conf.set("parquet.pig.schema", pigSchema);

        String requiredFieldList = super.props.getProperty("parquet.private.pig.required.fields");
        conf.set("parquet.private.pig.required.fields", requiredFieldList);
        LOG.debug("pig required fields: {}", requiredFieldList);

        String indexAccess = super.props.getProperty("parquet.private.pig.column.index.access");
        conf.set("parquet.private.pig.column.index.access", indexAccess);
        LOG.debug("parquet.private.pig.column.index.access: {}", indexAccess);

        super.setRecordReader(new GuaguaParquetRecordReader(conf, fileSplit));
    }

}
