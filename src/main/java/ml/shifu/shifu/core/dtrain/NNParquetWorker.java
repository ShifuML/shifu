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

import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.guagua.worker.WorkerContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.guagua.GuaguaParquetRecordReader;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.pig.LoadPushDown.RequiredFieldList;
import org.apache.pig.data.Tuple;
import org.apache.pig.impl.util.ObjectSerializer;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;

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
        if((super.count) % 100000 == 0) {
            LOG.info("Read {} records.", super.count);
        }

        double baggingSampleRate = super.modelConfig.getBaggingSampleRate();
        // if fixInitialInput = false, we only compare random value with baggingSampleRate to avoid parsing data.
        // if fixInitialInput = true, we should use hash code after parsing.
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
        double significance = NNConstants.DEFAULT_SIGNIFICANCE_VALUE;
        // use guava Splitter to iterate only once
        // use NNConstants.NN_DEFAULT_COLUMN_SEPARATOR to replace getModelConfig().getDataSetDelimiter(), super follows
        // the function in akka mode.
        int index = 0, inputsIndex = 0, outputIndex = 0;

        for(Object element: currentValue.getWritable()) {
            double doubleValue = 0d;
            if(element != null) {
                if(element instanceof Double) {
                    doubleValue = (Double) element;
                } else {
                    doubleValue = NumberFormatUtils.getDouble(element.toString().trim(), 0d);
                }
            }
            // double doubleValue = NumberFormatUtils.getDouble(input.trim(), 0.0d);
            if(index == (super.inputNodeCount + super.outputNodeCount)) {
                if(element != null && element instanceof Double) {
                    significance = (Double) element;
                } else {
                    significance = NumberFormatUtils.getDouble(element.toString().trim(),
                            NNConstants.DEFAULT_SIGNIFICANCE_VALUE);;
                }
                // break here if we reach weight column which is last column
                break;
            } else {
                ColumnConfig columnConfig = super.columnConfigList.get(requiredFieldList.getFields().get(index)
                        .getIndex());

                if(columnConfig != null && columnConfig.isTarget()) {
                    ideal[outputIndex++] = doubleValue;
                } else {
                    inputs[inputsIndex++] = doubleValue;
                    hashcode = hashcode * 31 + Double.valueOf(doubleValue).hashCode();
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

        MLDataPair pair = new BasicMLDataPair(new BasicMLData(inputs), new BasicMLData(ideal));
        pair.setSignificance(significance);

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
        LOG.info("pig schema: {}", pigSchema);
        conf.set("parquet.pig.schema", pigSchema);

        String requiredFieldList = super.props.getProperty("parquet.private.pig.required.fields");
        conf.set("parquet.private.pig.required.fields", requiredFieldList);
        LOG.info("pig required fields: {}", requiredFieldList);

        String indexAccess = super.props.getProperty("parquet.private.pig.column.index.access");
        conf.set("parquet.private.pig.column.index.access", indexAccess);
        LOG.info("parquet.private.pig.column.index.access: {}", indexAccess);

        super.setRecordReader(new GuaguaParquetRecordReader(conf, fileSplit));
    }
}
