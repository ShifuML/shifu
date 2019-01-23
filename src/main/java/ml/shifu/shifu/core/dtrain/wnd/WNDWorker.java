/*
 * Copyright [2013-2019] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.wnd;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import ml.shifu.guagua.ComputableMonitor;
import ml.shifu.guagua.hadoop.io.GuaguaLineRecordReader;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.worker.AbstractWorkerComputable;
import ml.shifu.guagua.worker.WorkerContext;

/**
 * {@link WNDWorker} is responsible for loading part of data into memory, do iteration gradients computation and send
 * back to master for master aggregation. After master aggregation is done, received latest weights to do next
 * iteration.
 * 
 * <p>
 * {@link WNDWorker} needs to be recovered like load snapshot models and load data into memory again and can be used to
 * train with current iterations. All fault tolerance related state recovery should be taken care in this worker.
 * 
 * <p>
 * Data loading into memory as memory list includes two parts: numerical float array and sparse input object array which
 * is for categorical variables. To leverage sparse feature of categorical variables, sparse object is leveraged to
 * save memory and matrix computation.
 * 
 * <p>
 * TODO mini batch matrix support, matrix computation support
 * TODO forward, backward abstraction
 * TODO embedding arch logic
 * TODO variable/field based optimization to compute gradients
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
@ComputableMonitor(timeUnit = TimeUnit.SECONDS, duration = 3600)
public class WNDWorker extends
        AbstractWorkerComputable<WNDParams, WNDParams, GuaguaWritableAdapter<LongWritable>, GuaguaWritableAdapter<Text>> {

    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        // initialize Hadoop based line (long, string) reader
        super.setRecordReader(new GuaguaLineRecordReader(fileSplit));
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.worker.AbstractWorkerComputable#init(ml.shifu.guagua.worker.WorkerContext)
     */
    @Override
    public void init(WorkerContext<WNDParams, WNDParams> context) {
        // TODO initialize all worker related parameters, wide, deep, model arch ...
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.worker.AbstractWorkerComputable#doCompute(ml.shifu.guagua.worker.WorkerContext)
     */
    @Override
    public WNDParams doCompute(WorkerContext<WNDParams, WNDParams> context) {
        // TODO major worker computation logic like get mini-batch data from memory and forward, backward computation to
        // compute gradients, error computation ...
        return null;
    }

    /**
     * Logic to load data into memory list which includes float array for numerical features and sparse object array for
     * categorical features.
     */
    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<WNDParams, WNDParams> context) {
        // TODO
    }

}
