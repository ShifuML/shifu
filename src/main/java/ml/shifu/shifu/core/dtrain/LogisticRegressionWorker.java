/*
 * Copyright [2013-2014] eBay Software Foundation
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

import java.io.File;
import java.io.IOException;
import java.io.Serializable;

import ml.shifu.guagua.hadoop.io.GuaguaLineRecordReader;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.util.MemoryDiskList;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.guagua.worker.AbstractWorkerComputable;
import ml.shifu.guagua.worker.WorkerContext;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Splitter;

/**
 * {@link LogisticRegressionWorker} defines logic to accumulate local <a
 * href=http://en.wikipedia.org/wiki/Logistic_regression >logistic regression</a> gradients.
 * 
 * <p>
 * At first iteration, wait for master to use the consistent initiating model.
 * 
 * <p>
 * At other iterations, workers include:
 * <ul>
 * <li>1. Update local model by using global model from last step..</li>
 * <li>2. Accumulate gradients by using local worker input data.</li>
 * <li>3. Send new local gradients to master by returning parameters.</li>
 * </ul>
 */
public class LogisticRegressionWorker
        extends
        AbstractWorkerComputable<LogisticRegressionParams, LogisticRegressionParams, GuaguaWritableAdapter<LongWritable>, GuaguaWritableAdapter<Text>> {

    private static final Logger LOG = LoggerFactory.getLogger(LogisticRegressionWorker.class);

    /**
     * Input column number
     */
    private int inputNum;

    /**
     * Output column number
     */
    private int outputNum;

    /**
     * In-memory data which located in memory at the first iteration.
     */
    private MemoryDiskList<Data> dataList;

    /**
     * Local logistic regression model.
     */
    private double[] weights;

    /**
     * A splitter to split data with specified delimiter.
     */
    private Splitter splitter = Splitter.on(",");

    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        this.setRecordReader(new GuaguaLineRecordReader(fileSplit));
    }

    @Override
    public void init(WorkerContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        this.inputNum = NumberFormatUtils.getInt(LogisticRegressionContants.LR_INPUT_NUM,
                LogisticRegressionContants.LR_INPUT_DEFAULT_NUM);
        this.outputNum = 1;
        double memoryFraction = Double.valueOf(context.getProps().getProperty("guagua.data.memoryFraction", "0.5"));
        String tmpFolder = context.getProps().getProperty("guagua.data.tmpfolder", "tmp");
        this.dataList = new MemoryDiskList<Data>((long) (Runtime.getRuntime().maxMemory() * memoryFraction), tmpFolder
                + File.separator + System.currentTimeMillis());
        // cannot find a good place to close these two data set, using Shutdown hook
        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            @Override
            public void run() {
                LogisticRegressionWorker.this.dataList.close();
            }
        }));
    }

    @Override
    public LogisticRegressionParams doCompute(WorkerContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        if(context.isFirstIteration()) {
            return new LogisticRegressionParams();
        } else {
            this.weights = context.getLastMasterResult().getParameters();
            double[] gradients = new double[this.inputNum + 1];
            double finalError = 0.0d;
            int size = 0;
            this.dataList.reOpen();
            for(Data data: dataList) {
                double error = sigmoid(data.inputs, this.weights) - data.outputs[0];
                finalError += error * error / 2;
                for(int i = 0; i < gradients.length; i++) {
                    gradients[i] += error * data.inputs[i];
                }
                size++;
            }
            LOG.info("Iteration {} with error {}", context.getCurrentIteration(), finalError / size);
            return new LogisticRegressionParams(gradients, finalError / size);
        }
    }

    /**
     * Compute sigmoid value by dot operation of two vectors.
     */
    private double sigmoid(double[] inputs, double[] weights) {
        double value = 0.0d;
        for(int i = 0; i < weights.length; i++) {
            value += weights[i] * inputs[i];
        }
        return 1.0d / (1.0d + Math.exp(-value));
    }

    @Override
    protected void postLoad(WorkerContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        this.dataList.switchState();
    }

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        String line = currentValue.getWritable().toString();
        double[] inputData = new double[inputNum + 1];
        double[] outputData = new double[outputNum];
        int count = 0, inputIndex = 0, outputIndex = 0;
        inputData[inputIndex++] = 1.0d;
        for(String unit: splitter.split(line)) {
            if(count < inputNum) {
                inputData[inputIndex++] = Double.valueOf(unit);
            } else if(count >= inputNum && count < (inputNum + outputNum)) {
                outputData[outputIndex++] = Double.valueOf(unit);
            } else {
                break;
            }
            count++;
        }
        this.dataList.append(new Data(inputData, outputData));
    }

    private static class Data implements Serializable {

        private static final long serialVersionUID = 903201066309036170L;

        public Data(double[] inputs, double[] outputs) {
            this.inputs = inputs;
            this.outputs = outputs;
        }

        private final double[] inputs;
        private final double[] outputs;
    }

}
