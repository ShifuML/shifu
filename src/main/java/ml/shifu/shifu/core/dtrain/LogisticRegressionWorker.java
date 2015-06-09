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
import java.util.List;
import java.util.Properties;

import ml.shifu.guagua.hadoop.io.GuaguaLineRecordReader;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.util.MemoryDiskList;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.guagua.worker.AbstractWorkerComputable;
import ml.shifu.guagua.worker.WorkerContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.util.CommonUtils;

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
     * Candidate column number
     */
    private int candidateNum;

    /**
     * In-memory data which located in memory at the first iteration.
     */
    private MemoryDiskList<Data> dataList;

    /**
     * Local logistic regression model.
     */
    private double[] weights;
    
    /**
     * Model Config read from HDFS
     */
    private ModelConfig modelConfig;

    /**
     * Column Config list read from HDFS
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * A splitter to split data with specified delimiter.
     */
    private Splitter splitter = Splitter.on("|");

    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {
        this.setRecordReader(new GuaguaLineRecordReader(fileSplit));
    }

    @Override
    public void init(WorkerContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        loadConfigFiles(context.getProps());
        //this.inputNum = NumberFormatUtils.getInt(LogisticRegressionContants.LR_INPUT_NUM,
          //      LogisticRegressionContants.LR_INPUT_DEFAULT_NUM);
        int[] inputOutputIndex = NNUtils.getInputOutputCandidateCounts(this.columnConfigList);
        this.inputNum = inputOutputIndex[0] == 0 ? inputOutputIndex[2] : inputOutputIndex[0];
        this.outputNum = 1;
        this.candidateNum = inputOutputIndex[2];
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
            double[] gradients = new double[this.inputNum];
            double finalError = 0.0d;
            int size = 0;
            this.dataList.reOpen();
            for(Data data: dataList) {
            	double result = sigmoid(data.inputs, this.weights);
            	LOG.info("iteration_result:"+result);
                double error = result - data.outputs[0];
               // finalError += error * error / 2;
                finalError += cost(result,data.outputs[0]);
            	LOG.info("iteration_final:"+finalError);
                for(int i = 0; i < gradients.length; i++) {
                    gradients[i] += error * data.inputs[i] * data.significance;
                }
                size++;
            }
        	LOG.info("finish_final:"+finalError);
        	LOG.info("finish_size:"+size);
            LOG.info("Iteration {} with error {}", context.getCurrentIteration(), finalError / size);
            return new LogisticRegressionParams(gradients, finalError/size);
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
    
    public double cost(double result, double output){
    	if(output==1.0d){
    		return -Math.log(result);
    	}
    	else{
    		return -Math.log(1-result);
    	}
    }

    @Override
    protected void postLoad(WorkerContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        this.dataList.switchState();
    }

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue,
            WorkerContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        String line = currentValue.getWritable().toString();
        double[] inputData = new double[inputNum];
        double[] outputData = new double[outputNum];
        int count = 0, inputIndex = 0, outputIndex = 0;
        long hashcode = 0;
        double significance = NNConstants.DEFAULT_SIGNIFICANCE_VALUE;
        //inputData[inputIndex++] = 1.0d;
        LOG.info("inputnum:"+inputNum);
        LOG.info("outputnum:"+outputNum);
        LOG.info("current_line"+line);
        LOG.info("columnconfig_size:"+this.columnConfigList.size());
        for(String unit: splitter.split(line)) {
            double doubleValue = NumberFormatUtils.getDouble(unit.trim(), 0.0d);
            if(count == this.columnConfigList.size()) {
                significance = NumberFormatUtils.getDouble(unit.trim(), NNConstants.DEFAULT_SIGNIFICANCE_VALUE);
                break;
            }
            else{
                ColumnConfig columnConfig = this.columnConfigList.get(count);
            if(columnConfig != null && columnConfig.isTarget()) {
                outputData[outputIndex++] = doubleValue;
            } else {
                if(this.inputNum == this.candidateNum) {
                    // all variables are not set final-selectByFilter
                    if(CommonUtils.isGoodCandidate(columnConfig)) {
                        inputData[inputIndex++] = doubleValue;
                        hashcode = hashcode * 31 + Double.valueOf(doubleValue).hashCode();
                    }
                } else {
                    // final select some variables
                    if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                            && columnConfig.isFinalSelect()) {
                        inputData[inputIndex++] = doubleValue;
                        // only fixInitialInput=true, hashcode is effective. Remove Arrays.hashcode to avoid one
                        // iteration for the input columns. Last weight column should be excluded.
                        hashcode = hashcode * 31 + Double.valueOf(doubleValue).hashCode();
                    }
                }
            }
            }
            count++;
        }
        this.dataList.append(new Data(inputData, outputData,significance));
    }
    
    private void loadConfigFiles(final Properties props) {
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
    }

    private static class Data implements Serializable {

        private static final long serialVersionUID = 903201066309036170L;

        public Data(double[] inputs, double[] outputs, double significance) {
            this.inputs = inputs;
            this.outputs = outputs;
            this.significance = significance;
        }
        private final double significance;
        private final double[] inputs;
        private final double[] outputs;
    }
    
    public void test(String input)
    {
    	int count = 0;
    	for(String unit: splitter.split(input)){
    		System.out.println("count:"+count);
    		System.out.println("current_value:"+unit);
    		count++;

    	}
    }
    
    public static void main(String[] args){
    	String line ="|1|1.126082|-1.979292|1.307047|1.018816|1.555019|3.408517|2.816882|2.619519|2.274134|2.31";
    	LogisticRegressionWorker w =  new LogisticRegressionWorker();
    	//w.test(line);
    	w.cost(1, 1);
    	
    	
    }

}
