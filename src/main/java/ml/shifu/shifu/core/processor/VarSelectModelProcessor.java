/**
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.shifu.core.processor;

import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.List;

import ml.shifu.guagua.GuaguaConstants;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceConstants;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelBasicConf.RunMode;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.AbstractTrainer;
import ml.shifu.shifu.core.VariableSelector;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.core.dtrain.NNConstants;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.core.varselect.Constants;
import ml.shifu.shifu.core.varselect.VarSelectMapper;
import ml.shifu.shifu.core.varselect.VarSelectReducer;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Environment;

import org.apache.commons.collections.ListUtils;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.map.MultithreadedMapper;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.pig.impl.util.JarManager;
import org.apache.zookeeper.ZooKeeper;
import org.encog.ml.data.MLDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Splitter;

/**
 * Variable selection processor, select the variable based on KS/IV value, or </p>
 * <p/>
 * Selection variable based on the wrapper training processor.
 * </p>
 */
public class VarSelectModelProcessor extends BasicModelProcessor implements Processor {

    private final static Logger log = LoggerFactory.getLogger(VarSelectModelProcessor.class);

    /**
     * run for the variable selection
     */
    @Override
    public int run() throws Exception {
        setUp(ModelStep.VARSELECT);

        CommonUtils.updateColumnConfigFlags(modelConfig, columnConfigList);

        VariableSelector selector = new VariableSelector(this.modelConfig, this.columnConfigList);

        if(!modelConfig.getVarSelectWrapperEnabled()) {
            this.columnConfigList = selector.selectByFilter();
            try {
                this.saveColumnConfigList();
            } catch (ShifuException e) {
                throw new ShifuException(ShifuErrorCode.ERROR_WRITE_COLCONFIG, e);
            }
        } else {
            // SE means sensitivity
            if(super.getModelConfig().getDataSet().getSource() == SourceType.HDFS
                    && super.getModelConfig().getBasic().getRunMode() == RunMode.mapred) {
                validateDistributedWrapperVarSelect();
                syncDataToHdfs(super.modelConfig.getDataSet().getSource());
                distributedWrapper(selector);
            } else {
                wrapper(selector);
            }
        }
        log.info("Step Finished: varselect");

        clearUp(ModelStep.VARSELECT);
        return 0;
    }

    private void validateDistributedWrapperVarSelect() {
        if(!("R".equalsIgnoreCase(this.modelConfig.getVarSelectWrapperBy()) || "SE".equalsIgnoreCase(this.modelConfig
                .getVarSelectWrapperBy()))) {
            throw new IllegalArgumentException(
                    "Only R(Remove) and SE(Sensitivity Selection) wrapperBy methods are supported so far in distributed variable selection.");
        }

        if(!NNConstants.NN_ALG_NAME.equalsIgnoreCase(super.getModelConfig().getTrain().getAlgorithm())) {
            throw new IllegalArgumentException(
                    "Currently we only support NN distributed training to do wrapper by analyzing variable selection.");
        }

        if(super.getModelConfig().getDataSet().getSource() != SourceType.HDFS) {
            throw new IllegalArgumentException(
                    "Currently we only support distributed wrapper by analyzing on HDFS source type.");
        }

        if(super.getModelConfig().getBasic().getRunMode() != RunMode.mapred) {
            throw new IllegalArgumentException(
                    "Currently we only support distributed wrapper by analyzing on HDFS source type.");
        }
    }

    private String addRuntimeJars() {
        List<String> jars = new ArrayList<String>(16);
        // jackson-databind-*.jar
        jars.add(JarManager.findContainingJar(ObjectMapper.class));
        // jackson-core-*.jar
        jars.add(JarManager.findContainingJar(JsonParser.class));
        // jackson-annotations-*.jar
        jars.add(JarManager.findContainingJar(JsonIgnore.class));
        // commons-compress-*.jar
        jars.add(JarManager.findContainingJar(BZip2CompressorInputStream.class));
        // commons-lang-*.jar
        jars.add(JarManager.findContainingJar(StringUtils.class));
        // commons-collections-*.jar
        jars.add(JarManager.findContainingJar(ListUtils.class));
        // common-io-*.jar
        jars.add(JarManager.findContainingJar(org.apache.commons.io.IOUtils.class));
        // guava-*.jar
        jars.add(JarManager.findContainingJar(Splitter.class));
        // encog-core-*.jar
        jars.add(JarManager.findContainingJar(MLDataSet.class));
        // shifu-*.jar
        jars.add(JarManager.findContainingJar(getClass()));
        // guagua-core-*.jar
        jars.add(JarManager.findContainingJar(GuaguaConstants.class));
        // guagua-mapreduce-*.jar
        jars.add(JarManager.findContainingJar(GuaguaMapReduceConstants.class));
        // zookeeper-*.jar
        jars.add(JarManager.findContainingJar(ZooKeeper.class));

        return StringUtils.join(jars, NNConstants.LIB_JAR_SEPARATOR);
    }

    /**
     * Wrapper through {@link TrainModelProcessor} and a MapReduce job to analyze biggest sensitivity MSE.
     */
    private void distributedWrapper(VariableSelector selector) throws Exception {
        // 1. Train a model using current selected variables, if no variables selected, use all candidate variables.
        TrainModelProcessor trainModelProcessor = new TrainModelProcessor();
        trainModelProcessor.setForVarSelect(true);
        trainModelProcessor.run();

        // 2. Submit a MapReduce job to analyze sensitivity MSE.
        SourceType source = this.modelConfig.getDataSet().getSource();
        Configuration conf = new Configuration();
        // add jars to hadoop mapper and reducer
        new GenericOptionsParser(conf, new String[] { "-libjars", addRuntimeJars() });

        conf.setBoolean("mapred.map.tasks.speculative.execution", true);
        conf.setBoolean("mapred.reduce.tasks.speculative.execution", true);
        conf.set(
                Constants.SHIFU_MODEL_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(source)
                        .makeQualified(new Path(super.getPathFinder().getModelConfigPath(source))).toString());
        conf.set(
                Constants.SHIFU_COLUMN_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(source)
                        .makeQualified(new Path(super.getPathFinder().getColumnConfigPath(source))).toString());
        conf.set("mapred.job.queue.name", Environment.getProperty(Environment.HADOOP_JOB_QUEUE, "default"));
        conf.set(Constants.SHIFU_MODELSET_SOURCE_TYPE, source.toString());

        Float wrapperRatio = this.modelConfig.getVarSelect().getWrapperRatio();
        if(wrapperRatio == null) {
            log.warn("wrapperRatio in var select is not set. Using default value 0.05.");
            wrapperRatio = 0.05f;
        }

        if(wrapperRatio.compareTo(Float.valueOf(1.0f)) >= 0) {
            throw new IllegalArgumentException("WrapperRatio should be in (0, 1).");
        }
        conf.setFloat(Constants.SHIFU_VARSELECT_WRAPPER_RATIO, wrapperRatio);

        Job job = new Job(conf, "Shifu: Variable Selection Wrapper Job : " + this.modelConfig.getModelSetName());
        job.setJarByClass(getClass());
        boolean isSEVarSelMulti = "true".equalsIgnoreCase(Environment.getProperty("shifu.varsel.se.multi", "false"));
        if(isSEVarSelMulti) {
            job.setMapperClass(MultithreadedMapper.class);
            MultithreadedMapper.setMapperClass(job, VarSelectMapper.class);
            MultithreadedMapper.setNumberOfThreads(job, 6);
        } else {
            job.setMapperClass(VarSelectMapper.class);
        }

        job.setMapOutputKeyClass(LongWritable.class);
        job.setMapOutputValueClass(DoubleWritable.class);
        job.setInputFormatClass(TextInputFormat.class);
        FileInputFormat.setInputPaths(
                job,
                ShifuFileUtils.getFileSystemBySourceType(source).makeQualified(
                        new Path(super.getPathFinder().getNormalizedDataPath())));

        job.setReducerClass(VarSelectReducer.class);
        // Only one reducer, no need set combiner because of distinct keys in map outputs.
        job.setNumReduceTasks(1);
        job.setOutputKeyClass(LongWritable.class);
        job.setOutputValueClass(NullWritable.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        String varSelectMSEOutputPath = super.getPathFinder().getVarSelectMSEOutputPath(source);
        FileOutputFormat.setOutputPath(job, new Path(varSelectMSEOutputPath));

        // clean output firstly
        ShifuFileUtils.deleteFile(varSelectMSEOutputPath, source);

        // submit job
        if(job.waitForCompletion(true)) {
            if(!ShifuFileUtils.isFileExists(varSelectMSEOutputPath + Path.SEPARATOR + "part-r-00000", source)) {
                throw new RuntimeException("Var select MSE stats output file not exist.");
            }

            for(ColumnConfig config: super.columnConfigList) {
                if(config.isFinalSelect()) {
                    config.setFinalSelect(false);
                }
            }

            BufferedReader reader = null;
            try {
                reader = ShifuFileUtils.getReader(varSelectMSEOutputPath + Path.SEPARATOR + "part-r-00000", source);
                String str = null;
                int count = 0;
                while((str = reader.readLine()) != null) {
                    ++count;
                    ColumnConfig columnConfig = this.columnConfigList.get(Integer.parseInt(str));
                    columnConfig.setFinalSelect(true);
                    log.info("Variable {} is selected.", columnConfig.getColumnName());
                }
                log.info("{} variables are selected.", count);
            } finally {
                IOUtils.closeQuietly(reader);
            }

            this.saveColumnConfigList();
            this.syncDataToHdfs(this.modelConfig.getDataSet().getSource());
        }
    }

    /**
     * user wrapper to select variable
     * 
     * @param selector
     * @throws Exception
     */
    private void wrapper(VariableSelector selector) throws Exception {

        NormalizeModelProcessor n = new NormalizeModelProcessor();

        // runNormalize();
        n.run();

        TrainModelProcessor t = new TrainModelProcessor(false, false);
        t.run();

        AbstractTrainer trainer = t.getTrainer(0);

        if(trainer instanceof NNTrainer) {
            selector.selectByWrapper((NNTrainer) trainer);
            try {
                this.saveColumnConfigList();
            } catch (ShifuException e) {
                throw new ShifuException(ShifuErrorCode.ERROR_WRITE_COLCONFIG, e);
            }
        }
    }

}
