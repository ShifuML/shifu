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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import ml.shifu.guagua.GuaguaConstants;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceClient;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceConstants;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelBasicConf.RunMode;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.AbstractTrainer;
import ml.shifu.shifu.core.VariableSelector;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.core.dtrain.NNConstants;
import ml.shifu.shifu.core.dvarsel.VarSelMaster;
import ml.shifu.shifu.core.dvarsel.VarSelMasterResult;
import ml.shifu.shifu.core.dvarsel.VarSelOutput;
import ml.shifu.shifu.core.dvarsel.VarSelWorker;
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;
import ml.shifu.shifu.core.dvarsel.wrapper.WrapperMasterConductor;
import ml.shifu.shifu.core.dvarsel.wrapper.WrapperWorkerConductor;
import ml.shifu.shifu.core.mr.input.CombineInputFormat;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.core.varselect.ColumnInfo;
import ml.shifu.shifu.core.varselect.VarSelectMapper;
import ml.shifu.shifu.core.varselect.VarSelectReducer;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;

import org.apache.commons.collections.ListUtils;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.jexl2.JexlException;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.map.MultithreadedMapper;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.pig.impl.util.JarManager;
import org.apache.zookeeper.ZooKeeper;
import org.encog.ml.data.MLDataSet;
import org.mortbay.log.Log;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Splitter;

/**
 * Variable selection processor, select the variable based on KS/IV value, or </p>
 * 
 * <p>
 * Selection variable based on the wrapper training processor.
 * 
 * <p>
 * For sensitive variable selection, each time wrapperRatio percent of variables will be removed. If continue do
 * variable selection, continue to run varselect command. Current design will do variable selection continuously.
 */
public class VarSelectModelProcessor extends BasicModelProcessor implements Processor {

    private final static Logger log = LoggerFactory.getLogger(VarSelectModelProcessor.class);

    /**
     * Run for the variable selection
     */
    @Override
    public int run() throws Exception {
        log.info("Step Start: varselect");
        long start = System.currentTimeMillis();

        setUp(ModelStep.VARSELECT);

        validateNormalize();

        syncDataToHdfs(super.modelConfig.getDataSet().getSource());

        VariableSelector selector = new VariableSelector(this.modelConfig, this.columnConfigList);

        if(!modelConfig.getVarSelectWrapperEnabled()) {
            // Select by local KS, IV
            CommonUtils.updateColumnConfigFlags(modelConfig, columnConfigList);

            this.columnConfigList = selector.selectByFilter();
            try {
                this.saveColumnConfigList();
            } catch (ShifuException e) {
                throw new ShifuException(ShifuErrorCode.ERROR_WRITE_COLCONFIG, e);
            }
        } else {
            // wrapper method
            if(super.getModelConfig().getDataSet().getSource() == SourceType.HDFS
                    && super.getModelConfig().getBasic().getRunMode() == RunMode.mapred) {
                if(Constants.WRAPPER_BY_SE.equalsIgnoreCase(modelConfig.getVarSelect().getWrapperBy())
                        || Constants.WRAPPER_BY_REMOVE.equalsIgnoreCase(modelConfig.getVarSelect().getWrapperBy())) {
                    // SE method supports remove and sensitivity se so far
                    validateDistributedWrapperVarSelect();
                    syncDataToHdfs(super.modelConfig.getDataSet().getSource());
                    distributedSEWrapper();
                } else if(Constants.WRAPPER_BY_VOTED.equalsIgnoreCase(modelConfig.getVarSelect().getWrapperBy())) {
                    votedVariablesSelection();
                }
            } else {
                // local wrapper mode: old
                wrapper(selector);
            }
        }

        clearUp(ModelStep.VARSELECT);
        log.info("Step Finished: varselect with {} ms", (System.currentTimeMillis() - start));
        return 0;
    }

    private void validateNormalize() throws IOException {
        if(!ShifuFileUtils.isFileExists(
                new PathFinder(modelConfig).getNormalizedDataPath(this.modelConfig.getDataSet().getSource()),
                this.modelConfig.getDataSet().getSource())) {
            throw new IllegalStateException("Cannot find normalized data, please do 'Shifu normalize' firstly.");
        }
    }

    private void validateDistributedWrapperVarSelect() {
        if(!(Constants.WRAPPER_BY_REMOVE.equalsIgnoreCase(this.modelConfig.getVarSelectWrapperBy()) || Constants.WRAPPER_BY_SE
                .equalsIgnoreCase(this.modelConfig.getVarSelectWrapperBy()))) {
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

    private void votedVariablesSelection() throws ClassNotFoundException, IOException, InterruptedException {
        log.info("Start voted variables selection ");
        // sync data back to hdfs
        super.syncDataToHdfs(modelConfig.getDataSet().getSource());

        SourceType sourceType = super.getModelConfig().getDataSet().getSource();

        final List<String> args = new ArrayList<String>();
        // prepare parameter
        prepareVarSelParams(args, sourceType);

        Path columnIdsPath = getVotedSelectionPath(sourceType);
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT,
                ml.shifu.shifu.util.Constants.VAR_SEL_COLUMN_IDS_OUPUT, columnIdsPath.toString()));

        long start = System.currentTimeMillis();

        GuaguaMapReduceClient guaguaClient = new GuaguaMapReduceClient();

        guaguaClient.createJob(args.toArray(new String[0])).waitForCompletion(true);

        log.info("Voted variables selection finished in {}ms.", System.currentTimeMillis() - start);

        persistColumnIds(columnIdsPath);
        super.syncDataToHdfs(sourceType);
    }

    private int persistColumnIds(Path path) {
        try {
            List<Scanner> scanners = ShifuFileUtils.getDataScanners(path.toString(), modelConfig.getDataSet()
                    .getSource());

            List<Integer> ids = null;
            for(Scanner scanner: scanners) {
                while(scanner.hasNextLine()) {
                    String[] raw = scanner.nextLine().trim().split("\\|");

                    @SuppressWarnings("unused")
                    int idSize = Integer.parseInt(raw[0]);

                    ids = CommonUtils.stringToIntegerList(raw[1]);

                }
            }

            // prevent multiply running setting
            for(ColumnConfig config: columnConfigList) {
                if(!config.isForceSelect()) {
                    config.setFinalSelect(Boolean.FALSE);
                }
            }

            for(Integer id: ids) {
                this.columnConfigList.get(id).setFinalSelect(Boolean.TRUE);
            }

            super.saveColumnConfigList();

        } catch (IOException e) {
            e.printStackTrace();
            return -1;
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
            return -1;
        }

        return 0;
    }

    private Path getVotedSelectionPath(SourceType sourceType) {
        return ShifuFileUtils.getFileSystemBySourceType(sourceType).makeQualified(
                new Path(getPathFinder().getVarSelsPath(sourceType), "VarSels"));
    }

    private void prepareVarSelParams(final List<String> args, final SourceType sourceType) {
        args.add("-libjars");

        args.add(addRuntimeJars());

        args.add("-i");
        args.add(ShifuFileUtils.getFileSystemBySourceType(sourceType)
                .makeQualified(new Path(modelConfig.getDataSetRawPath())).toString());

        String zkServers = Environment.getProperty(Environment.ZOO_KEEPER_SERVERS);
        if(StringUtils.isEmpty(zkServers)) {
            log.warn("No specified zookeeper settings from zookeeperServers in shifuConfig file, Guagua will set embeded zookeeper server in client process. For big data applications, specified zookeeper servers are strongly recommended.");
        } else {
            args.add("-z");
            args.add(zkServers);
        }

        // setting the class
        args.add("-w");
        args.add(VarSelWorker.class.getName());

        args.add("-m");
        args.add(VarSelMaster.class.getName());

        args.add("-c");
        // the reason to add 1 is that the first iteration in D-NN implementation is used for training preparation.
        // FIXME, how to set iteration number
        int expectVarCount = this.modelConfig.getVarSelectFilterNum();
        int forceSelectCount = 0;
        int candidateCount = 0;
        for(ColumnConfig columnConfig: columnConfigList) {
            if(columnConfig.isForceSelect()) {
                forceSelectCount++;
            }
            if(CommonUtils.isGoodCandidate(columnConfig)) {
                candidateCount++;
            }
        }

        args.add(String.valueOf(Math.min(expectVarCount, candidateCount) - forceSelectCount + 1));

        args.add("-mr");
        args.add(VarSelMasterResult.class.getName());

        args.add("-wr");
        args.add(VarSelWorkerResult.class.getName());

        // setting conductor
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT,
                ml.shifu.shifu.util.Constants.VAR_SEL_MASTER_CONDUCTOR,
                Environment.getProperty(Environment.VAR_SEL_MASTER_CONDUCTOR, WrapperMasterConductor.class.getName())));

        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT,
                ml.shifu.shifu.util.Constants.VAR_SEL_WORKER_CONDUCTOR,
                Environment.getProperty(Environment.VAR_SEL_MASTER_CONDUCTOR, WrapperWorkerConductor.class.getName())));

        // setting queue
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.MAPRED_JOB_QUEUE_NAME,
                Environment.getProperty(Environment.HADOOP_JOB_QUEUE, ml.shifu.shifu.util.Constants.DEFAULT_JOB_QUEUE)));

        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_MASTER_INTERCEPTERS,
                VarSelOutput.class.getName()));

        // setting model config column config
        args.add(String.format(
                NNConstants.MAPREDUCE_PARAM_FORMAT,
                NNConstants.SHIFU_NN_MODEL_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(sourceType).makeQualified(
                        new Path(super.getPathFinder().getModelConfigPath(sourceType)))));
        args.add(String.format(
                NNConstants.MAPREDUCE_PARAM_FORMAT,
                NNConstants.SHIFU_NN_COLUMN_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(sourceType).makeQualified(
                        new Path(super.getPathFinder().getColumnConfigPath(sourceType)))));

        // source type
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.NN_MODELSET_SOURCE_TYPE, sourceType));

        // computation time
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_COMPUTATION_TIME_THRESHOLD,
                60 * 60 * 1000l));
        setHeapSizeAndSplitSize(args);

        // one can set guagua conf in shifuconfig
        for(Map.Entry<Object, Object> entry: Environment.getProperties().entrySet()) {
            if(entry.getKey().toString().startsWith("nn") || entry.getKey().toString().startsWith("guagua")
                    || entry.getKey().toString().startsWith("mapred")) {
                args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, entry.getKey().toString(), entry.getValue()
                        .toString()));
            }
        }
    }

    // GuaguaOptionsParser doesn't to support *.jar currently.
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

        jars.add(JarManager.findContainingJar(JexlException.class));

        return StringUtils.join(jars, NNConstants.LIB_JAR_SEPARATOR);
    }

    /**
     * Wrapper through {@link TrainModelProcessor} and a MapReduce job to analyze biggest sensitivity RMS.
     */
    private void distributedSEWrapper() throws Exception {
        // 1. Train a model using current selected variables, if no variables selected, use all candidate variables.
        TrainModelProcessor trainModelProcessor = new TrainModelProcessor();
        trainModelProcessor.setForVarSelect(true);
        trainModelProcessor.run();

        // 2. Submit a MapReduce job to analyze sensitivity RMS.
        SourceType source = this.modelConfig.getDataSet().getSource();
        Configuration conf = new Configuration();
        prepareSEJobConf(source, conf);

        Job job = new Job(conf, "Shifu: Variable Selection Wrapper Job : " + this.modelConfig.getModelSetName());
        job.setJarByClass(getClass());
        boolean isSEVarSelMulti = Boolean.TRUE.toString().equalsIgnoreCase(
                Environment.getProperty(Constants.SHIFU_VARSEL_SE_MULTI, Constants.SHIFU_DEFAULT_VARSEL_SE_MULTI));
        if(isSEVarSelMulti) {
            job.setMapperClass(MultithreadedMapper.class);
            MultithreadedMapper.setMapperClass(job, VarSelectMapper.class);
            int threads;
            try {
                threads = Integer.parseInt(Environment.getProperty(Constants.SHIFU_VARSEL_SE_MULTI_THREAD,
                        Constants.SHIFU_DEFAULT_VARSEL_SE_MULTI_THREAD + ""));
            } catch (Exception e) {
                Log.warn("'shifu.varsel.se.multi.thread' should be a int value, set default value: {}",
                        Constants.SHIFU_DEFAULT_VARSEL_SE_MULTI_THREAD);
                threads = Constants.SHIFU_DEFAULT_VARSEL_SE_MULTI_THREAD;
            }
            MultithreadedMapper.setNumberOfThreads(job, threads);
        } else {
            job.setMapperClass(VarSelectMapper.class);
        }

        job.setMapOutputKeyClass(LongWritable.class);
        job.setMapOutputValueClass(ColumnInfo.class);
        job.setInputFormatClass(TextInputFormat.class);
        FileInputFormat.setInputPaths(
                job,
                ShifuFileUtils.getFileSystemBySourceType(source).makeQualified(
                        new Path(super.getPathFinder().getNormalizedDataPath())));

        job.setReducerClass(VarSelectReducer.class);
        // Only one reducer, no need set combiner because of distinct keys in map outputs.
        job.setNumReduceTasks(1);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        String varSelectMSEOutputPath = super.getPathFinder().getVarSelectMSEOutputPath(source);
        FileOutputFormat.setOutputPath(job, new Path(varSelectMSEOutputPath));
        MultipleOutputs.addNamedOutput(job, Constants.SHIFU_VARSELECT_SE_OUTPUT_NAME, TextOutputFormat.class,
                Text.class, Text.class);

        // clean output firstly
        ShifuFileUtils.deleteFile(varSelectMSEOutputPath, source);

        // submit job
        if(job.waitForCompletion(true)) {
            postProcess4SEVarSelect(source, varSelectMSEOutputPath);
        } else {
            log.error("VarSelect SE hadoop job is failed, please re-try varselect step.");
        }

    }

    private void prepareSEJobConf(SourceType source, Configuration conf) throws IOException {
        // add jars to hadoop mapper and reducer
        new GenericOptionsParser(conf, new String[] { "-libjars", addRuntimeJars() });

        conf.setBoolean(GuaguaMapReduceConstants.MAPRED_MAP_TASKS_SPECULATIVE_EXECUTION, true);
        conf.setBoolean(GuaguaMapReduceConstants.MAPRED_REDUCE_TASKS_SPECULATIVE_EXECUTION, true);
        conf.set(
                Constants.SHIFU_MODEL_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(source)
                        .makeQualified(new Path(super.getPathFinder().getModelConfigPath(source))).toString());
        conf.set(
                Constants.SHIFU_COLUMN_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(source)
                        .makeQualified(new Path(super.getPathFinder().getColumnConfigPath(source))).toString());
        conf.set(NNConstants.MAPRED_JOB_QUEUE_NAME, Environment.getProperty(Environment.HADOOP_JOB_QUEUE, "default"));
        conf.set(Constants.SHIFU_MODELSET_SOURCE_TYPE, source.toString());
        // set mapreduce.job.max.split.locations to 30 to suppress warnings
        conf.setInt(GuaguaMapReduceConstants.MAPREDUCE_JOB_MAX_SPLIT_LOCATIONS, 30);
        // Tmp set to false because of some cluster by default use gzip while CombineInputFormat will split gzip file (a
        // bug)
        conf.setBoolean(CombineInputFormat.SHIFU_VS_SPLIT_COMBINABLE, false);

        Float wrapperRatio = this.modelConfig.getVarSelect().getWrapperRatio();
        if(wrapperRatio == null) {
            log.warn("wrapperRatio in var select is not set. Using default value 0.05.");
            wrapperRatio = 0.05f;
        }

        if(wrapperRatio.compareTo(Float.valueOf(1.0f)) >= 0) {
            throw new IllegalArgumentException("WrapperRatio should be in (0, 1).");
        }
        conf.setFloat(Constants.SHIFU_VARSELECT_WRAPPER_RATIO, wrapperRatio);
    }

    private void postProcess4SEVarSelect(SourceType source, String varSelectMSEOutputPath) throws IOException {
        String outputFilePattern = varSelectMSEOutputPath + Path.SEPARATOR + "part-r-*";
        if(!ShifuFileUtils.isFileExists(outputFilePattern, source)) {
            throw new RuntimeException("Var select MSE stats output file not exist.");
        }

        for(ColumnConfig config: super.columnConfigList) {
            if(config.isFinalSelect()) {
                config.setFinalSelect(false);
            }
        }

        List<Scanner> scanners = null;
        try {
            // here only works for 1 reducer
            FileStatus[] globStatus = ShifuFileUtils.getFileSystemBySourceType(source).globStatus(
                    new Path(outputFilePattern));
            if(globStatus == null || globStatus.length == 0) {
                throw new RuntimeException("Var select MSE stats output file not exist.");
            }
            scanners = ShifuFileUtils.getDataScanners(globStatus[0].getPath().toString(), source);
            String str = null;
            int count = 0;
            Scanner scanner = scanners.get(0);
            while(scanner.hasNext()) {
                ++count;
                str = scanner.nextLine().trim();
                ColumnConfig columnConfig = this.columnConfigList.get(Integer.parseInt(str));
                columnConfig.setFinalSelect(true);
                log.info("Variable {} is selected.", columnConfig.getColumnName());
            }
            log.info("{} variables are selected.", count);
            log.info(
                    "Sensitivity analysis report is in {}/{}-* file(s) with format 'column_index\tcolumn_name\tmean\trms\tvariance'.",
                    varSelectMSEOutputPath, Constants.SHIFU_VARSELECT_SE_OUTPUT_NAME);
        } finally {
            if(scanners != null) {
                for(Scanner scanner: scanners) {
                    if(scanner != null) {
                        scanner.close();
                    }
                }
            }
        }

        this.saveColumnConfigList();
        this.syncDataToHdfs(this.modelConfig.getDataSet().getSource());
    }

    private void setHeapSizeAndSplitSize(final List<String> args) {
        // args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, GuaguaMapReduceConstants.MAPRED_CHILD_JAVA_OPTS,
        // "-Xmn128m -Xms1G -Xmx1G -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps"));
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, GuaguaMapReduceConstants.MAPRED_CHILD_JAVA_OPTS,
                "-Xmn128m -Xms1G -Xmx1G"));
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_SPLIT_COMBINABLE,
                Environment.getProperty(GuaguaConstants.GUAGUA_SPLIT_COMBINABLE, "true")));
        args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT,
                GuaguaConstants.GUAGUA_SPLIT_MAX_COMBINED_SPLIT_SIZE,
                Environment.getProperty(GuaguaConstants.GUAGUA_SPLIT_MAX_COMBINED_SPLIT_SIZE, "268435456")));
    }

    /**
     * user wrapper to select variable
     * 
     * @param selector
     * @throws Exception
     */
    private void wrapper(VariableSelector selector) throws Exception {
        NormalizeModelProcessor n = new NormalizeModelProcessor();

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
