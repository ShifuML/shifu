/*
 * Copyright [2012-2014] PayPal Software Foundation
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

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;

import org.apache.commons.collections.ListUtils;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.io.IOUtils;
import org.apache.commons.jexl2.JexlException;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.map.MultithreadedMapper;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.pig.impl.util.JarManager;
import org.apache.zookeeper.ZooKeeper;
import org.encog.ml.BasicML;
import org.encog.ml.data.MLDataSet;
import org.jboss.netty.bootstrap.ServerBootstrap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Splitter;

import ml.shifu.guagua.GuaguaConstants;
import ml.shifu.guagua.hadoop.util.HDPUtils;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceClient;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceConstants;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.TreeModel;
import ml.shifu.shifu.core.VariableSelector;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.core.dvarsel.VarSelMaster;
import ml.shifu.shifu.core.dvarsel.VarSelMasterResult;
import ml.shifu.shifu.core.dvarsel.VarSelOutput;
import ml.shifu.shifu.core.dvarsel.VarSelWorker;
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;
import ml.shifu.shifu.core.dvarsel.wrapper.CandidateGenerator;
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

/**
 * Variable selection processor, select the variable based on KS/IV value, or
 * </p>
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

    private boolean isToReset = false;

    public VarSelectModelProcessor() {
        // default constructor
    }

    public VarSelectModelProcessor(Map<String, Object> otherConfigs) {
        super.otherConfigs = otherConfigs;
    }

    public VarSelectModelProcessor(boolean isToReset) {
        this.isToReset = isToReset;
    }

    @SuppressWarnings("unused")
    private static final double BAD_IV_THRESHOLD = 0.02d;

    
    private void validateParameters()throws Exception{
        //String alg = super.getModelConfig().getTrain().getAlgorithm();
        String filterBy = this.modelConfig.getVarSelectFilterBy();
        if(filterBy.equalsIgnoreCase(Constants.FILTER_BY_SE)||filterBy.equalsIgnoreCase(Constants.FILTER_BY_ST)){
            validateSEParameters();
            validateNormalize();
        }
    }
    private void prepareSelect() throws Exception{
        setUp(ModelStep.VARSELECT);
        validateParameters();
        //reset all selections if user specify or select by absolute number
        if(isToReset) {
            log.info("Reset all selections data including type,final select etc!");
            resetAllFinalSelect();
        }
        if(this.modelConfig.getVarSelectFilterNum()>0){
            for(ColumnConfig columnConfig: this.columnConfigList) {
                if(columnConfig.isFinalSelect()) {
                    columnConfig.setFinalSelect(false);
                }
            }
        }
        // sync to make sure load from hdfs config is consistent with local configuration
        syncDataToHdfs(super.modelConfig.getDataSet().getSource());
    }
    
    //private boolean is
    
    /**
     * Run for the variable selection
     */
    @Override
    public int run() throws Exception {
        log.info("Step Start: varselect");
        long start = System.currentTimeMillis();
        try {
             prepareSelect();
             if(modelConfig.isRegression()){
                 VariableSelector selector = new VariableSelector(this.modelConfig, this.columnConfigList);
                 String filterBy = this.modelConfig.getVarSelectFilterBy();
                 if(filterBy.equalsIgnoreCase(Constants.FILTER_BY_KS)||filterBy.equalsIgnoreCase(Constants.FILTER_BY_IV)||
                         filterBy.equalsIgnoreCase(Constants.FILTER_BY_PARETO)||filterBy.equalsIgnoreCase(Constants.FILTER_BY_MIX)){
                     CommonUtils.updateColumnConfigFlags(modelConfig, columnConfigList);
                     this.columnConfigList = selector.selectByFilter();
                 }else if(filterBy.equalsIgnoreCase(Constants.FILTER_BY_FI)){
                     selectByFeatureImportance();
                 }else if(filterBy.equalsIgnoreCase(Constants.FILTER_BY_SE)||filterBy.equalsIgnoreCase(Constants.FILTER_BY_ST)){
                     distributedSEWrapper();
                 }else if(filterBy.equalsIgnoreCase(Constants.FILTER_BY_VOTED)){
                     votedVariablesSelection();
                 }
             }else {
                 // multiple classification, select all candidate at first, TODO add SE for multi-classification
                 for(ColumnConfig config: this.columnConfigList) {
                     if(CommonUtils.isGoodCandidate(modelConfig.isRegression(), config)) {
                         config.setFinalSelect(true);
                     }
                 }
             }
            // save column config to file and sync to
            clearUp(ModelStep.VARSELECT);
        } catch (Exception e) {
            log.error("Error:", e);
            return -1;
        }
        log.info("Step Finished: varselect with {} ms", (System.currentTimeMillis() - start));
        return 0;
    }
    
    private void selectByFeatureImportance() throws Exception{
        List<BasicML> models = CommonUtils.loadBasicModels(this.modelConfig, this.columnConfigList,
                null);
        if(models == null || models.size() < 1) {
            TrainModelProcessor trainModelProcessor = new TrainModelProcessor();
            trainModelProcessor.setForVarSelect(true);
            trainModelProcessor.run();
        }
        List<Map<Integer, MutablePair<String, Double>>> importanceList = new ArrayList<Map<Integer, MutablePair<String, Double>>>();
        Map<Integer, MutablePair<String, Double>> mergedResult = null;
        for(BasicML basicModel: models) {
            if(basicModel instanceof TreeModel) {
                TreeModel model = (TreeModel) basicModel;
                Map<Integer, MutablePair<String, Double>> importances = model
                        .getFeatureImportances();
                importanceList.add(importances);
            }
        }
        if(importanceList.size() < 1) {
            throw new IllegalArgumentException(
                    "Feature importance calculation abort due to no tree model found!!");
        }
        mergedResult = this.mergeImportanceList(importanceList);
        this.writeFeatureImportance(mergedResult);
        if(super.modelConfig.getVarSelect().getFilterEnable()) {
            this.postProcessFIVarSelect(mergedResult);
        }
    }

    private Map<Integer, MutablePair<String, Double>> mergeImportanceList(
            List<Map<Integer, MutablePair<String, Double>>> list) {
        Map<Integer, MutablePair<String, Double>> finalResult = new HashMap<Integer, MutablePair<String, Double>>();
        int size = list.size();
        for(Map<Integer, MutablePair<String, Double>> item: list) {
            for(Entry<Integer, MutablePair<String, Double>> entry: item.entrySet()) {
                if(!finalResult.containsKey(entry.getKey())) {
                    MutablePair<String, Double> value = MutablePair.of(entry.getValue().getKey(),
                            entry.getValue().getValue() / size);
                    finalResult.put(entry.getKey(), value);
                } else {
                    MutablePair<String, Double> current = finalResult.get(entry.getKey());
                    double entryValue = entry.getValue().getValue();
                    current.setValue(current.getValue() + entryValue / size);
                    finalResult.put(entry.getKey(), current);
                }
            }
        }
        return TreeModel.sortByValue(finalResult, false);
    }

    private void writeFeatureImportance(Map<Integer, MutablePair<String, Double>> importances) throws IOException {
        ShifuFileUtils.createFileIfNotExists(this.pathFinder.getLocalFeatureImportancePath(), SourceType.LOCAL);
        BufferedWriter writer = ShifuFileUtils.getWriter(this.pathFinder.getLocalFeatureImportancePath(),
                SourceType.LOCAL);
        log.info("Writing feature importances to file "+this.pathFinder.getLocalFeatureImportancePath());
        try {
            writer.write("column_id\t\tcolumn_name\t\timportance");
            writer.newLine();
            for(Map.Entry<Integer, MutablePair<String, Double>> entry: importances.entrySet()) {
                String content = entry.getKey() + "\t\t" + entry.getValue().getKey() + "\t\t"
                        + entry.getValue().getValue();
                writer.write(content);
                writer.newLine();
            }
            writer.flush();
        } finally {
            IOUtils.closeQuietly(writer);
        }
    }

    public void resetAllFinalSelect() {
        log.info("!!! Reset all variables finalSelect = false");
        for(ColumnConfig columnConfig: this.columnConfigList) {
            columnConfig.setFinalSelect(false);
            columnConfig.setColumnFlag(null);
        }

        try {
            CommonUtils.updateColumnConfigFlags(this.modelConfig, this.columnConfigList);
        } catch (IOException e) {
            log.error("Fail to update ColumnConfig.json flags.", e);
        }
    }

    private void validateNormalize() throws IOException {
        if(!ShifuFileUtils.isFileExists(
                new PathFinder(modelConfig).getNormalizedDataPath(this.modelConfig.getDataSet().getSource()),
                this.modelConfig.getDataSet().getSource())) {
            throw new IllegalStateException("Cannot find normalized data, please do 'Shifu normalize' firstly.");
        }
    }

    private void validateSEParameters() {
        if(!NNConstants.NN_ALG_NAME.equalsIgnoreCase(super.getModelConfig().getTrain().getAlgorithm())
                && !"LR".equalsIgnoreCase(super.getModelConfig().getTrain().getAlgorithm())) {
            throw new IllegalArgumentException(
                    "Currently we only support NN and LR distributed training to do wrapper by analyzing variable selection.");
        }

        if(super.getModelConfig().getDataSet().getSource() != SourceType.HDFS) {
            throw new IllegalArgumentException(
                    "Currently we only support distributed wrapper by analyzing on HDFS source type.");
        }

        if(!super.getModelConfig().isMapReduceRunMode()) {
            throw new IllegalArgumentException(
                    "Currently we only support distributed wrapper by on MAPRED or DIST mode.");
        }
    }

    private void votedVariablesSelection() throws ClassNotFoundException, IOException, InterruptedException {
        log.info("Start voted variables selection ");
        // sync data back to hdfs
        super.syncDataToHdfs(modelConfig.getDataSet().getSource());

        SourceType sourceType = super.getModelConfig().getDataSet().getSource();
        Configuration conf = new Configuration();

        final List<String> args = new ArrayList<String>();
        // prepare parameter
        prepareVarSelParams(args, sourceType);

        Path columnIdsPath = getVotedSelectionPath(sourceType);
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                ml.shifu.shifu.util.Constants.VAR_SEL_COLUMN_IDS_OUPUT, columnIdsPath.toString()));

        long start = System.currentTimeMillis();

        GuaguaMapReduceClient guaguaClient = new GuaguaMapReduceClient();

        String hdpVersion = HDPUtils.getHdpVersionForHDP224();
        if(StringUtils.isNotBlank(hdpVersion)) {
            // for hdp 2.2.4, hdp.version should be set and configuration files should be add to container class path
            conf.set("hdp.version", hdpVersion);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("hdfs-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("core-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("mapred-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("yarn-site.xml"), conf);
        }
        guaguaClient.createJob(args.toArray(new String[0])).waitForCompletion(true);

        log.info("Voted variables selection finished in {}ms.", System.currentTimeMillis() - start);

        persistColumnIds(columnIdsPath);
        super.syncDataToHdfs(sourceType);
    }

    private int persistColumnIds(Path path) {
        try {
            List<Scanner> scanners = ShifuFileUtils.getDataScanners(path.toString(),
                    modelConfig.getDataSet().getSource());

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

            super.saveColumnConfigListAndColumnStats(false);

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
        return ShifuFileUtils.getFileSystemBySourceType(sourceType)
                .makeQualified(new Path(getPathFinder().getVarSelsPath(sourceType), "VarSels"));
    }

    @SuppressWarnings("unused")
    private void prepareVarSelParams(final List<String> args, final SourceType sourceType) {
        args.add("-libjars");

        args.add(addRuntimeJars());

        args.add("-i");
        args.add(ShifuFileUtils.getFileSystemBySourceType(sourceType)
                .makeQualified(new Path(modelConfig.getDataSetRawPath())).toString());

        String zkServers = Environment.getProperty(Environment.ZOO_KEEPER_SERVERS);
        if(StringUtils.isEmpty(zkServers)) {
            log.warn(
                    "No specified zookeeper settings from zookeeperServers in shifuConfig file, Guagua will set embeded zookeeper server in client process. For big data applications, specified zookeeper servers are strongly recommended.");
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

        int iterationCnt = (Integer) this.modelConfig.getVarSelect().getParams()
                .get(CandidateGenerator.POPULATION_MULTIPLY_CNT) + 1;
        args.add(Integer.toString(iterationCnt));

        args.add("-mr");
        args.add(VarSelMasterResult.class.getName());

        args.add("-wr");
        args.add(VarSelWorkerResult.class.getName());

        // setting conductor
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                ml.shifu.shifu.util.Constants.VAR_SEL_MASTER_CONDUCTOR,
                Environment.getProperty(Environment.VAR_SEL_MASTER_CONDUCTOR, WrapperMasterConductor.class.getName())));

        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                ml.shifu.shifu.util.Constants.VAR_SEL_WORKER_CONDUCTOR,
                Environment.getProperty(Environment.VAR_SEL_MASTER_CONDUCTOR, WrapperWorkerConductor.class.getName())));

        // setting queue
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.MAPRED_JOB_QUEUE_NAME, Environment
                .getProperty(Environment.HADOOP_JOB_QUEUE, ml.shifu.shifu.util.Constants.DEFAULT_JOB_QUEUE)));

        // MAPRED timeout
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, NNConstants.MAPRED_TASK_TIMEOUT, Environment
                .getInt(NNConstants.MAPRED_TASK_TIMEOUT, ml.shifu.shifu.util.Constants.DEFAULT_MAPRED_TIME_OUT)));

        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_MASTER_INTERCEPTERS,
                VarSelOutput.class.getName()));

        // setting model config column config
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.SHIFU_MODEL_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(sourceType)
                        .makeQualified(new Path(super.getPathFinder().getModelConfigPath(sourceType)))));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.SHIFU_COLUMN_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(sourceType)
                        .makeQualified(new Path(super.getPathFinder().getColumnConfigPath(sourceType)))));

        // source type
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, CommonConstants.MODELSET_SOURCE_TYPE,
                sourceType));

        // computation time
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                GuaguaConstants.GUAGUA_COMPUTATION_TIME_THRESHOLD, 60 * 60 * 1000l));
        setHeapSizeAndSplitSize(args);

        // one can set guagua conf in shifuconfig
        for(Map.Entry<Object, Object> entry: Environment.getProperties().entrySet()) {
            if(CommonUtils.isHadoopConfigurationInjected(entry.getKey().toString())) {
                args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, entry.getKey().toString(),
                        entry.getValue().toString()));
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
        // netty-*.jar
        jars.add(JarManager.findContainingJar(ServerBootstrap.class));

        jars.add(JarManager.findContainingJar(JexlException.class));

        String hdpVersion = HDPUtils.getHdpVersionForHDP224();
        if(StringUtils.isNotBlank(hdpVersion)) {
            // for hdp 2.2.4, hdp.version should be set and configuration files should be add to container class path
            jars.add(HDPUtils.findContainingFile("hdfs-site.xml"));
            jars.add(HDPUtils.findContainingFile("core-site.xml"));
            jars.add(HDPUtils.findContainingFile("mapred-site.xml"));
            jars.add(HDPUtils.findContainingFile("yarn-site.xml"));
        }

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
        // 2.1 prepare se job conf
        prepareSEJobConf(source, conf);
        // 2.2 get output path
        String varSelectMSEOutputPath = super.getPathFinder().getVarSelectMSEOutputPath(source);

        // 2.3 create se job
        Job job = createSEMapReduceJob(source, conf, varSelectMSEOutputPath);

        // 2.4 clean output firstly
        ShifuFileUtils.deleteFile(varSelectMSEOutputPath, source);

        // 2.5 submit job
        if(job.waitForCompletion(true)) {
            // 2.6 post process 4 var select
            if(super.modelConfig.getVarSelect().getFilterEnable()) {
                postProcess4SEVarSelect(source, varSelectMSEOutputPath);
            } else {
                log.info("Only print sensitivity analysis report.");
                log.info(
                        "Sensitivity analysis report is in {}/{}-* file(s) with format 'column_index\tcolumn_name\tmean\trms\tvariance'.",
                        varSelectMSEOutputPath, Constants.SHIFU_VARSELECT_SE_OUTPUT_NAME);
            }
        } else {
            log.error("VarSelect SE hadoop job is failed, please re-try varselect step.");
        }

    }

    private Job createSEMapReduceJob(SourceType source, Configuration conf, String varSelectMSEOutputPath)
            throws IOException {
        @SuppressWarnings("deprecation")
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
                log.warn("'shifu.varsel.se.multi.thread' should be a int value, set default value: {}",
                        Constants.SHIFU_DEFAULT_VARSEL_SE_MULTI_THREAD);
                threads = Constants.SHIFU_DEFAULT_VARSEL_SE_MULTI_THREAD;
            }
            MultithreadedMapper.setNumberOfThreads(job, threads);
        } else {
            job.setMapperClass(VarSelectMapper.class);
        }

        job.setMapOutputKeyClass(LongWritable.class);
        job.setMapOutputValueClass(ColumnInfo.class);
        job.setInputFormatClass(CombineInputFormat.class);
        FileInputFormat.setInputPaths(job, ShifuFileUtils.getFileSystemBySourceType(source)
                .makeQualified(new Path(super.getPathFinder().getNormalizedDataPath())));

        job.setReducerClass(VarSelectReducer.class);
        // Only one reducer, no need set combiner because of distinct keys in map outputs.
        job.setNumReduceTasks(1);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileOutputFormat.setOutputPath(job, new Path(varSelectMSEOutputPath));
        MultipleOutputs.addNamedOutput(job, Constants.SHIFU_VARSELECT_SE_OUTPUT_NAME, TextOutputFormat.class,
                Text.class, Text.class);
        return job;
    }

    private void prepareSEJobConf(SourceType source, Configuration conf) throws IOException {
        // add jars to hadoop mapper and reducer
        new GenericOptionsParser(conf, new String[] { "-libjars", addRuntimeJars() });

        conf.setBoolean(CombineInputFormat.SHIFU_VS_SPLIT_COMBINABLE, true);

        conf.setBoolean(GuaguaMapReduceConstants.MAPRED_MAP_TASKS_SPECULATIVE_EXECUTION, true);
        conf.setBoolean(GuaguaMapReduceConstants.MAPRED_REDUCE_TASKS_SPECULATIVE_EXECUTION, true);
        conf.set(Constants.SHIFU_MODEL_CONFIG, ShifuFileUtils.getFileSystemBySourceType(source)
                .makeQualified(new Path(super.getPathFinder().getModelConfigPath(source))).toString());
        conf.set(Constants.SHIFU_COLUMN_CONFIG, ShifuFileUtils.getFileSystemBySourceType(source)
                .makeQualified(new Path(super.getPathFinder().getColumnConfigPath(source))).toString());
        conf.set(NNConstants.MAPRED_JOB_QUEUE_NAME, Environment.getProperty(Environment.HADOOP_JOB_QUEUE, "default"));
        conf.set(Constants.SHIFU_MODELSET_SOURCE_TYPE, source.toString());
        // set mapreduce.job.max.split.locations to 100 to suppress warnings
        conf.setInt(GuaguaMapReduceConstants.MAPREDUCE_JOB_MAX_SPLIT_LOCATIONS, 100);
        // Tmp set to false because of some cluster by default use gzip while CombineInputFormat will split gzip file (a
        // bug)
        conf.setBoolean(CombineInputFormat.SHIFU_VS_SPLIT_COMBINABLE, false);
        conf.set("mapred.reduce.slowstart.completed.maps",
                Environment.getProperty("mapred.reduce.slowstart.completed.maps", "0.9"));

        Float filterOutRatio = this.modelConfig.getVarSelect().getFilterOutRatio();
        if(filterOutRatio == null) {
            log.warn("filterOutRatio in var select is not set. Using default value 0.05.");
            filterOutRatio = 0.05f;
        }

        if(filterOutRatio.compareTo(Float.valueOf(1.0f)) >= 0) {
            throw new IllegalArgumentException("WrapperRatio should be in (0, 1).");
        }
        conf.setFloat(Constants.SHIFU_VARSELECT_FILTEROUT_RATIO, filterOutRatio);
        String hdpVersion = HDPUtils.getHdpVersionForHDP224();
        if(StringUtils.isNotBlank(hdpVersion)) {
            // for hdp 2.2.4, hdp.version should be set and configuration files should be add to container class path
            conf.set("hdp.version", hdpVersion);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("hdfs-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("core-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("mapred-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("yarn-site.xml"), conf);
        }
        // one can set guagua conf in shifuconfig
        for(Map.Entry<Object, Object> entry: Environment.getProperties().entrySet()) {
            if(CommonUtils.isHadoopConfigurationInjected(entry.getKey().toString())) {
                conf.set(entry.getKey().toString(), entry.getValue().toString());
            }
        }
    }

    private void postProcessFIVarSelect(Map<Integer, MutablePair<String, Double>> importances) {
        int selectCnt = 0;
        for(ColumnConfig config: super.columnConfigList) {
            // enable ForceSelect
            if(config.isForceSelect()) {
                config.setFinalSelect(true);
                selectCnt++;
                log.info("Variable {} is selected, since it is in ForceSelect list.", config.getColumnName());
            }
        }
        VariableSelector.setFilterNumberByFilterOutRatio(this.modelConfig, this.columnConfigList);
        int targetCnt = this.modelConfig.getVarSelectFilterNum();
        List<Integer> candidateColumnIdList = new ArrayList<Integer>();
        candidateColumnIdList.addAll(importances.keySet());
        int i = 0;
        // try to select another (targetCnt - selectCnt) variables, but we need to exclude those
        // force-selected variables
        for(ColumnConfig columnConfig: this.columnConfigList) {
            if(columnConfig.isFinalSelect()) {
                columnConfig.setFinalSelect(false);
            }
        }
        while(selectCnt < targetCnt && i < targetCnt) {
            Integer columnId = candidateColumnIdList.get(i++);
            ColumnConfig columnConfig = this.columnConfigList.get(columnId);
            if(!columnConfig.isForceSelect() && !columnConfig.isForceRemove()) {
                columnConfig.setFinalSelect(true);
                selectCnt++;
                log.info("Variable {} is selected.", columnConfig.getColumnName());
            }
        }
        log.info("{} variables are selected.", selectCnt);
    }

    private void postProcess4SEVarSelect(SourceType source, String varSelectMSEOutputPath) throws IOException {
        String outputFilePattern = varSelectMSEOutputPath + Path.SEPARATOR + "part-r-*";
        if(!ShifuFileUtils.isFileExists(outputFilePattern, source)) {
            throw new RuntimeException("Var select MSE stats output file not exist.");
        }

        int selectCnt = 0;
        for(ColumnConfig config: super.columnConfigList) {
            if(config.isFinalSelect()) {
                config.setFinalSelect(false);
            }

            // enable ForceSelect
            if(config.isForceSelect()) {
                config.setFinalSelect(true);
                selectCnt++;
                log.info("Variable {} is selected, since it is in ForceSelect list.", config.getColumnName());
            }
        }

        List<Scanner> scanners = null;
        try {
            // here only works for 1 reducer
            FileStatus[] globStatus = ShifuFileUtils.getFileSystemBySourceType(source)
                    .globStatus(new Path(outputFilePattern));
            if(globStatus == null || globStatus.length == 0) {
                throw new RuntimeException("Var select MSE stats output file not exist.");
            }
            scanners = ShifuFileUtils.getDataScanners(globStatus[0].getPath().toString(), source);
            String str = null;
            int targetCnt = 0; // total variable count that user want to select
            List<Integer> candidateColumnIdList = new ArrayList<Integer>();
            Scanner scanner = scanners.get(0);
            while(scanner.hasNext()) {
                ++targetCnt;
                str = scanner.nextLine().trim();
                candidateColumnIdList.add(Integer.parseInt(str));
            }

            int i = 0;
            // try to select another (targetCnt - selectCnt) variables, but we need to exclude those
            // force-selected variables
            while(selectCnt < targetCnt && i < targetCnt) {
                Integer columnId = candidateColumnIdList.get(i++);
                ColumnConfig columnConfig = this.columnConfigList.get(columnId);
                if(!columnConfig.isForceSelect() && !columnConfig.isForceRemove()) {
                    columnConfig.setFinalSelect(true);
                    selectCnt++;
                    log.info("Variable {} is selected.", columnConfig.getColumnName());
                }
            }

            log.info("{} variables are selected.", selectCnt);
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
    }

    @Override
    protected void clearUp(ModelStep step) throws IOException {
        autoVarSelCondition();
        try {
            this.saveColumnConfigListAndColumnStats(true);
        } catch (Exception e) {
            throw new ShifuException(ShifuErrorCode.ERROR_WRITE_COLCONFIG, e);
        }
        this.syncDataToHdfs(this.modelConfig.getDataSet().getSource());
    }

    /**
     * To do some auto variable selection like remove ID-like variables, remove variable with high missing rate.
     */
    private void autoVarSelCondition() {
        // here we do loop again as it is not bad for variables less than 100,000
        for(ColumnConfig config: columnConfigList) {
            if(isHighMissingRateColumn(config) && !config.isForceSelect()) {
                log.warn(
                        "Column {} is with very high missing rate, set final select to false. If not, you can check it manually in ColumnConfig.json",
                        config.getColumnName());
                config.setFinalSelect(false);
                continue;
            }
        }
    }

    /**
     * Check is high rate is very high.
     */
    private boolean isHighMissingRateColumn(ColumnConfig config) {
        Double missingPercentage = config.getMissingPercentage();
        if(missingPercentage != null && missingPercentage >= modelConfig.getVarSelect().getMissingRateThreshold()) {
            return true;
        }
        return false;
    }

    /**
     * Check if column is ID-like.
     */
    @SuppressWarnings("unused")
    private boolean isIDLikeVariable(ColumnConfig config) {
        Long distinctCount = config.getColumnStats().getDistinctCount();
        Long totalCount = config.getColumnStats().getTotalCount();
        if(totalCount != null && distinctCount != null && totalCount >= 10000
                && distinctCount * 1.0 / totalCount >= 0.97d) {
            return true;
        }

        return false;
    }

    private void setHeapSizeAndSplitSize(final List<String> args) {
        // args.add(String.format(NNConstants.MAPREDUCE_PARAM_FORMAT, GuaguaMapReduceConstants.MAPRED_CHILD_JAVA_OPTS,
        // "-Xmn128m -Xms1G -Xmx1G -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps"));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaMapReduceConstants.MAPRED_CHILD_JAVA_OPTS,
                "-Xmn128m -Xms1G -Xmx1G"));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT, GuaguaConstants.GUAGUA_SPLIT_COMBINABLE,
                Environment.getProperty(GuaguaConstants.GUAGUA_SPLIT_COMBINABLE, "true")));
        args.add(String.format(CommonConstants.MAPREDUCE_PARAM_FORMAT,
                GuaguaConstants.GUAGUA_SPLIT_MAX_COMBINED_SPLIT_SIZE,
                Environment.getProperty(GuaguaConstants.GUAGUA_SPLIT_MAX_COMBINED_SPLIT_SIZE, "268435456")));
    }


}
