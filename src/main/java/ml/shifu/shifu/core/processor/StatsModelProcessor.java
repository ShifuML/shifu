/*
 * Copyright [2012-2014] PayPal Software Foundation
 * <p/>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p/>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p/>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.processor;

import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.SortedMap;
import java.util.TreeMap;

import ml.shifu.guagua.hadoop.util.HDPUtils;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceConstants;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.guagua.util.ReflectionUtils;
import ml.shifu.shifu.column.NSColumnUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnFlag;
import ml.shifu.shifu.container.obj.ModelStatsConf;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ColumnStatsCalculator;
import ml.shifu.shifu.core.binning.ColumnConfigDynamicBinning;
import ml.shifu.shifu.core.binning.obj.AbstractBinInfo;
import ml.shifu.shifu.core.binning.obj.CategoricalBinInfo;
import ml.shifu.shifu.core.binning.obj.NumericalBinInfo;
import ml.shifu.shifu.core.correlation.CorrelationMapper;
import ml.shifu.shifu.core.correlation.CorrelationMultithreadedMapper;
import ml.shifu.shifu.core.correlation.CorrelationReducer;
import ml.shifu.shifu.core.correlation.CorrelationWritable;
import ml.shifu.shifu.core.correlation.FastCorrelationMapper;
import ml.shifu.shifu.core.correlation.FastCorrelationMultithreadedMapper;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.core.mr.input.CombineInputFormat;
import ml.shifu.shifu.core.processor.stats.AbstractStatsExecutor;
import ml.shifu.shifu.core.processor.stats.AkkaStatsWorker;
import ml.shifu.shifu.core.processor.stats.DIBStatsExecutor;
import ml.shifu.shifu.core.processor.stats.MapReducerStatsWorker;
import ml.shifu.shifu.core.processor.stats.MunroPatIStatsExecutor;
import ml.shifu.shifu.core.processor.stats.MunroPatStatsExecutor;
import ml.shifu.shifu.core.processor.stats.SPDTIStatsExecutor;
import ml.shifu.shifu.core.processor.stats.SPDTStatsExecutor;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.fs.SourceFile;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;

import org.apache.commons.codec.binary.Base64;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.io.IOUtils;
import org.apache.commons.jexl2.JexlException;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.pig.impl.util.JarManager;
import org.encog.ml.data.MLDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Predicate;
import com.google.common.base.Splitter;

/**
 * statistics, max/min/avg/std for each column dataset if it's numerical
 */
public class StatsModelProcessor extends BasicModelProcessor implements Processor {

    private final static Logger log = LoggerFactory.getLogger(StatsModelProcessor.class);

    public StatsModelProcessor(Map<String, Object> params) {
        this.params = params;
    }

    public StatsModelProcessor() {
    }

    /**
     * runner for statistics
     */
    @Override
    public int run() throws Exception {
        log.info("Step Start: stats");
        long start = System.currentTimeMillis();
        try {
            // 0. set up and sync to HDFS
            setUp(ModelStep.STATS);
            // resync ModelConfig.json/ColumnConfig.json to HDFS
            syncDataToHdfs(modelConfig.getDataSet().getSource());

            if(getBooleanParam(this.params, Constants.IS_COMPUTE_CORR)) {
                // 1. validate if run stats before run stats -correlation
                boolean foundValidMeanValueColumn = false;
                for(ColumnConfig config: this.columnConfigList) {
                    if(!config.isMeta() && !config.isTarget()) {
                        if(config.getMean() != null) {
                            foundValidMeanValueColumn = true;
                            break;
                        }
                    }
                }

                if(!foundValidMeanValueColumn) {
                    log.warn("Some mean value of column is null, could you check if you run 'shifu stats'.");
                    return -1;
                }

                // 2. compute correlation
                log.info("Start computing correlation value ...");

                SourceType source = this.modelConfig.getDataSet().getSource();
                String corrPath = super.getPathFinder().getCorrelationPath(source);

                // check if can start from existing output
                boolean reuseCorrResult = Environment.getBoolean("shifu.stats.corr.reuse", Boolean.FALSE);
                if(reuseCorrResult && ShifuFileUtils.isFileExists(corrPath, SourceType.HDFS)) {
                    dumpCorrelationResult(source, corrPath);
                } else {
                    runCorrMapReduceJob();
                }

                // 3. save column config list
                saveColumnConfigList();
            } else if(getBooleanParam(this.params, Constants.IS_COMPUTE_PSI)) {
                // 1. validate if run stats before run stats -correlation
                boolean foundValidMeanValueColumn = false;
                for(ColumnConfig config: this.columnConfigList) {
                    if(!config.isMeta() && !config.isTarget()) {
                        if(config.getMean() != null) {
                            foundValidMeanValueColumn = true;
                            break;
                        }
                    }
                }

                if(!foundValidMeanValueColumn) {
                    log.warn("Some mean value of column is null, could you check if you run 'shifu stats'.");
                    return -1;
                }

                if(StringUtils.isNotEmpty(modelConfig.getPsiColumnName())) {
                    new MapReducerStatsWorker(this, modelConfig, columnConfigList).runPSI();
                } else {
                    log.warn("To Run PSI please set your PSI column in dataSet::psiColumnName.");
                }
            } else if(getBooleanParam(this.params, Constants.IS_REBIN)) {
                // run the re-binning
                String backupColumnConfigPath = this.pathFinder.getBackupColumnConfig();
                if(!ShifuFileUtils.isFileExists(new Path(backupColumnConfigPath), SourceType.LOCAL)) {
                    ShifuFileUtils.createDirIfNotExists(new SourceFile(Constants.TMP, SourceType.LOCAL));
                    saveColumnConfigList(backupColumnConfigPath, this.columnConfigList);
                } else { // existing backup ColumnConfig.json, use binning info in it to do rebin
                    List<ColumnConfig> backColumnConfigList = CommonUtils.loadColumnConfigList(backupColumnConfigPath,
                            SourceType.LOCAL, false);
                    for(ColumnConfig backupColumnConfig: backColumnConfigList) {
                        for(ColumnConfig columnConfig: this.columnConfigList) {
                            if(NSColumnUtils.isColumnEqual(backupColumnConfig.getColumnName(),
                                    columnConfig.getColumnName())) {
                                columnConfig.setColumnBinning(backupColumnConfig.getColumnBinning());
                            }
                        }
                    }
                }

                List<ColumnConfig> rebinColumns = new ArrayList<ColumnConfig>();
                List<String> catVariables = getStringList(this.params, Constants.REQUEST_VARS, ",");
                for(ColumnConfig columnConfig: this.columnConfigList) {
                    if(CollectionUtils.isEmpty(catVariables) || isRequestColumn(catVariables, columnConfig)) {
                        rebinColumns.add(columnConfig);
                    }
                }

                if(CollectionUtils.isNotEmpty(rebinColumns)) {
                    for(ColumnConfig columnConfig: rebinColumns) {
                        doReBin(columnConfig);
                    }
                }

                // use the merge ColumnConfig.json to replace current one
                saveColumnConfigList();
            } else {
                AbstractStatsExecutor statsExecutor = null;

                if(modelConfig.isMapReduceRunMode()) {
                    if(modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.DynamicBinning)) {
                        statsExecutor = new DIBStatsExecutor(this, modelConfig, columnConfigList);
                    } else if(modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.MunroPat)) {
                        statsExecutor = new MunroPatStatsExecutor(this, modelConfig, columnConfigList);
                    } else if(modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.MunroPatI)) {
                        statsExecutor = new MunroPatIStatsExecutor(this, modelConfig, columnConfigList);
                    } else if(modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.SPDT)) {
                        statsExecutor = new SPDTStatsExecutor(this, modelConfig, columnConfigList);
                    } else if(modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.SPDTI)) {
                        statsExecutor = new SPDTIStatsExecutor(this, modelConfig, columnConfigList);
                    } else {
                        statsExecutor = new SPDTIStatsExecutor(this, modelConfig, columnConfigList);
                    }
                } else if(modelConfig.isLocalRunMode()) {
                    statsExecutor = new AkkaStatsWorker(this, modelConfig, columnConfigList);
                } else {
                    throw new ShifuException(ShifuErrorCode.ERROR_UNSUPPORT_MODE);
                }
                statsExecutor.doStats();

                // update the backup ColumnConfig.json after running stats
                String backupColumnConfigPath = this.pathFinder.getBackupColumnConfig();
                ShifuFileUtils.createDirIfNotExists(new SourceFile(Constants.TMP, SourceType.LOCAL));
                saveColumnConfigList(backupColumnConfigPath, this.columnConfigList);
            }

            syncDataToHdfs(modelConfig.getDataSet().getSource());
            clearUp(ModelStep.STATS);
        } catch (Exception e) {
            log.error("Error:", e);
            return -1;
        }

        log.info("Step Finished: stats with {} ms", (System.currentTimeMillis() - start));
        return 0;
    }

    // OptionsParser doesn't to support *.jar currently.
    private String addRuntimeJars() {
        List<String> jars = new ArrayList<String>(16);
        // common-codec
        jars.add(JarManager.findContainingJar(Base64.class));
        // commons-compress-*.jar
        jars.add(JarManager.findContainingJar(BZip2CompressorInputStream.class));
        // commons-lang-*.jar
        jars.add(JarManager.findContainingJar(StringUtils.class));
        // common-io-*.jar
        jars.add(JarManager.findContainingJar(org.apache.commons.io.IOUtils.class));
        // common-collections
        jars.add(JarManager.findContainingJar(Predicate.class));
        // guava-*.jar
        jars.add(JarManager.findContainingJar(Splitter.class));
        // guagua-core-*.jar
        jars.add(JarManager.findContainingJar(NumberFormatUtils.class));
        // shifu-*.jar
        jars.add(JarManager.findContainingJar(getClass()));
        // jexl-*.jar
        jars.add(JarManager.findContainingJar(JexlException.class));
        // encog-core-*.jar
        jars.add(JarManager.findContainingJar(MLDataSet.class));
        // jackson-databind-*.jar
        jars.add(JarManager.findContainingJar(ObjectMapper.class));
        // jackson-core-*.jar
        jars.add(JarManager.findContainingJar(JsonParser.class));
        // jackson-annotations-*.jar
        jars.add(JarManager.findContainingJar(JsonIgnore.class));

        return StringUtils.join(jars, NNConstants.LIB_JAR_SEPARATOR);
    }

    private void runCorrMapReduceJob() throws IOException, InterruptedException, ClassNotFoundException {
        SourceType source = this.modelConfig.getDataSet().getSource();
        Configuration conf = new Configuration();

        String modelConfigPath = ShifuFileUtils.getFileSystemBySourceType(source)
                .makeQualified(new Path(super.getPathFinder().getModelConfigPath(source))).toString();
        String columnConfigPath = ShifuFileUtils.getFileSystemBySourceType(source)
                .makeQualified(new Path(super.getPathFinder().getColumnConfigPath(source))).toString();

        // add jars and files to hadoop mapper and reducer
        new GenericOptionsParser(conf, new String[] { "-libjars", addRuntimeJars(), "-files",
                modelConfigPath + "," + columnConfigPath });

        conf.setBoolean(GuaguaMapReduceConstants.MAPRED_MAP_TASKS_SPECULATIVE_EXECUTION, true);
        conf.setBoolean(GuaguaMapReduceConstants.MAPRED_REDUCE_TASKS_SPECULATIVE_EXECUTION, true);
        conf.set(NNConstants.MAPRED_JOB_QUEUE_NAME, Environment.getProperty(Environment.HADOOP_JOB_QUEUE, "default"));
        conf.setInt(GuaguaMapReduceConstants.MAPREDUCE_JOB_MAX_SPLIT_LOCATIONS, 5000);
        conf.set(
                Constants.SHIFU_MODEL_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(source)
                        .makeQualified(new Path(super.getPathFinder().getModelConfigPath(source))).toString());
        conf.set(
                Constants.SHIFU_COLUMN_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(source)
                        .makeQualified(new Path(super.getPathFinder().getColumnConfigPath(source))).toString());
        conf.set(Constants.SHIFU_MODELSET_SOURCE_TYPE, source.toString());

        // too many data needed to be transfered to reducer, set default completed maps to a smaller one 0.7 to start
        // copy data in reducer earlier.
        conf.set("mapred.reduce.slowstart.completed.maps",
                Environment.getProperty("mapred.reduce.slowstart.completed.maps", "0.7"));

        String hdpVersion = HDPUtils.getHdpVersionForHDP224();
        if(StringUtils.isNotBlank(hdpVersion)) {
            // for hdp 2.2.4, hdp.version should be set and configuration files should be add to container class path
            conf.set("hdp.version", hdpVersion);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("hdfs-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("core-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("mapred-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("yarn-site.xml"), conf);
        }

        conf.setBoolean(CombineInputFormat.SHIFU_VS_SPLIT_COMBINABLE, true);
        conf.setBoolean("mapreduce.input.fileinputformat.input.dir.recursive", true);

        boolean isFastCorrelation = Environment.getProperty("shifu.correlation.fast", "false").equalsIgnoreCase(
                Boolean.TRUE.toString());

        int threads = parseThreadNum();
        conf.setInt("mapreduce.map.cpu.vcores", threads);

        // one can set guagua conf in shifuconfig
        for(Map.Entry<Object, Object> entry: Environment.getProperties().entrySet()) {
            if(CommonUtils.isHadoopConfigurationInjected(entry.getKey().toString())) {
                conf.set(entry.getKey().toString(), entry.getValue().toString());
            }
        }

        // if one of two memory settings is null, automatically set mapper memory by column size, if not set it from
        // system properties which is set from command line like 'shifu stats -c -Dmapreduce.map.memory.mb=3072
        // -Dmapreduce.map.java.opts=-Xmx3000M'
        if(System.getProperty("mapreduce.map.memory.mb") == null
                || System.getProperty("mapreduce.map.java.opts") == null) {
            setMapperMemory(conf, threads, isFastCorrelation);
        } else {
            conf.set("mapreduce.map.memory.mb", System.getProperty("mapreduce.map.memory.mb"));
            conf.set("mapreduce.map.java.opts", System.getProperty("mapreduce.map.java.opts"));
            log.info("Corrrelation map memory is set to {}MB from command line parameters.",
                    System.getProperty("mapreduce.map.memory.mb"));
        }

        @SuppressWarnings("deprecation")
        Job job = new Job(conf, "Shifu: Correlation Computing Job : " + this.modelConfig.getModelSetName());
        job.setJarByClass(getClass());

        if(isFastCorrelation) {
            job.setMapperClass(FastCorrelationMultithreadedMapper.class);
            FastCorrelationMultithreadedMapper.setMapperClass(job, FastCorrelationMapper.class);
            FastCorrelationMultithreadedMapper.setNumberOfThreads(job, threads);
        } else {
            job.setMapperClass(CorrelationMultithreadedMapper.class);
            CorrelationMultithreadedMapper.setMapperClass(job, CorrelationMapper.class);
            CorrelationMultithreadedMapper.setNumberOfThreads(job, threads);
        }

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(CorrelationWritable.class);

        job.setInputFormatClass(CombineInputFormat.class);
        FileInputFormat.setInputPaths(
                job,
                ShifuFileUtils.getFileSystemBySourceType(source).makeQualified(
                        new Path(super.modelConfig.getDataSetRawPath())));

        job.setReducerClass(CorrelationReducer.class);

        // 3000 features will be 30 reducers, 600 will be 6, much more reducer to avoid data all copied to one reducer
        // especially when features over 3000, each mapper output is 700M, 400 mapper will be 280G size
        job.setNumReduceTasks(this.columnConfigList.size() < 50 ? 2 : this.columnConfigList.size() / 50);

        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        String corrPath = super.getPathFinder().getCorrelationPath(source);
        FileOutputFormat.setOutputPath(job, new Path(corrPath));

        // clean output firstly
        ShifuFileUtils.deleteFile(corrPath, source);

        // submit job
        if(job.waitForCompletion(true)) {
            dumpCorrelationResult(source, corrPath);
        } else {
            throw new RuntimeException("MapReduce Correlation Computing Job failed.");
        }
    }

    /**
     * If 3000 * 3000 correlation computing, per default threads number setting, memory should be set according to
     * column size to avoid OOM issue.
     */
    private void setMapperMemory(Configuration conf, int threads, boolean isFastCorrelation) {
        int memoryBuffer = 1024;
        int memoryInContainer = 0;
        int youngMemory = 1024;
        int columnSize = this.columnConfigList.size();
        if(isFastCorrelation) {
            memoryInContainer += (1L * columnSize * columnSize * 8 * 6 * threads) / (1024 * 1024);
            if(columnSize > 4000) {
                memoryInContainer += 3072;
                youngMemory = 2500;
            }
        } else {
            // <1000 -> 2G; <=2000 2.5G; <=3000 3G; <=4000 4G; <=5000; 5G
            memoryInContainer = columnSize;
            if(memoryInContainer > 3000 && memoryInContainer <= 4000) {
                memoryInContainer = (int) (memoryInContainer * 1.4d);
                youngMemory = 2000;
            } else if(memoryInContainer > 4000 && memoryInContainer <= 5000) {
                memoryInContainer = (int) (memoryInContainer * 1.5d);
                youngMemory = 2500;
            } else if(memoryInContainer > 5000) {
                memoryInContainer = (int) (memoryInContainer * 1.6d);
                youngMemory = 3000;
            }
            if(memoryInContainer < 2048) {
                memoryInContainer = 2048; // at least 2048M
            }
        }

        memoryInContainer += memoryBuffer; // (MB, 1024 is buffer)

        log.info("Corrrelation map memory is set to {}MB.", memoryInContainer);

        if(youngMemory >= (memoryInContainer - memoryBuffer) / 2) {
            youngMemory = (memoryInContainer - memoryBuffer) / 3;
        }

        conf.set("mapreduce.map.memory.mb", memoryInContainer + "");
        conf.set(
                "mapreduce.map.java.opts",
                "-Xms"
                        + (memoryInContainer - memoryBuffer)
                        + "m -Xmx"
                        + (memoryInContainer - memoryBuffer)
                        + "m -Xmn"
                        + youngMemory
                        + "m -server -XX:MaxPermSize=128M -XX:PermSize=64M -XX:+UseParallelGC -XX:+UseParallelOldGC -XX:ParallelGCThreads=8 -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps ");
    }

    private int parseThreadNum() {
        int threads = 6;
        try {
            threads = Integer
                    .parseInt(Environment.getProperty(Constants.SHIFU_CORRELATION_MULTI_THREADS, threads + ""));
        } catch (Exception e) {
            log.warn("'shifu.correlation.multi.threads' should be a int value, set default value: {}", threads);
        }
        if(threads <= 0) {
            threads = 6;
        }
        return threads;
    }

    private void dumpCorrelationResult(SourceType source, String corrPath) throws IOException {
        String outputFilePattern = corrPath + Path.SEPARATOR + "part-*";
        if(!ShifuFileUtils.isFileExists(outputFilePattern, source)) {
            throw new RuntimeException("Correlation computing output file not exist.");
        }
        computeCorrValue(dumpCorrInfo(source, outputFilePattern));
    }

    /**
     * Compute correlation value according to correlation statistics from correlation MR job.
     * 
     * @param corrMap
     *            CorrelationWritable map read from MR job output file
     * @throws IOException
     *             any IOException to write correlation value to csv file.
     */
    private void computeCorrValue(SortedMap<Integer, CorrelationWritable> corrMap) throws IOException {
        boolean hasCandidates = CommonUtils.hasCandidateColumns(this.columnConfigList);
        String localCorrelationCsv = super.pathFinder.getLocalCorrelationCsvPath();
        ShifuFileUtils.createFileIfNotExists(localCorrelationCsv, SourceType.LOCAL);
        BufferedWriter writer = null;
        Map<Integer, double[]> finalCorrMap = new HashMap<Integer, double[]>();
        try {
            writer = ShifuFileUtils.getWriter(localCorrelationCsv, SourceType.LOCAL);
            writer.write(getColumnIndexes());
            writer.newLine();
            writer.write(getColumnNames());
            writer.newLine();

            for(Entry<Integer, CorrelationWritable> entry: corrMap.entrySet()) {
                ColumnConfig xColumnConfig = this.columnConfigList.get(entry.getKey());
                if(xColumnConfig.getColumnFlag() == ColumnFlag.Meta
                        || (hasCandidates && !ColumnFlag.Candidate.equals(xColumnConfig.getColumnFlag()))) {
                    continue;
                }
                CorrelationWritable xCw = corrMap.get(entry.getKey());
                double[] corrArray = new double[this.columnConfigList.size()];
                for(int i = 0; i < corrArray.length; i++) {
                    ColumnConfig yColumnConfig = this.columnConfigList.get(i);
                    if(yColumnConfig.getColumnFlag() == ColumnFlag.Meta) {
                        continue;
                    }
                    if(entry.getKey() > i) {
                        double[] reverseDoubleArray = finalCorrMap.get(i);
                        if(reverseDoubleArray != null) {
                            corrArray[i] = reverseDoubleArray[entry.getKey()];
                        } else {
                            corrArray[i] = 0d;
                        }
                        // not compute all, only up-right matrix are computed, such case, just get [i, j] from [j, i]
                        continue;
                    }

                    double numerator = xCw.getAdjustCount()[i] * xCw.getXySum()[i] - xCw.getAdjustSumX()[i]
                            * xCw.getAdjustSumY()[i];
                    double denominator1 = Math.sqrt(xCw.getAdjustCount()[i] * xCw.getXxSum()[i]
                            - xCw.getAdjustSumX()[i] * xCw.getAdjustSumX()[i]);
                    double denominator2 = Math.sqrt(xCw.getAdjustCount()[i] * xCw.getYySum()[i]
                            - xCw.getAdjustSumY()[i] * xCw.getAdjustSumY()[i]);
                    if(Double.compare(denominator1, Double.valueOf(0d)) == 0
                            || Double.compare(denominator2, Double.valueOf(0d)) == 0) {
                        corrArray[i] = 0d;
                    } else {
                        corrArray[i] = numerator / (denominator1 * denominator2);
                    }

                    // if(corrArray[i] > 1.0005d || (entry.getKey() == 54 && i == 2124)) {
                    if(corrArray[i] > 1.0005d) {
                        log.warn("Correlation value for columns {} {} > 1, below is debug info.", entry.getKey(), i);
                        log.warn("DEBUG: corr {}, value > 1d, numerator " + numerator + " denominator1 " + denominator1
                                + " denominator2 " + denominator2 + " {}, {}", numerator
                                / (denominator1 * denominator2), entry.getKey(), i);
                        log.warn(
                                "DEBUG: xCw.getAdjustCount()[i] * xCw.getXySum()[i] - xCw.getAdjustSumX()[i]  * xCw.getAdjustSumY()[i] : {} * {} - {} * {} ",
                                xCw.getAdjustCount()[i], xCw.getXySum()[i], xCw.getAdjustSumX()[i],
                                xCw.getAdjustSumY()[i]);
                        log.warn(
                                "DEBUG: xCw.getAdjustCount()[i] * xCw.getXxSum()[i] - xCw.getAdjustSumX()[i] * xCw.getAdjustSumX()[i] : {} * {} - {} * {} ",
                                xCw.getAdjustCount()[i], xCw.getXxSum()[i], xCw.getAdjustSumX()[i],
                                xCw.getAdjustSumX()[i]);
                        log.warn(
                                "DEBUG: xCw.getAdjustCount()[i] * xCw.getYySum()[i] - xCw.getAdjustSumY()[i] * xCw.getAdjustSumY()[i] : {} * {} -  {} * {} ",
                                xCw.getAdjustCount()[i], xCw.getYySum()[i], xCw.getAdjustSumY()[i],
                                xCw.getAdjustSumY()[i]);
                    }

                }

                // put to current map
                finalCorrMap.put(entry.getKey(), corrArray);

                // write to csv
                String corrStr = Arrays.toString(corrArray);
                String adjustCorrStr = corrStr.substring(1, corrStr.length() - 1);
                writer.write(entry.getKey() + "," + this.columnConfigList.get(entry.getKey()).getColumnName() + ","
                        + adjustCorrStr);
                writer.newLine();
            }
        } finally {
            IOUtils.closeQuietly(writer);
        }
    }

    /**
     * Dump {@link CorrelationWritable} from correlation MR job output file. This may need more memory if high column
     * number. Local memory should be set to 4G instead of 2G.
     * 
     * @param source
     *            source type
     * @param outputFilePattern
     *            output file pattern like part-*
     * @return Sorted map including CorrelationWritable info
     * @throws IOException
     *             any IO exception in reading output file
     * @throws UnsupportedEncodingException
     *             encoding exception to de-serialize correlation info in output file
     */
    private SortedMap<Integer, CorrelationWritable> dumpCorrInfo(SourceType source, String outputFilePattern)
            throws IOException, UnsupportedEncodingException {
        SortedMap<Integer, CorrelationWritable> corrMap = new TreeMap<Integer, CorrelationWritable>();
        FileStatus[] globStatus = ShifuFileUtils.getFileSystemBySourceType(source).globStatus(
                new Path(outputFilePattern));
        if(globStatus == null || globStatus.length == 0) {
            throw new RuntimeException("Correlation computing output file not exist.");
        }
        for(FileStatus fileStatus: globStatus) {
            List<Scanner> scanners = ShifuFileUtils.getDataScanners(fileStatus.getPath().toString(), source);
            for(Scanner scanner: scanners) {
                while(scanner.hasNext()) {
                    String str = scanner.nextLine().trim();
                    if(str.contains(Constants.TAB_STR)) {
                        String[] splits = str.split(Constants.TAB_STR);
                        String corrStr = splits[1];
                        int columnIndex = Integer.parseInt(splits[0].trim());
                        corrMap.put(columnIndex, bytesToObject(Base64.decodeBase64(corrStr.getBytes("utf-8"))));
                    }
                }
            }
            closeScanners(scanners);
        }
        return corrMap;
    }

    /**
     * De-serialize from bytes to object. One should provide the class name before de-serializing the object.
     * 
     * @param data
     *            byte array for deserialization
     * @return {@link CorrelationWritable} instance after deserialization
     * @throws NullPointerException
     *             if className or data is null.
     * @throws RuntimeException
     *             if any io exception or other reflection exception.
     */
    public CorrelationWritable bytesToObject(byte[] data) {
        if(data == null) {
            throw new NullPointerException(String.format(
                    "data and className should not be null. data:%s, className:%s", Arrays.toString(data)));
        }
        CorrelationWritable result = (CorrelationWritable) ReflectionUtils.newInstance(CorrelationWritable.class
                .getName());
        DataInputStream dataIn = null;
        try {
            ByteArrayInputStream in = new ByteArrayInputStream(data);
            dataIn = new DataInputStream(in);
            result.readFields(dataIn);
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            if(dataIn != null) {
                try {
                    dataIn.close();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }
        return result;
    }

    private String getColumnIndexes() {
        StringBuilder header = new StringBuilder("ColumnIndex,");
        for(ColumnConfig config: columnConfigList) {
            header.append(',').append(config.getColumnNum());
        }
        return header.toString();
    }

    private String getColumnNames() {
        StringBuilder header = new StringBuilder(",ColumnName");
        for(ColumnConfig config: columnConfigList) {
            header.append(',').append(config.getColumnName());
        }
        return header.toString();
    }

    private void doReBin(ColumnConfig columnConfig) throws IOException {
        int expectBinNum = getIntParam(this.params, Constants.EXPECTED_BIN_NUM);
        double ivKeepRatio = getDoubleParam(this.params, Constants.IV_KEEP_RATIO, 1.0d);
        long minimumInstCnt = getLongParam(this.params, Constants.MINIMUM_BIN_INST_CNT);

        ColumnConfigDynamicBinning columnConfigDynamicBinning = new ColumnConfigDynamicBinning(columnConfig,
                expectBinNum, ivKeepRatio, minimumInstCnt);

        List<AbstractBinInfo> binInfos = columnConfigDynamicBinning.run();

        long[] binCountNeg = new long[binInfos.size() + 1];
        long[] binCountPos = new long[binInfos.size() + 1];
        for(int i = 0; i < binInfos.size(); i++) {
            AbstractBinInfo binInfo = binInfos.get(i);
            binCountNeg[i] = binInfo.getNegativeCnt();
            binCountPos[i] = binInfo.getPositiveCnt();
        }
        binCountNeg[binCountNeg.length - 1] = columnConfig.getBinCountNeg().get(
                columnConfig.getBinCountNeg().size() - 1);
        binCountPos[binCountPos.length - 1] = columnConfig.getBinCountPos().get(
                columnConfig.getBinCountPos().size() - 1);

        double[] binWeightNeg = new double[binInfos.size() + 1];
        double[] binWeightPos = new double[binInfos.size() + 1];
        for(int i = 0; i < binInfos.size(); i++) {
            AbstractBinInfo binInfo = binInfos.get(i);
            binWeightNeg[i] = binInfo.getWeightNeg();
            binWeightPos[i] = binInfo.getWeightPos();
        }

        binWeightNeg[binWeightNeg.length - 1] = columnConfig.getBinWeightedNeg().get(
                columnConfig.getBinWeightedNeg().size() - 1);
        binWeightPos[binWeightPos.length - 1] = columnConfig.getBinWeightedPos().get(
                columnConfig.getBinWeightedPos().size() - 1);

        ColumnStatsCalculator.ColumnMetrics columnCountMetrics = ColumnStatsCalculator.calculateColumnMetrics(
                binCountNeg, binCountPos);
        ColumnStatsCalculator.ColumnMetrics columnWeightMetrics = ColumnStatsCalculator.calculateColumnMetrics(
                binWeightNeg, binWeightPos);

        columnConfig.setBinLength(binInfos.size() + 1);
        if(columnConfig.isCategorical()) {
            List<String> values = new ArrayList<String>();
            for(AbstractBinInfo binInfo: binInfos) {
                CategoricalBinInfo categoricalBinInfo = (CategoricalBinInfo) binInfo;
                values.add(StringUtils.join(categoricalBinInfo.getValues(), Constants.CATEGORICAL_GROUP_VAL_DELIMITER));
            }
            columnConfig.setBinCategory(values);
        } else {
            List<Double> values = new ArrayList<Double>();
            for(AbstractBinInfo binInfo: binInfos) {
                NumericalBinInfo numericalBinInfo = (NumericalBinInfo) binInfo;
                values.add(numericalBinInfo.getLeftThreshold());
            }
            columnConfig.setBinBoundary(values);
        }
        columnConfig.setBinCountNeg(convertToIntList(binCountNeg));
        columnConfig.setBinCountPos(convertToIntList(binCountPos));

        List<Double> binPosRates = new ArrayList<Double>();
        for(AbstractBinInfo binInfo: binInfos) {
            binPosRates.add(binInfo.getPositiveRate());
        }
        // fix a bug here to not add missing bin pos rate at last one of binPosRate
        if(binPosRates.size() + 1 == binCountPos.length) {
            long missingSumCnt = binCountPos[binCountPos.length - 1] + binCountNeg[binCountNeg.length - 1];
            if(missingSumCnt > 0) {
                binPosRates.add(binCountPos[binCountPos.length - 1] * 1d / missingSumCnt);
            } else {
                binPosRates.add(Double.NaN);
            }
        }

        columnConfig.setBinPosCaseRate(binPosRates);

        columnConfig.setBinWeightedNeg(convertIntoDoubleList(binWeightNeg));
        columnConfig.setBinWeightedPos(convertIntoDoubleList(binWeightPos));

        columnConfig.setIv(columnCountMetrics.getIv());
        columnConfig.setKs(columnCountMetrics.getKs());
        columnConfig.getColumnStats().setWoe(columnCountMetrics.getWoe());
        columnConfig.getColumnBinning().setBinCountWoe(columnCountMetrics.getBinningWoe());

        columnConfig.getColumnStats().setWeightedIv(columnWeightMetrics.getIv());
        columnConfig.getColumnStats().setWeightedKs(columnWeightMetrics.getWoe());
        columnConfig.getColumnStats().setWeightedWoe(columnWeightMetrics.getWoe());
        columnConfig.getColumnBinning().setBinWeightedWoe(columnWeightMetrics.getBinningWoe());
    }

    private List<Double> convertIntoDoubleList(double[] binWeights) {
        List<Double> doubleList = new ArrayList<Double>(binWeights.length);
        for(double weight: binWeights) {
            doubleList.add(weight);
        }
        return doubleList;
    }

    private List<Integer> convertToIntList(long[] binCounts) {
        List<Integer> binCountList = new ArrayList<Integer>(binCounts.length);
        for(long count: binCounts) {
            binCountList.add((int) count);
        }
        return binCountList;
    }

}
