/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.processor.stats;

import static ml.shifu.shifu.util.Constants.LOCAL_DATE_STATS_CSV_FILE_NAME;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.zip.GZIPInputStream;

import org.apache.commons.codec.binary.Base64;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.Predicate;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.io.IOUtils;
import org.apache.commons.jexl2.JexlException;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
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

import com.clearspring.analytics.stream.cardinality.HyperLogLogPlus;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Splitter;
import com.google.common.collect.Lists;

import ml.shifu.guagua.hadoop.util.HDPUtils;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceConstants;
import ml.shifu.guagua.util.FileUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnFlag;
import ml.shifu.shifu.container.obj.ColumnType;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.binning.BinningInfoWritable;
import ml.shifu.shifu.core.binning.UpdateBinningInfoMapper;
import ml.shifu.shifu.core.binning.UpdateBinningInfoReducer;
import ml.shifu.shifu.core.datestat.DateStatComputeMapper;
import ml.shifu.shifu.core.datestat.DateStatComputeReducer;
import ml.shifu.shifu.core.datestat.DateStatInfoWritable;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.core.mr.input.CombineInputFormat;
import ml.shifu.shifu.core.processor.BasicModelProcessor;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.fs.SourceFile;
import ml.shifu.shifu.pig.PigExecutor;
import ml.shifu.shifu.udf.CalculateStatsUDF;
import ml.shifu.shifu.util.Base64Utils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.JSONUtils;
import ml.shifu.shifu.util.NumberUtils;
import ml.shifu.shifu.util.ValueVisitor;

/**
 * Created by zhanhu on 6/30/16.
 */
public class MapReducerStatsWorker extends AbstractStatsExecutor {

    private static Logger LOG = LoggerFactory.getLogger(MapReducerStatsWorker.class);

    public static final Long MINIMUM_DISTINCT_CNT = 2L;
    public static final Long MAXIMUM_DISTINCT_CNT = 1000L;

    protected PathFinder pathFinder = null;

    protected boolean isUpdateStatsOnly;

    public MapReducerStatsWorker(BasicModelProcessor processor, ModelConfig modelConfig,
            List<ColumnConfig> columnConfigList, boolean isUpdateStatsOnly) {
        super(processor, modelConfig, columnConfigList);
        this.isUpdateStatsOnly = isUpdateStatsOnly;
        pathFinder = processor.getPathFinder();
    }

    @Override
    public boolean doStats() throws Exception {
        LOG.info("delete historical pre-train data");
        if(this.modelConfig.isMultiTask()) {
            ShifuFileUtils.deleteFile(pathFinder.getPreTrainingStatsPath(this.getMtlIndex()),
                    modelConfig.getDataSet().getSource());
        } else {
            ShifuFileUtils.deleteFile(pathFinder.getPreTrainingStatsPath(), modelConfig.getDataSet().getSource());
        }
        Map<String, String> paramsMap = new HashMap<String, String>();
        paramsMap.put("delimiter", CommonUtils.escapePigString(modelConfig.getDataSetDelimiter()));
        int columnParallel = getColumnParallelValue();
        paramsMap.put("column_parallel", Integer.toString(columnParallel));

        paramsMap.put("histo_scale_factor", Environment.getProperty("shifu.stats.histo.scale.factor", "100"));

        try {
            runStatsPig(paramsMap);
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_RUNNING_PIG_JOB, e);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }

        // sync Down
        LOG.info("Updating ColumnConfig with stats...");
        // update column config
        updateColumnConfigWithPreTrainingStats();

        // check categorical columns and numerical columns and warning
        checkNumericalAndCategoricalColumns();

        // save it to local/hdfs
        if(this.isUpdateStatsOnly) {
            LOG.info("Update stats only in ColumnConfig.json.new local file.");
            JSONUtils.writeValue(new File(pathFinder.getColumnConfigPath(SourceType.LOCAL) + ".new"),
                    this.columnConfigList);
        } else {
            processor.saveColumnConfigList();
            processor.syncDataToHdfs(modelConfig.getDataSet().getSource());

            runPSI();

            if(StringUtils.isNotBlank(modelConfig.getDataSet().getDateColumnName())) {
                // run, only when the date column available
                updateDateStatWithMRJob();
            }
        }

        return true;
    }

    /**
     * Logic to tune # of reducers in 1st MR job.
     */
    private int getColumnParallelValue() throws IOException {
        int columnParallel = 0;
        if(columnConfigList.size() <= 100) {
            columnParallel = columnConfigList.size() * 2;
        } else if(columnConfigList.size() <= 500) {
            columnParallel = columnConfigList.size();
        } else if(columnConfigList.size() <= 1000) {
            // 1000 => 200 reducers
            columnParallel = columnConfigList.size() / 2;
        } else if(columnConfigList.size() > 1000 && columnConfigList.size() <= 2000) {
            // 2000 => 320 reducers
            columnParallel = columnConfigList.size() / 4;
        } else if(columnConfigList.size() > 2000 && columnConfigList.size() <= 3000) {
            // 3000 => 420 reducers
            columnParallel = columnConfigList.size() / 6;
        } else if(columnConfigList.size() > 3000 && columnConfigList.size() <= 4000) {
            // 4000 => 500
            columnParallel = columnConfigList.size() / 8;
        } else {
            // 5000 => 500
            columnParallel = columnConfigList.size() / 10;
        }
        // limit max reducer to 999
        int parallelNumbByVolume = getParallelNumByDataVolume();
        columnParallel = Math.min(columnParallel, parallelNumbByVolume);
        columnParallel = columnParallel > 999 ? 999 : columnParallel;
        return columnParallel;
    }

    private int getParallelNumByDataVolume() throws IOException {
        long fileSize = ShifuFileUtils.getFileOrDirectorySize(modelConfig.getDataSet().getDataPath(),
                modelConfig.getDataSet().getSource());
        LOG.info("File Size is - {}, for {}", fileSize, modelConfig.getDataSet().getDataPath());
        if(ShifuFileUtils.isCompressedFileOrDirectory(modelConfig.getDataSet().getDataPath(),
                modelConfig.getDataSet().getSource())) {
            LOG.info("File is compressed, for {}", modelConfig.getDataSet().getDataPath());
            fileSize = fileSize * 3; // multi 3 times, if the file is compressed
        }
        return (int) (fileSize / (256 * 1024 * 1024l)); // each reducer handle 256MB data
    }

    /**
     * According to sample values and distinct count in each column, check if user set wrong for numerical and
     * categorical features. Only warning message are output to console for user to check.
     */
    private void checkNumericalAndCategoricalColumns() {
        for(ColumnConfig config: this.columnConfigList) {
            if(config != null && !config.isMeta() && !config.isTarget()) {
                List<String> sampleValues = config.getSampleValues();
                if(config.isNumerical() && sampleValues != null) {
                    int nums = numberCount(sampleValues);
                    if((nums * 1d / sampleValues.size()) < 0.5d) {
                        LOG.warn(
                                "Column {} with index {} is set to numrical but numbers are less than 50% in ColumnConfig::SampleValues, please check if it is numerical feature.",
                                config.getColumnName(), config.getColumnNum());
                    }
                }

                if(config.isCategorical() && sampleValues != null) {
                    int nums = numberCount(sampleValues);
                    if((nums * 1d / sampleValues.size()) > 0.95d && config.getColumnStats().getDistinctCount() != null
                            && config.getColumnStats().getDistinctCount() > 5000) {
                        LOG.warn(
                                "Column {} with index {} is set to categorical but numbers are more than 95% in ColumnConfig::SampleValues and distinct count is over 5000, please check if it is categorical feature.",
                                config.getColumnName(), config.getColumnNum());
                    }
                }
            }
        }
    }

    private int numberCount(List<String> sampleValues) {
        int numbers = 0;
        for(String str: sampleValues) {
            try {
                Double.parseDouble(str);
                numbers += 1;
            } catch (Exception ignore) {
            }
        }
        return numbers;
    }

    protected void runStatsPig(Map<String, String> paramsMap) throws Exception {
        if(!this.isUpdateStatsOnly) {
            paramsMap.put("group_binning_parallel", Integer.toString(columnConfigList.size() / (5 * 8)));

            if(this.modelConfig.isMultiTask()) {
                ShifuFileUtils.deleteFile(
                        pathFinder.getUpdatedBinningInfoPath(modelConfig.getDataSet().getSource(), this.getMtlIndex()),
                        modelConfig.getDataSet().getSource());
                paramsMap.put(CommonConstants.MTL_INDEX, this.getMtlIndex() + "");
            } else {
                ShifuFileUtils.deleteFile(pathFinder.getUpdatedBinningInfoPath(modelConfig.getDataSet().getSource()),
                        modelConfig.getDataSet().getSource());
            }

            LOG.debug("this.pathFinder.getOtherConfigs() => " + this.pathFinder.getOtherConfigs());
            PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getScriptPath("scripts/StatsSpdtI.pig"),
                    paramsMap, modelConfig.getDataSet().getSource(), this.pathFinder);
        }
        // update
        LOG.info("Updating binning info ...");
        updateBinningInfoWithMRJob();
    }

    protected void updateDateStatWithMRJob() throws IOException, InterruptedException, ClassNotFoundException {
        if(StringUtils.isEmpty(this.modelConfig.getDateColumnName())) {
            LOG.info("ModelConfig#dataSet#dateColumnName is not set, skip updateDateStatWithMRJob.");
            return;
        }

        RawSourceData.SourceType source = this.modelConfig.getDataSet().getSource();

        Configuration conf = new Configuration();
        prepareJobConf(source, conf, null);

        @SuppressWarnings("deprecation")
        Job job = new Job(conf, "Shifu: Date Stats Job : " + this.modelConfig.getModelSetName());
        job.setJarByClass(getClass());
        job.setMapperClass(DateStatComputeMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(DateStatInfoWritable.class);
        job.setInputFormatClass(CombineInputFormat.class);
        Path filePath = new Path(super.modelConfig.getDataSetRawPath());
        FileInputFormat.setInputPaths(job,
                ShifuFileUtils.getFileSystemBySourceType(source, filePath).makeQualified(filePath));

        job.setReducerClass(DateStatComputeReducer.class);

        int mapperSize = new CombineInputFormat().getSplits(job).size();
        LOG.info("DEBUG: Test mapper size is {} ", mapperSize);
        Integer reducerSize = Environment.getInt(CommonConstants.SHIFU_DAILYSTAT_REDUCER);
        if(reducerSize != null) {
            job.setNumReduceTasks(Environment.getInt(CommonConstants.SHIFU_DAILYSTAT_REDUCER, 20));
        } else {
            // By average, each reducer handle 100 variables
            int newReducerSize = (this.columnConfigList.size() / 100) + 1;
            LOG.info("Adjust date stat info reducer size to {} ", newReducerSize);
            job.setNumReduceTasks(newReducerSize);
        }
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        String preTrainingInfo = this.pathFinder.getPreTrainingStatsPath(source);
        Path path = new Path(preTrainingInfo);
        LOG.info("Output path:" + path);
        FileOutputFormat.setOutputPath(job, path);

        // clean output firstly
        ShifuFileUtils.deleteFile(preTrainingInfo, source);

        // submit job
        if(!job.waitForCompletion(true)) {
            throw new RuntimeException("MapReduce Job Updating date stat Info failed.");
        } else {
            long totalValidCount = job.getCounters().findCounter(Constants.SHIFU_GROUP_COUNTER, "TOTAL_VALID_COUNT")
                    .getValue();
            long invalidTagCount = job.getCounters().findCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG")
                    .getValue();
            long filterOut = job.getCounters().findCounter(Constants.SHIFU_GROUP_COUNTER, "FILTER_OUT_COUNT")
                    .getValue();
            long weightExceptions = job.getCounters().findCounter(Constants.SHIFU_GROUP_COUNTER, "WEIGHT_EXCEPTION")
                    .getValue();
            LOG.info(
                    "Total valid records {}, invalid tag records {}, filter out records {}, weight exception records {}",
                    totalValidCount, invalidTagCount, filterOut, weightExceptions);

            if(totalValidCount > 0L && invalidTagCount * 1d / totalValidCount >= 0.8d) {
                LOG.warn("Too many invalid tags, please check you configuration on positive tags and negative tags.");
            }
            copyFileToLocal(conf, path);
        }
    }

    private void copyFileToLocal(Configuration conf, Path path) throws IOException {
        FileSystem hdfs = FileSystem.get(conf);
        RemoteIterator<LocatedFileStatus> locatedFileStatusRemoteIterator = hdfs.listFiles(path, false);
        List<Path> list = new ArrayList<>();
        while(locatedFileStatusRemoteIterator.hasNext()) {
            LocatedFileStatus next = locatedFileStatusRemoteIterator.next();
            Path p = next.getPath();
            if(p.getName().endsWith("gz")) {
                list.add(p);
            }
        }
        Collections.sort(list, new Comparator<Path>() {

            private Integer getDigitInPath(String path) {
                String resultStr = StringUtils.substringBefore(StringUtils.substringAfterLast(path, "-"), ".");
                if(StringUtils.isEmpty(resultStr)) {
                    return 0;
                }
                return Integer.parseInt(resultStr);
            }

            @Override
            public int compare(Path o1, Path o2) {
                return getDigitInPath(o1.getName()) - getDigitInPath(o2.getName());
            }
        });
        String dateStatsOutputFileName = this.modelConfig.getStats().getDateStatsOutputFileName();
        File file = new File(StringUtils.isEmpty(dateStatsOutputFileName) ? LOCAL_DATE_STATS_CSV_FILE_NAME
                : dateStatsOutputFileName);
        OutputStream out = org.apache.commons.io.FileUtils.openOutputStream(file);
        // add title in csv file
        IOUtils.write(
                "variable name|date|column type|max|min|mean|median value|count|missing count|standard deviation|missing ratio|WOE|KS|IV|weighted WOE|weighted KS|weighted IV|skewness|kurtosis|cardinality|P25th|P75th\n",
                out);
        for(Path p: list) {
            FSDataInputStream in = hdfs.open(p);
            GZIPInputStream gzin = new GZIPInputStream(in);
            IOUtils.copy(gzin, out);
            IOUtils.closeQuietly(gzin);
        }
        IOUtils.closeQuietly(out);
        LOG.info("Copy file to local:" + file.getAbsolutePath());
    }

    protected void updateBinningInfoWithMRJob() throws IOException, InterruptedException, ClassNotFoundException {
        RawSourceData.SourceType source = this.modelConfig.getDataSet().getSource();

        String filePath = Constants.BINNING_INFO_FILE_NAME;
        File binInfoFile = null;

        BufferedWriter writer = null;
        List<Scanner> scanners = null;
        try {
            if(this.modelConfig.isMultiTask()) {
                scanners = ShifuFileUtils
                        .getDataScanners(pathFinder.getUpdatedBinningInfoPath(source, this.getMtlIndex()), source);
                filePath = Constants.BINNING_INFO_FILE_NAME + "." + this.getMtlIndex();
            } else {
                scanners = ShifuFileUtils.getDataScanners(pathFinder.getUpdatedBinningInfoPath(source), source);
                filePath = Constants.BINNING_INFO_FILE_NAME;
            }
            binInfoFile = new File(filePath);
            writer = new BufferedWriter(
                    new OutputStreamWriter(new FileOutputStream(binInfoFile), Charset.forName("UTF-8")));
            for(Scanner scanner: scanners) {
                while(scanner.hasNextLine()) {
                    String line = scanner.nextLine();
                    writer.write(line + "\n");
                }
            }
        } finally {
            // release
            processor.closeScanners(scanners);
            IOUtils.closeQuietly(writer);
        }

        Configuration conf = new Configuration();
        prepareJobConf(source, conf, binInfoFile.toString());
        conf.set(CommonConstants.MTL_INDEX, this.getMtlIndex() + "");

        @SuppressWarnings("deprecation")
        Job job = new Job(conf, "Shifu: Stats Updating Binning Job : " + this.modelConfig.getModelSetName());
        job.setJarByClass(getClass());
        job.setMapperClass(UpdateBinningInfoMapper.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(BinningInfoWritable.class);
        job.setInputFormatClass(CombineInputFormat.class);
        Path rawDataPath = new Path(super.modelConfig.getDataSetRawPath());
        FileInputFormat.setInputPaths(job,
                ShifuFileUtils.getFileSystemBySourceType(source, rawDataPath).makeQualified(rawDataPath));

        job.setReducerClass(UpdateBinningInfoReducer.class);

        int mapperSize = new CombineInputFormat().getSplits(job).size();
        LOG.info("Mapper size is {} ", mapperSize);
        Integer reducerSize = Environment.getInt(CommonConstants.SHIFU_UPDATEBINNING_REDUCER);
        if(reducerSize != null) {
            job.setNumReduceTasks(Environment.getInt(CommonConstants.SHIFU_UPDATEBINNING_REDUCER, 20));
        } else {
            // By average, each reducer handle 100 variables
            int newReducerSize = (this.columnConfigList.size() / 100) + 1;
            LOG.info("Adjust updating binning info reducer size to {} ", newReducerSize);
            job.setNumReduceTasks(newReducerSize);
        }
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        String preTrainingInfo;
        if(this.modelConfig.isMultiTask()) {
            preTrainingInfo = this.pathFinder.getPreTrainingStatsPath(source, this.getMtlIndex());
            FileOutputFormat.setOutputPath(job, new Path(preTrainingInfo));
        } else {
            preTrainingInfo = this.pathFinder.getPreTrainingStatsPath(source);
            FileOutputFormat.setOutputPath(job, new Path(preTrainingInfo));
        }

        // clean output firstly
        ShifuFileUtils.deleteFile(preTrainingInfo, source);

        // submit job
        if(!job.waitForCompletion(true)) {
            FileUtils.deleteQuietly(new File(filePath));
            throw new RuntimeException("MapReduce Job Updateing Binning Info failed.");
        } else {
            long totalValidCount = job.getCounters().findCounter(Constants.SHIFU_GROUP_COUNTER, "TOTAL_VALID_COUNT")
                    .getValue();
            long invalidTagCount = job.getCounters().findCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG")
                    .getValue();
            long filterOut = job.getCounters().findCounter(Constants.SHIFU_GROUP_COUNTER, "FILTER_OUT_COUNT")
                    .getValue();
            long weightExceptions = job.getCounters().findCounter(Constants.SHIFU_GROUP_COUNTER, "WEIGHT_EXCEPTION")
                    .getValue();
            LOG.info(
                    "Total valid records {}, invalid tag records {}, filter out records {}, weight exception records {}",
                    totalValidCount, invalidTagCount, filterOut, weightExceptions);

            if(totalValidCount > 0L && invalidTagCount * 1d / totalValidCount >= 0.8d) {
                LOG.warn("Too many invalid tags, please check you configuration on positive tags and negative tags.");
            }
        }
        FileUtils.deleteQuietly(new File(filePath));
    }

    private void prepareJobConf(RawSourceData.SourceType source, final Configuration conf, String filePath)
            throws IOException {
        // add jars to hadoop mapper and reducer
        if(StringUtils.isNotEmpty(filePath) && !isUpdateStatsOnly) {
            new GenericOptionsParser(conf, new String[] { "-libjars", addRuntimeJars(), "-files", filePath });
        } else {
            new GenericOptionsParser(conf, new String[] { "-libjars", addRuntimeJars() });
        }

        conf.setBoolean(CombineInputFormat.SHIFU_VS_SPLIT_COMBINABLE, true);
        conf.setBoolean("mapreduce.input.fileinputformat.input.dir.recursive", true);
        conf.setBoolean(Constants.IS_UPDATE_STATS_ONLY, this.isUpdateStatsOnly);

        conf.set(Constants.SHIFU_STATS_EXLCUDE_MISSING,
                Environment.getProperty(Constants.SHIFU_STATS_EXLCUDE_MISSING, "true"));

        conf.setBoolean(GuaguaMapReduceConstants.MAPRED_MAP_TASKS_SPECULATIVE_EXECUTION, true);
        conf.setBoolean(GuaguaMapReduceConstants.MAPRED_REDUCE_TASKS_SPECULATIVE_EXECUTION, true);
        conf.setBoolean(GuaguaMapReduceConstants.MAPREDUCE_MAP_SPECULATIVE, true);
        conf.setBoolean(GuaguaMapReduceConstants.MAPREDUCE_REDUCE_SPECULATIVE, true);
        Path modelConfPath = new Path(this.pathFinder.getModelConfigPath(source));
        conf.set(Constants.SHIFU_MODEL_CONFIG, ShifuFileUtils.getFileSystemBySourceType(source, modelConfPath)
                .makeQualified(modelConfPath).toString());
        Path columnConfPath = new Path(this.pathFinder.getColumnConfigPath(source));
        conf.set(Constants.SHIFU_COLUMN_CONFIG, ShifuFileUtils.getFileSystemBySourceType(source, columnConfPath)
                .makeQualified(columnConfPath).toString());
        conf.set(NNConstants.MAPRED_JOB_QUEUE_NAME, Environment.getProperty(Environment.HADOOP_JOB_QUEUE, "default"));
        conf.set(Constants.SHIFU_MODELSET_SOURCE_TYPE, source.toString());

        // set mapreduce.job.max.split.locations to 30 to suppress warnings
        conf.setInt(GuaguaMapReduceConstants.MAPREDUCE_JOB_MAX_SPLIT_LOCATIONS, 5000);
        conf.set("mapred.reduce.slowstart.completed.maps",
                Environment.getProperty("mapred.reduce.slowstart.completed.maps", "0.8"));

        conf.set(Constants.SHIFU_STATS_FILTER_EXPRESSIONS, super.modelConfig.getSegmentFilterExpressionsAsString());
        LOG.info("segment expressions is {}", super.modelConfig.getSegmentFilterExpressionsAsString());

        String hdpVersion = HDPUtils.getHdpVersionForHDP224();
        if(StringUtils.isNotBlank(hdpVersion)) {
            // for hdp 2.2.4, hdp.version should be set and configuration files should be add to container class path
            conf.set("hdp.version", hdpVersion);
        }

        // one can set guagua conf in shifuconfig
        CommonUtils.injectHadoopShifuEnvironments(new ValueVisitor() {
            @Override
            public void inject(Object key, Object value) {
                conf.set(key.toString(), value.toString());
            }
        });
    }

    // GuaguaOptionsParser doesn't to support *.jar currently.
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
        // shifu-*.jar
        jars.add(JarManager.findContainingJar(getClass()));
        // jexl
        jars.add(JarManager.findContainingJar(JexlException.class));
        // encog-core-*.jar
        jars.add(JarManager.findContainingJar(MLDataSet.class));
        // jackson-databind-*.jar
        jars.add(JarManager.findContainingJar(ObjectMapper.class));
        // jackson-core-*.jar
        jars.add(JarManager.findContainingJar(JsonParser.class));
        // jackson-annotations-*.jar
        jars.add(JarManager.findContainingJar(JsonIgnore.class));
        // stream-llib-*.jar
        jars.add(JarManager.findContainingJar(HyperLogLogPlus.class));

        return StringUtils.join(jars, NNConstants.LIB_JAR_SEPARATOR);
    }

    /**
     * update the max/min/mean/std/binning information from stats step
     * 
     * @throws IOException
     *             in stats processing from hdfs files
     */
    public void updateColumnConfigWithPreTrainingStats() throws IOException {
        List<Scanner> scanners;
        if(this.modelConfig.isMultiTask()) {
            scanners = ShifuFileUtils.getDataScanners(pathFinder.getPreTrainingStatsPath(this.getMtlIndex()),
                    modelConfig.getDataSet().getSource());
        } else {
            scanners = ShifuFileUtils.getDataScanners(pathFinder.getPreTrainingStatsPath(),
                    modelConfig.getDataSet().getSource());
        }
        int initSize = columnConfigList.size();
        for(Scanner scanner: scanners) {
            scanStatsResult(scanner, initSize);
        }

        // release
        processor.closeScanners(scanners);

        Collections.sort(this.columnConfigList, new Comparator<ColumnConfig>() {
            @Override
            public int compare(ColumnConfig o1, ColumnConfig o2) {
                return o1.getColumnNum().compareTo(o2.getColumnNum());
            }
        });
    }

    private Set<String> getColumnNames(List<ColumnConfig> columnConfigs) {
        Set<String> columnNames = new HashSet<>();
        if (columnConfigs == null) {
            return columnNames;
        }
        for (ColumnConfig columnConfig : columnConfigs) {
            columnNames.add(columnConfig.getColumnName());
        }
        return columnNames;
    }

    /**
     * Scan the stats result and save them into column configure
     * 
     * @param scanner
     *            the scanners to be read
     */
    private void scanStatsResult(Scanner scanner, int ccInitSize) {
        Set<String> columnNames = getColumnNames(this.columnConfigList);
        while(scanner.hasNextLine()) {
            String[] raw = scanner.nextLine().trim().split("\\|");

            if(raw.length == 1) {
                continue;
            }

            if(raw.length < 25) {
                LOG.info("The stats data has " + raw.length + " fields.");
                LOG.info("The stats data is - " + Arrays.toString(raw));
            }

            int columnNum = Integer.parseInt(raw[0]);

            int corrColumnNum = columnNum;
            if(columnNum >= ccInitSize) {
                corrColumnNum = columnNum % ccInitSize;
            }
            try {
                ColumnConfig basicConfig = this.columnConfigList.get(corrColumnNum);
                LOG.debug("basicConfig is - " + basicConfig.getColumnName() + " corrColumnNum:" + corrColumnNum);

                ColumnConfig config = null;
                if(columnNum >= ccInitSize) {
                    config = new ColumnConfig();
                    config.setColumnNum(columnNum);
                    config.setVersion(basicConfig.getVersion());
                    config.setColumnType(basicConfig.getColumnType());
                    config.setColumnFlag(basicConfig.getColumnFlag() == ColumnFlag.Target ? ColumnFlag.Meta
                            : basicConfig.getColumnFlag());
                    config.setSegment(true);
                    // If we have 30 features and column number is 31, we got column name "columnname_seg1"
                    String columnName = basicConfig.getColumnName() + "_seg" + (columnNum / ccInitSize);
                    if (columnNames.contains(columnName)) {
                        // If "columnname_seg1" exists, we will find a unique one.
                        columnName = CommonUtils.getUniqueName(columnNames, columnName);
                    }
                    config.setColumnName(columnName);
                    columnNames.add(columnName);

                    LOG.debug("basicConfig is - " + basicConfig.getColumnName() + " corrColumnNum:" + corrColumnNum
                            + ", currColumnName: " + columnNum + ", currColumnType:" + config.getColumnType());

                    this.columnConfigList.add(config);
                } else {
                    config = basicConfig;
                }

                if(config.isHybrid()) {
                    String[] splits = CommonUtils.split(raw[1], Constants.HYBRID_BIN_STR_DILIMETER);
                    config.setBinBoundary(CommonUtils.stringToDoubleList(splits[0]));
                    String binCategory = Base64Utils.base64Decode(splits[1]);
                    config.setBinCategory(
                            CommonUtils.stringToStringList(binCategory, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));
                } else if(config.isCategorical()) {
                    String binCategory = Base64Utils.base64Decode(raw[1]);
                    config.setBinCategory(
                            CommonUtils.stringToStringList(binCategory, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));
                    config.setBinBoundary(null);
                } else {
                    config.setBinBoundary(CommonUtils.stringToDoubleList(raw[1]));
                    config.setBinCategory(null);
                }
                config.setBinCountNeg(CommonUtils.stringToIntegerList(raw[2]));
                config.setBinCountPos(CommonUtils.stringToIntegerList(raw[3]));
                // config.setBinAvgScore(CommonUtils.stringToIntegerList(raw[4]));
                config.setBinPosCaseRate(CommonUtils.stringToDoubleList(raw[5]));
                config.setBinLength(config.getBinCountNeg().size());
                config.setKs(NumberUtils.parseDouble(raw[6]));
                config.setIv(NumberUtils.parseDouble(raw[7]));
                config.setMax(NumberUtils.parseDouble(raw[8]));
                config.setMin(NumberUtils.parseDouble(raw[9]));
                config.setMean(NumberUtils.parseDouble(raw[10]));
                config.setStdDev(NumberUtils.parseDouble(raw[11], Double.NaN));

                // magic?
                config.setColumnType(ColumnType.of(raw[12]));

                config.setMedian(NumberUtils.parseDouble(raw[13]));

                config.setMissingCnt(NumberUtils.parseLong(raw[14]));
                config.setTotalCount(NumberUtils.parseLong(raw[15]));
                config.setMissingPercentage(NumberUtils.parseDouble(raw[16]));

                config.setBinWeightedNeg(CommonUtils.stringToDoubleList(raw[17]));
                config.setBinWeightedPos(CommonUtils.stringToDoubleList(raw[18]));
                config.getColumnStats().setWoe(NumberUtils.parseDouble(raw[19]));
                config.getColumnStats().setWeightedWoe(NumberUtils.parseDouble(raw[20]));
                config.getColumnStats().setWeightedKs(NumberUtils.parseDouble(raw[21]));
                config.getColumnStats().setWeightedIv(NumberUtils.parseDouble(raw[22]));
                config.getColumnBinning().setBinCountWoe(CommonUtils.stringToDoubleList(raw[23]));
                config.getColumnBinning().setBinWeightedWoe(CommonUtils.stringToDoubleList(raw[24]));
                // TODO magic code?
                if(raw.length >= 26) {
                    config.getColumnStats().setSkewness(NumberUtils.parseDouble(raw[25]));
                }
                if(raw.length >= 27) {
                    config.getColumnStats().setKurtosis(NumberUtils.parseDouble(raw[26]));
                }
                if(raw.length >= 30) {
                    config.getColumnStats().setValidNumCount(NumberUtils.parseLong(raw[29]));
                }
                if(raw.length >= 31) {
                    config.getColumnStats().setDistinctCount(NumberUtils.parseLong(raw[30]));
                }
                if(raw.length >= 32) {
                    if(raw[31] != null) {
                        List<String> sampleValues = Arrays.asList(Base64Utils.base64Decode(raw[31]).split(","));
                        config.setSampleValues(sampleValues);
                    }
                }
                if(raw.length >= 33) {
                    config.getColumnStats().set25th(NumberUtils.parseDouble(raw[32]));
                }
                if(raw.length >= 34) {
                    config.getColumnStats().set75th(NumberUtils.parseDouble(raw[33]));
                }
                if(raw.length >= 35) {
                    config.setHashSeed((int) NumberUtils.parseLong(raw[34]));
                }
            } catch (Exception e) {
                LOG.error(String.format("Fail to process following column : %s name: %s error: %s", columnNum,
                        this.columnConfigList.get(corrColumnNum).getColumnName(), e.getMessage()), e);
                continue;
            }
        }
    }

    public void runPSI() throws IOException {
        // check if could run PSI
        boolean toRunPSIWithStats = Environment.getBoolean("shifu.stats.psi.together", true);
        if(!toRunPSIWithStats) {
            LOG.info("shifu.stats.psi.together is not set, skip PSI calculate.");
            return;
        }
        if(StringUtils.isEmpty(modelConfig.getPsiColumnName())) {
            LOG.info("ModelConfig#stats#psiColumnName is not set, skip PSI calculate.");
            return;
        }
        ColumnConfig columnConfig = CommonUtils.findColumnConfigByName(columnConfigList,
                modelConfig.getPsiColumnName());
        if(columnConfig == null || isBadPSIColumn(columnConfig.getColumnStats().getDistinctCount())) {
            LOG.error(
                    "Unable compute PSI with ModelConfig#stats#psiColumnName \"{}\", the distinct count {} should be [2, 1000], not match ColumnConfig#columnBinning#binCategory count",
                    columnConfig != null ? columnConfig.getColumnName() : "unknown",
                    columnConfig != null
                            ? (columnConfig.getColumnStats().getDistinctCount() == null ? "null"
                                    : columnConfig.getColumnStats().getDistinctCount())
                            : "null");
            return;
        }
        LOG.info("Start to use {} to compute the PSI ", columnConfig.getColumnName());

        doRunPSI();
        processor.saveColumnConfigList();
        processor.syncDataToHdfs(modelConfig.getDataSet().getSource());
    }

    /**
     * Calculate the PSI
     * 
     * @throws IOException
     *             in scanners read exception
     */
    public void doRunPSI() throws IOException {
        Map<String, String> paramsMap = new HashMap<>();
        paramsMap.put("delimiter", CommonUtils.escapePigString(modelConfig.getDataSetDelimiter()));
        paramsMap.put("PSIColumn", modelConfig.getPsiColumnName().trim());
        paramsMap.put("column_parallel", Integer.toString(columnConfigList.size() / 10));
        paramsMap.put("value_index", "2");
        paramsMap.put(CommonConstants.MTL_INDEX, this.getMtlIndex() + "");

        PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getScriptPath("scripts/PSI.pig"), paramsMap);

        String psiPath;
        if(this.modelConfig.isMultiTask()) {
            psiPath = pathFinder.getPSIInfoPath(this.getMtlIndex());
        } else {
            psiPath = pathFinder.getPSIInfoPath();
        }

        List<Scanner> scanners = ShifuFileUtils.getDataScanners(psiPath, modelConfig.getDataSet().getSource());
        if(CollectionUtils.isEmpty(scanners)) {
            LOG.info("The PSI got failure during the computation");
            return;
        }

        String delimiter = Environment.getProperty(Constants.SHIFU_OUTPUT_DATA_DELIMITER, Constants.DEFAULT_DELIMITER);
        Splitter splitter = Splitter.on(delimiter).trimResults();

        List<String> unitStats = new ArrayList<String>(this.columnConfigList.size());
        for(Scanner scanner: scanners) {
            while(scanner.hasNext()) {
                // String[] output = scanner.nextLine().trim().split("\\|");
                String[] output = Lists.newArrayList(splitter.split(scanner.nextLine())).toArray(new String[0]);
                try {
                    int columnNum = Integer.parseInt(output[0]);
                    ColumnConfig config = this.columnConfigList.get(columnNum);
                    config.setPSI(Double.parseDouble(output[1]));
                    unitStats.add(output[0] + "|" + output[2] // PSI std
                            + "|" + output[3] // cosine
                            + "|" + output[4] // cosine std
                            + "|" + output[5]);
                    // config.setUnitStats(
                    // Arrays.asList(StringUtils.split(output[2], CalculateStatsUDF.CATEGORY_VAL_SEPARATOR)));
                } catch (Exception e) {
                    LOG.error("error in parsing", e);
                }
            }
            // close scanner
            IOUtils.closeQuietly(scanner);
        }

        // write unit stat into a temporary file
        ShifuFileUtils.createDirIfNotExists(new SourceFile(Constants.TMP, RawSourceData.SourceType.LOCAL));
        String ccUnitStatsFile;
        if(modelConfig.isMultiTask()) {
            ccUnitStatsFile = this.pathFinder.getColumnConfigUnitStatsPath(this.getMtlIndex());
        } else {
            ccUnitStatsFile = this.pathFinder.getColumnConfigUnitStatsPath();
        }
        ShifuFileUtils.writeLines(unitStats, ccUnitStatsFile, RawSourceData.SourceType.LOCAL);

        LOG.info("The Unit Stats is stored in - {}.", ccUnitStatsFile);
        LOG.info("Run PSI - done.");
    }

    private boolean isBadPSIColumn(Long distinctCount) {
        return (distinctCount == null || distinctCount < MINIMUM_DISTINCT_CNT || distinctCount > MAXIMUM_DISTINCT_CNT);
    }

}
