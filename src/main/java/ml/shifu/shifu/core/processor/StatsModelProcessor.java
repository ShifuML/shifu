/**
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
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import ml.shifu.guagua.mapreduce.GuaguaMapReduceConstants;
import ml.shifu.guagua.util.FileUtils;
import ml.shifu.shifu.actor.AkkaSystemExecutor;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnType;
import ml.shifu.shifu.container.obj.ModelStatsConf;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.binning.BinningInfoWritable;
import ml.shifu.shifu.core.binning.UpdateBinningInfoMapper;
import ml.shifu.shifu.core.binning.UpdateBinningInfoReducer;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.core.mr.input.CombineInputFormat;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.pig.PigExecutor;
import ml.shifu.shifu.udf.CalculateStatsUDF;
import ml.shifu.shifu.util.Base64Utils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.HDPUtils;

import org.apache.commons.codec.binary.Base64;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.Predicate;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.jexl2.JexlException;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
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

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Splitter;

/**
 * statistics, max/min/avg/std for each column dataset if it's numerical
 */
public class StatsModelProcessor extends BasicModelProcessor implements Processor {

    private final static Logger log = LoggerFactory.getLogger(StatsModelProcessor.class);

    /**
     * runner for statistics
     */
    @Override
    public int run() throws Exception {
        log.info("Step Start: stats");
        long start = System.currentTimeMillis();
        try {
            setUp(ModelStep.STATS);

            syncDataToHdfs(modelConfig.getDataSet().getSource());

            if(modelConfig.isMapReduceRunMode()) {
                runMapRedStats();
            } else if(modelConfig.isLocalRunMode()) {
                runAkkaStats();
            } else {
                throw new ShifuException(ShifuErrorCode.ERROR_UNSUPPORT_MODE);
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

    /**
     * run akka stats
     */
    private void runAkkaStats() {
        List<Scanner> scanners = null;

        try {
            SourceType sourceType = modelConfig.getDataSet().getSource();
            // the bug is caused when merging code? please take care
            scanners = ShifuFileUtils.getDataScanners(
                    ShifuFileUtils.expandPath(modelConfig.getDataSetRawPath(), sourceType), sourceType);
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_INPUT_NOT_FOUND, e);
        }

        if(CollectionUtils.isEmpty(scanners)) {
            throw new ShifuException(ShifuErrorCode.ERROR_INPUT_NOT_FOUND,
                    ", please check your data and start from init");
        }

        log.info("Num of Scanners: " + scanners.size());

        AkkaSystemExecutor.getExecutor().submitStatsCalJob(modelConfig, columnConfigList, scanners);

        // release
        closeScanners(scanners);
    }

    /**
     * run mapreduce stats
     * 
     * @throws IOException
     */
    private void runMapRedStats() throws IOException {
        log.info("delete historical pre-train data");

        ShifuFileUtils.deleteFile(pathFinder.getPreTrainingStatsPath(), modelConfig.getDataSet().getSource());
        Map<String, String> paramsMap = new HashMap<String, String>();
        paramsMap.put("delimiter", CommonUtils.escapePigString(modelConfig.getDataSetDelimiter()));
        if(columnConfigList.size() <= 1000) {
            paramsMap.put("column_parallel", Integer.toString(columnConfigList.size() / 5));
        } else {
            paramsMap.put("column_parallel", Integer.toString(columnConfigList.size() / 4));
        }
        paramsMap.put("histo_scale_factor", Environment.getProperty("shifu.stats.histo.scale.factor", "100"));

        // FIXME how to estimate mapper size then to estimate reducer size, in stats, reducer size is estimated by
        // column_parallel is not a good
        // new PigInputFormat().getSplits(jobcontext)
        // execute pig job
        try {
            log.info("execute binning algorithm: {}", modelConfig.getBinningAlgorithm().toString());
            if(modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.MunroPat)) {
                PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getAbsolutePath("scripts/Stats.pig"),
                        paramsMap);
            } else if(modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.SPDT)) {
                PigExecutor.getExecutor().submitJob(modelConfig,
                        pathFinder.getAbsolutePath("scripts/PreTrainingStats.pig"), paramsMap);
            } else if(modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.SPDTI)) {
                paramsMap.put("group_binning_parallel", Integer.toString(columnConfigList.size() / (5 * 8)));
                ShifuFileUtils.deleteFile(pathFinder.getUpdatedBinningInfoPath(modelConfig.getDataSet().getSource()),
                        modelConfig.getDataSet().getSource());

                PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getAbsolutePath("scripts/StatsSpdtI.pig"),
                        paramsMap);
                // update
                log.info("Updating binning info ...");
                updateBinningInfoWithMRJob();
            } else if(modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.MunroPatI)) {
                ShifuFileUtils.deleteFile(pathFinder.getUpdatedBinningInfoPath(modelConfig.getDataSet().getSource()),
                        modelConfig.getDataSet().getSource());
                PigExecutor.getExecutor().submitJob(modelConfig,
                        pathFinder.getAbsolutePath("scripts/StatsMunroPatI.pig"), paramsMap);
                // update
                log.info("Updating binning info ...");
                updateBinningInfoWithMRJob();
            }
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_RUNNING_PIG_JOB, e);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }

        // sync Down
        log.info("Updating ColumnConfig with stats...");
        // update column config
        updateColumnConfigWithPreTrainingStats();
        // save it to local/hdfs
        saveColumnConfigListAndColumnStats(true);
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

        return StringUtils.join(jars, NNConstants.LIB_JAR_SEPARATOR);
    }

    private void prepareJobConf(SourceType source, Configuration conf, String filePath) throws IOException {
        // add jars to hadoop mapper and reducer
        new GenericOptionsParser(conf, new String[] { "-libjars", addRuntimeJars(), "-files", filePath });

        conf.setBoolean(CombineInputFormat.SHIFU_VS_SPLIT_COMBINABLE, true);

        conf.set(Constants.SHIFU_STATS_EXLCUDE_MISSING,
                Environment.getProperty(Constants.SHIFU_STATS_EXLCUDE_MISSING, "true"));

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
        conf.set("mapred.reduce.slowstart.completed.maps",
                Environment.getProperty("mapred.reduce.slowstart.completed.maps", "0.8"));
        String hdpVersion = HDPUtils.getHdpVersionForHDP224();
        if(StringUtils.isNotBlank(hdpVersion)) {
            // for hdp 2.2.4, hdp.version should be set and configuration files should be add to container class path
            conf.set("hdp.version", hdpVersion);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("hdfs-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("core-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("mapred-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("yarn-site.xml"), conf);
        }
    }

    private void updateBinningInfoWithMRJob() throws IOException, InterruptedException, ClassNotFoundException {
        SourceType source = this.modelConfig.getDataSet().getSource();
        Configuration conf = new Configuration();

        // TODO directly use hdfs file
        String filePath = Constants.BINNING_INFO_FILE_NAME;
        BufferedWriter writer = null;
        List<Scanner> scanners = null;
        try {
            scanners = ShifuFileUtils.getDataScanners(pathFinder.getUpdatedBinningInfoPath(source), source);
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File(filePath)),
                    Charset.forName("UTF-8")));
            for(Scanner scanner: scanners) {
                while(scanner.hasNextLine()) {
                    String line = scanner.nextLine();
                    writer.write(line + "\n");
                }
            }
        } finally {
            // release
            closeScanners(scanners);
            if(writer != null) {
                writer.close();
            }
        }

        prepareJobConf(source, conf, filePath);

        @SuppressWarnings("deprecation")
        Job job = new Job(conf, "Shifu: Stats Updating Binning Job : " + this.modelConfig.getModelSetName());
        job.setJarByClass(getClass());
        job.setMapperClass(UpdateBinningInfoMapper.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(BinningInfoWritable.class);
        job.setInputFormatClass(CombineInputFormat.class);
        FileInputFormat.setInputPaths(
                job,
                ShifuFileUtils.getFileSystemBySourceType(source).makeQualified(
                        new Path(super.modelConfig.getDataSetRawPath())));

        job.setReducerClass(UpdateBinningInfoReducer.class);
        job.setNumReduceTasks(1);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        String preTrainingInfo = super.getPathFinder().getPreTrainingStatsPath(source);
        FileOutputFormat.setOutputPath(job, new Path(preTrainingInfo));

        // clean output firstly
        ShifuFileUtils.deleteFile(preTrainingInfo, source);

        // submit job
        if(!job.waitForCompletion(true)) {
            FileUtils.deleteQuietly(new File(filePath));
            throw new RuntimeException("MapReduce Job Updateing Binning Info failed.");
        }
        FileUtils.deleteQuietly(new File(filePath));
    }

    /**
     * update the max/min/mean/std/binning information from stats step
     * 
     * @throws IOException
     */
    public void updateColumnConfigWithPreTrainingStats() throws IOException {
        List<Scanner> scanners = ShifuFileUtils.getDataScanners(pathFinder.getPreTrainingStatsPath(), modelConfig
                .getDataSet().getSource());
        for(Scanner scanner: scanners) {
            scanStatsResult(scanner);
        }

        // release
        closeScanners(scanners);
    }

    /**
     * Scan the stats result and save them into column configure
     * 
     * @param scanner
     */
    private void scanStatsResult(Scanner scanner) {
        while(scanner.hasNextLine()) {
            String[] raw = scanner.nextLine().trim().split("\\|");

            if(raw.length == 1) {
                continue;
            }

            if(raw.length < 25) {
                log.info("The stats data has " + raw.length + " fields.");
                log.info("The stats data is - " + Arrays.toString(raw));
            }

            int columnNum = Integer.parseInt(raw[0]);
            try {
                ColumnConfig config = this.columnConfigList.get(columnNum);

                if(config.isCategorical()) {
                    String binCategory = Base64Utils.base64Decode(raw[1]);
                    config.setBinCategory(CommonUtils.stringToStringList(binCategory,
                            CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));
                } else {
                    config.setBinBoundary(CommonUtils.stringToDoubleList(raw[1]));
                }
                config.setBinCountNeg(CommonUtils.stringToIntegerList(raw[2]));
                config.setBinCountPos(CommonUtils.stringToIntegerList(raw[3]));
                // config.setBinAvgScore(CommonUtils.stringToIntegerList(raw[4]));
                config.setBinPosCaseRate(CommonUtils.stringToDoubleList(raw[5]));
                config.setBinLength(config.getBinCountNeg().size());
                config.setKs(parseDouble(raw[6]));
                config.setIv(parseDouble(raw[7]));
                config.setMax(parseDouble(raw[8]));
                config.setMin(parseDouble(raw[9]));
                config.setMean(parseDouble(raw[10]));
                config.setStdDev(parseDouble(raw[11], Double.NaN));

                // magic?
                if(raw[12].equals("N")) {
                    config.setColumnType(ColumnType.N);
                } else {
                    config.setColumnType(ColumnType.C);
                }

                config.setMedian(parseDouble(raw[13]));

                config.setMissingCnt(parseLong(raw[14]));
                config.setTotalCount(parseLong(raw[15]));
                config.setMissingPercentage(parseDouble(raw[16]));

                config.setBinWeightedNeg(CommonUtils.stringToDoubleList(raw[17]));
                config.setBinWeightedPos(CommonUtils.stringToDoubleList(raw[18]));
                config.getColumnStats().setWoe(parseDouble(raw[19]));
                config.getColumnStats().setWeightedWoe(parseDouble(raw[20]));
                config.getColumnStats().setWeightedKs(parseDouble(raw[21]));
                config.getColumnStats().setWeightedIv(parseDouble(raw[22]));
                config.getColumnBinning().setBinCountWoe(CommonUtils.stringToDoubleList(raw[23]));
                config.getColumnBinning().setBinWeightedWoe(CommonUtils.stringToDoubleList(raw[24]));
                if(raw.length >= 26) {
                    config.getColumnStats().setSkewness(parseDouble(raw[25]));
                }
                if(raw.length >= 27) {
                    config.getColumnStats().setKurtosis(parseDouble(raw[26]));
                }
            } catch (Exception e) {
                log.error(String.format("Fail to process following column : %s name: %s error: %s", columnNum,
                        this.columnConfigList.get(columnNum).getColumnName(), e.getMessage()), e);
                continue;
            }
        }
    }

    private static double parseDouble(String str) {
        return parseDouble(str, 0d);
    }

    private static double parseDouble(String str, double dVal) {
        try {
            return Double.parseDouble(str);
        } catch (Exception e) {
            return dVal;
        }
    }

    private static long parseLong(String str) {
        return parseLong(str, 0L);
    }

    private static long parseLong(String str, long lVal) {
        try {
            return Long.parseLong(str);
        } catch (Exception e) {
            return lVal;
        }
    }

}
