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

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import ml.shifu.guagua.hadoop.util.HDPUtils;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceConstants;
import ml.shifu.shifu.actor.AkkaSystemExecutor;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.core.mr.input.CombineInputFormat;
import ml.shifu.shifu.core.posttrain.FeatureImportanceMapper;
import ml.shifu.shifu.core.posttrain.FeatureImportanceReducer;
import ml.shifu.shifu.core.posttrain.FeatureStatsWritable;
import ml.shifu.shifu.core.posttrain.PostTrainMapper;
import ml.shifu.shifu.core.posttrain.PostTrainReducer;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.pig.PigExecutor;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;

import org.apache.commons.codec.binary.Base64;
import org.apache.commons.collections.Predicate;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.jexl2.JexlException;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
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
 * Post train processor, update the avg score
 */
public class PostTrainModelProcessor extends BasicModelProcessor implements Processor {

    /**
     * log object
     */
    private final static Logger log = LoggerFactory.getLogger(PostTrainModelProcessor.class);

    /**
     * runner for post train
     */
    @Override
    public int run() throws Exception {
        log.info("Step Start: posttrain");
        long start = System.currentTimeMillis();
        try {
            setUp(ModelStep.POSTTRAIN);
            syncDataToHdfs(modelConfig.getDataSet().getSource());
            if(modelConfig.isClassification()) {
                throw new IllegalArgumentException(
                        "post train step is only effective in regresion, not classification.");
            }
            if(modelConfig.isMapReduceRunMode()) {
                runMapRedPostTrain();
            } else if(modelConfig.isLocalRunMode()) {
                runAkkaPostTrain();
            } else {
                log.error("Invalid RunMode Setting!");
            }

            clearUp(ModelStep.POSTTRAIN);
        } catch (Exception e) {
            log.error("Error:", e);
            return -1;
        }
        log.info("Step Finished: posttrain with {} ms", (System.currentTimeMillis() - start));
        return 0;
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

    private void runMapRedPostTrain() throws IOException, InterruptedException, ClassNotFoundException {
        SourceType source = modelConfig.getDataSet().getSource();
        String postTrainOutputPath = super.getPathFinder().getTrainScoresPath(source);

        // run mr job to compute bin avg score
        runMRBinAvgScoreJob(source, postTrainOutputPath);

        // read from output file for avg score update
        updateAvgScores(source, postTrainOutputPath);

        ShifuFileUtils.deleteFile(new Path(postTrainOutputPath, "part-r-00000*").toString(), source);

        saveColumnConfigList();

        if(super.modelConfig.getBasic().getPostTrainOn() != null && super.modelConfig.getBasic().getPostTrainOn()) {
            syncDataToHdfs(modelConfig.getDataSet().getSource());
            String output = super.getPathFinder().getPostTrainOutputPath(source);
            runMRFeatureImportanceJob(source, output);
            List<Integer> fss = getFeatureImportance(source, output);
            log.info("Feature importance list is: {}", fss);
        }
    }

    private void updateAvgScores(SourceType source, String postTrainOutputPath) throws IOException {
        List<Scanner> scanners = null;
        try {
            scanners = ShifuFileUtils.getDataScanners(postTrainOutputPath, source, new PathFilter() {
                @Override
                public boolean accept(Path path) {
                    return path.toString().contains("part-r-");
                }
            });

            for(Scanner scanner: scanners) {
                while(scanner.hasNextLine()) {
                    String line = scanner.nextLine().trim();
                    String[] keyValues = line.split("\t");
                    String key = keyValues[0];
                    String value = keyValues[1];
                    ColumnConfig config = this.columnConfigList.get(Integer.parseInt(key));
                    List<Integer> binAvgScores = new ArrayList<Integer>();
                    String[] avgScores = value.split(",");
                    for(int i = 0; i < avgScores.length; i++) {
                        binAvgScores.add(Integer.parseInt(avgScores[i]));
                    }
                    config.setBinAvgScore(binAvgScores);
                }
            }
        } finally {
            // release
            closeScanners(scanners);
        }
    }

    private List<Integer> getFeatureImportance(SourceType source, String output) throws IOException {
        List<Integer> featureImportance = new ArrayList<Integer>();
        List<Scanner> scanners = null;
        try {
            scanners = ShifuFileUtils.getDataScanners(output, source, new PathFilter() {
                @Override
                public boolean accept(Path path) {
                    return path.toString().contains("part-r-");
                }
            });

            for(Scanner scanner: scanners) {
                while(scanner.hasNextLine()) {
                    String line = scanner.nextLine().trim();
                    String[] keyValues = line.split("\t");
                    String key = keyValues[0];
                    featureImportance.add(Integer.parseInt(key));
                }
            }
        } finally {
            // release
            closeScanners(scanners);
        }
        return featureImportance;
    }

    private void runMRBinAvgScoreJob(SourceType source, String postTrainOutputPath) throws IOException,
            InterruptedException, ClassNotFoundException {
        Configuration conf = new Configuration();
        // add jars to hadoop mapper and reducer
        new GenericOptionsParser(conf, new String[] { "-libjars", addRuntimeJars() });

        conf.setBoolean(CombineInputFormat.SHIFU_VS_SPLIT_COMBINABLE, true);
        conf.setBoolean("mapreduce.input.fileinputformat.input.dir.recursive", true);

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
        conf.setInt(GuaguaMapReduceConstants.MAPREDUCE_JOB_MAX_SPLIT_LOCATIONS, 5000);

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

        // one can set guagua conf in shifuconfig
        for(Map.Entry<Object, Object> entry: Environment.getProperties().entrySet()) {
            if(CommonUtils.isHadoopConfigurationInjected(entry.getKey().toString())) {
                conf.set(entry.getKey().toString(), entry.getValue().toString());
            }
        }

        @SuppressWarnings("deprecation")
        Job job = new Job(conf, "Shifu: Post Train : " + this.modelConfig.getModelSetName());
        job.setJarByClass(getClass());
        job.setMapperClass(PostTrainMapper.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(FeatureStatsWritable.class);
        job.setInputFormatClass(CombineInputFormat.class);
        FileInputFormat.setInputPaths(
                job,
                ShifuFileUtils.getFileSystemBySourceType(source).makeQualified(
                        new Path(super.modelConfig.getDataSetRawPath())));

        MultipleOutputs.addNamedOutput(job, Constants.POST_TRAIN_OUTPUT_SCORE, TextOutputFormat.class,
                NullWritable.class, Text.class);

        job.setReducerClass(PostTrainReducer.class);
        job.setNumReduceTasks(1);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileOutputFormat.setOutputPath(job, new Path(postTrainOutputPath));

        // clean output firstly
        ShifuFileUtils.deleteFile(postTrainOutputPath, source);

        // submit job
        if(!job.waitForCompletion(true)) {
            throw new RuntimeException("Post train Bin Avg Score MapReduce job is failed.");
        }
    }

    private void runMRFeatureImportanceJob(SourceType source, String output) throws IOException, InterruptedException,
            ClassNotFoundException {
        Configuration conf = new Configuration();
        // add jars to hadoop mapper and reducer
        new GenericOptionsParser(conf, new String[] { "-libjars", addRuntimeJars() });

        conf.setBoolean(CombineInputFormat.SHIFU_VS_SPLIT_COMBINABLE, true);
        conf.setBoolean("mapreduce.input.fileinputformat.input.dir.recursive", true);

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
        conf.setInt(GuaguaMapReduceConstants.MAPREDUCE_JOB_MAX_SPLIT_LOCATIONS, 5000);

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

        // one can set guagua conf in shifuconfig
        for(Map.Entry<Object, Object> entry: Environment.getProperties().entrySet()) {
            if(CommonUtils.isHadoopConfigurationInjected(entry.getKey().toString())) {
                conf.set(entry.getKey().toString(), entry.getValue().toString());
            }
        }

        @SuppressWarnings("deprecation")
        Job job = new Job(conf, "Shifu: Post Train FeatureImportance : " + this.modelConfig.getModelSetName());
        job.setJarByClass(getClass());
        job.setMapperClass(FeatureImportanceMapper.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(DoubleWritable.class);
        job.setInputFormatClass(CombineInputFormat.class);
        FileInputFormat.setInputPaths(
                job,
                ShifuFileUtils.getFileSystemBySourceType(source).makeQualified(
                        new Path(super.modelConfig.getDataSetRawPath())));

        job.setReducerClass(FeatureImportanceReducer.class);
        job.setNumReduceTasks(1);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(DoubleWritable.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        FileOutputFormat.setOutputPath(job, new Path(output));

        // clean output firstly
        ShifuFileUtils.deleteFile(output, source);

        // submit job
        if(!job.waitForCompletion(true)) {
            throw new RuntimeException("Post train Feature Importance MapReduce job is failed.");
        }
    }

    /**
     * run pig post train
     * 
     * @throws IOException
     *             for any io exception
     */
    @SuppressWarnings("unused")
    private void runPigPostTrain() throws IOException {
        SourceType sourceType = modelConfig.getDataSet().getSource();

        ShifuFileUtils.deleteFile(pathFinder.getTrainScoresPath(), sourceType);
        ShifuFileUtils.deleteFile(pathFinder.getBinAvgScorePath(), sourceType);

        // prepare special parameters and execute pig
        Map<String, String> paramsMap = new HashMap<String, String>();
        paramsMap.put("pathHeader", modelConfig.getHeaderPath());
        paramsMap.put("pathDelimiter", CommonUtils.escapePigString(modelConfig.getHeaderDelimiter()));
        paramsMap.put("delimiter", CommonUtils.escapePigString(modelConfig.getDataSetDelimiter()));

        try {
            PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getScriptPath("scripts/PostTrain.pig"),
                    paramsMap);
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_RUNNING_PIG_JOB, e);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }

        // Sync Down
        columnConfigList = updateColumnConfigWithBinAvgScore(columnConfigList);
        saveColumnConfigList();
    }

    /**
     * run akka post train
     * 
     * @throws IOException
     *             for any io exception
     */
    private void runAkkaPostTrain() throws IOException {
        SourceType sourceType = modelConfig.getDataSet().getSource();

        List<Scanner> scanners = ShifuFileUtils.getDataScanners(pathFinder.getSelectedRawDataPath(sourceType),
                sourceType);

        log.info("Num of Scanners: " + scanners.size());
        AkkaSystemExecutor.getExecutor().submitPostTrainJob(modelConfig, columnConfigList, scanners);

        closeScanners(scanners);
    }

    /**
     * read the binary average score and update them into column list
     * 
     * @param columnConfigList
     *            input column config list
     * @return updated column config list
     * @throws IOException
     *             for any io exception
     */
    private List<ColumnConfig> updateColumnConfigWithBinAvgScore(List<ColumnConfig> columnConfigList)
            throws IOException {
        List<Scanner> scanners = ShifuFileUtils.getDataScanners(pathFinder.getBinAvgScorePath(), modelConfig
                .getDataSet().getSource());

        // CommonUtils.getDataScanners(pathFinder.getBinAvgScorePath(), modelConfig.getDataSet().getSource());
        for(Scanner scanner: scanners) {
            while(scanner.hasNextLine()) {
                List<Integer> scores = new ArrayList<Integer>();
                String[] raw = scanner.nextLine().split("\\|");
                int columnNum = Integer.parseInt(raw[0]);
                for(int i = 1; i < raw.length; i++) {
                    scores.add(Integer.valueOf(raw[i]));
                }
                ColumnConfig config = columnConfigList.get(columnNum);
                config.setBinAvgScore(scores);
            }
        }

        // release
        closeScanners(scanners);

        return columnConfigList;
    }

}
