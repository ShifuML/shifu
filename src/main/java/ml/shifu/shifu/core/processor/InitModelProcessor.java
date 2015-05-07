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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import ml.shifu.guagua.mapreduce.GuaguaMapReduceConstants;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnFlag;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnType;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.autotype.AutoTypeDistinctCountMapper;
import ml.shifu.shifu.core.autotype.AutoTypeDistinctCountReducer;
import ml.shifu.shifu.core.dtrain.NNConstants;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.HDPUtils;

import org.apache.commons.codec.binary.Base64;
import org.apache.commons.collections.Predicate;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.jexl2.JexlException;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
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

/**
 * Initialize processor, the purpose of this processor is create columnConfig based on modelConfig instance
 */
public class InitModelProcessor extends BasicModelProcessor implements Processor {

    /**
     * 
     */
    private static final String TAB_STR = "\t";
    /**
     * log object
     */
    private final static Logger log = LoggerFactory.getLogger(InitModelProcessor.class);

    /**
     * runner for init the model
     * 
     * @throws Exception
     */
    @Override
    public int run() throws Exception {
        log.info("Step Start: init");
        long start = System.currentTimeMillis();
        setUp(ModelStep.INIT);

        Map<Integer, Long> distinctCountMap = null;
        if(modelConfig.isMapReduceRunMode() && modelConfig.getDataSet().getAutoType()) {
            distinctCountMap = getApproxDistinctCountByMRJob();
        }

        // initialize ColumnConfig list
        int status = initColumnConfigList();
        if(status != 0) {
            return status;
        }

        if(distinctCountMap != null) {
            for(ColumnConfig columnConfig: columnConfigList) {
                Long distinctCount = distinctCountMap.get(columnConfig.getColumnNum());
                if(distinctCount != null) {
                    if(distinctCount < modelConfig.getDataSet().getAutoTypeThreshold().longValue()) {
                        columnConfig.setColumnType(ColumnType.C);
                        log.info("Column {} with index {} is set to categorical type according to auto type checking.", columnConfig.getColumnName(), columnConfig.getColumnNum());
                    }
                    columnConfig.getColumnStats().setDistinctCount(distinctCount);
                }
            }
        }
        // save ColumnConfig list into file
        saveColumnConfigList();

        clearUp(ModelStep.INIT);
        log.info("Step Finished: init with {} ms", (System.currentTimeMillis() - start));
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
        // stream-llib-*.jar
        jars.add(JarManager.findContainingJar(HyperLogLogPlus.class));

        return StringUtils.join(jars, NNConstants.LIB_JAR_SEPARATOR);
    }

    private Map<Integer, Long> getApproxDistinctCountByMRJob() throws IOException, InterruptedException,
            ClassNotFoundException {
        SourceType source = this.modelConfig.getDataSet().getSource();
        Configuration conf = new Configuration();

        // add jars to hadoop mapper and reducer
        new GenericOptionsParser(conf, new String[] { "-libjars", addRuntimeJars() });

        conf.setBoolean(GuaguaMapReduceConstants.MAPRED_MAP_TASKS_SPECULATIVE_EXECUTION, true);
        conf.setBoolean(GuaguaMapReduceConstants.MAPRED_REDUCE_TASKS_SPECULATIVE_EXECUTION, true);
        conf.set(NNConstants.MAPRED_JOB_QUEUE_NAME, Environment.getProperty(Environment.HADOOP_JOB_QUEUE, "default"));
        conf.setInt(GuaguaMapReduceConstants.MAPREDUCE_JOB_MAX_SPLIT_LOCATIONS, 30);
        conf.set(
                Constants.SHIFU_MODEL_CONFIG,
                ShifuFileUtils.getFileSystemBySourceType(source)
                        .makeQualified(new Path(super.getPathFinder().getModelConfigPath(source))).toString());
        conf.set("mapred.reduce.slowstart.completed.maps",
                Environment.getProperty("mapred.reduce.slowstart.completed.maps", "0.9"));
        String hdpVersion = HDPUtils.getHdpVersionForHDP224();
        if(StringUtils.isNotBlank(hdpVersion)) {
            conf.set("hdp.version", hdpVersion);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("hdfs-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("core-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("mapred-site.xml"), conf);
            HDPUtils.addFileToClassPath(HDPUtils.findContainingFile("yarn-site.xml"), conf);
        }

        Job job = new Job(conf, "Shifu: Column Type Auto Checking Job : " + this.modelConfig.getModelSetName());
        job.setJarByClass(getClass());
        job.setMapperClass(AutoTypeDistinctCountMapper.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(BytesWritable.class);
        job.setInputFormatClass(TextInputFormat.class);
        FileInputFormat.setInputPaths(
                job,
                ShifuFileUtils.getFileSystemBySourceType(source).makeQualified(
                        new Path(super.modelConfig.getDataSetRawPath())));

        job.setReducerClass(AutoTypeDistinctCountReducer.class);
        job.setNumReduceTasks(1);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(LongWritable.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        String autoTypePath = super.getPathFinder().getAutoTypeFilePath(source);
        FileOutputFormat.setOutputPath(job, new Path(autoTypePath));

        // clean output firstly
        ShifuFileUtils.deleteFile(autoTypePath, source);

        // submit job
        if(job.waitForCompletion(true)) {
            return getDistinctCountMap(source, autoTypePath);
        } else {
            throw new RuntimeException("MapReduce Job Auto Type Distinct Count failed.");
        }
    }

    private Map<Integer, Long> getDistinctCountMap(SourceType source, String autoTypePath) throws IOException {
        String outputFilePattern = autoTypePath + Path.SEPARATOR + "part-*";
        if(!ShifuFileUtils.isFileExists(outputFilePattern, source)) {
            throw new RuntimeException("Auto type checking output file not exist.");
        }

        Map<Integer, Long> distinctCountMap = new HashMap<Integer, Long>();
        List<Scanner> scanners = null;
        try {
            // here only works for 1 reducer
            FileStatus[] globStatus = ShifuFileUtils.getFileSystemBySourceType(source).globStatus(
                    new Path(outputFilePattern));
            if(globStatus == null || globStatus.length == 0) {
                throw new RuntimeException("Auto type checking output file not exist.");
            }
            scanners = ShifuFileUtils.getDataScanners(globStatus[0].getPath().toString(), source);
            Scanner scanner = scanners.get(0);
            String str = null;
            while(scanner.hasNext()) {
                str = scanner.nextLine().trim();
                if(str.contains(TAB_STR)) {
                    String[] splits = str.split(TAB_STR);
                    distinctCountMap.put(Integer.valueOf(splits[0]), Long.valueOf(splits[1]));
                }
            }
            return distinctCountMap;
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

    /**
     * initialize the columnConfig file
     * 
     * @throws IOException
     */
    private int initColumnConfigList() throws IOException {
        String[] fields = CommonUtils.getHeaders(modelConfig.getHeaderPath(), modelConfig.getHeaderDelimiter(),
                modelConfig.getDataSet().getSource());

        columnConfigList = new ArrayList<ColumnConfig>();
        for(int i = 0; i < fields.length; i++) {
            String varName = fields[i];
            ColumnConfig config = new ColumnConfig();
            config.setColumnNum(i);
            config.setColumnName(varName);
            columnConfigList.add(config);
        }

        CommonUtils.updateColumnConfigFlags(modelConfig, columnConfigList);

        boolean hasTarget = false;
        for(ColumnConfig config: columnConfigList) {
            if(config.isTarget()) {
                hasTarget = true;
            }
        }

        if(!hasTarget) {
            log.error("Target is not valid: " + modelConfig.getTargetColumnName());
            return 1;
        }

        return 0;
    }

}
