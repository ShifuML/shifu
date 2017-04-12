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
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnFlag;
import ml.shifu.shifu.container.obj.ModelStatsConf;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.correlation.CorrelationMapper;
import ml.shifu.shifu.core.correlation.CorrelationMultithreadedMapper;
import ml.shifu.shifu.core.correlation.CorrelationReducer;
import ml.shifu.shifu.core.correlation.CorrelationWritable;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.core.mr.input.CombineInputFormat;
import ml.shifu.shifu.core.processor.stats.AbstractStatsExecutor;
import ml.shifu.shifu.core.processor.stats.AkkaStatsWorker;
import ml.shifu.shifu.core.processor.stats.DIBStatsExecutor;
import ml.shifu.shifu.core.processor.stats.MunroPatIStatsExecutor;
import ml.shifu.shifu.core.processor.stats.MunroPatStatsExecutor;
import ml.shifu.shifu.core.processor.stats.SPDTIStatsExecutor;
import ml.shifu.shifu.core.processor.stats.SPDTStatsExecutor;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;

import org.apache.commons.codec.binary.Base64;
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

    private boolean isComputeCorr = false;

    public StatsModelProcessor(boolean isComputeCorr) {
        this.isComputeCorr = isComputeCorr;
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
            setUp(ModelStep.STATS);
            // resync ModelConfig.json/ColumnConfig.json to HDFS
            syncDataToHdfs(modelConfig.getDataSet().getSource());

            if(isComputeCorr) {
                // 1. validate if run stats before run stats -correlation
                boolean foundValidMeanValueColumn = false;
                for(ColumnConfig config: this.columnConfigList) {
                    if(!config.isMeta() && !config.isTarget() && config.isNumerical()) {
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

                log.info("Start computing correlation value ...");
                // 2. compute correlation
                runCorrMapReduceJob();
                // 3. save column config list
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

        // add jars to hadoop mapper and reducer
        new GenericOptionsParser(conf, new String[] { "-libjars", addRuntimeJars() });

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

        // too many data needed to be transfered to reducer, set default completed maps to a smaller one 0.5 to start
        // copy data in reducer earlier.
        conf.set("mapred.reduce.slowstart.completed.maps",
                Environment.getProperty("mapred.reduce.slowstart.completed.maps", "0.5"));

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

        // one can set guagua conf in shifuconfig
        for(Map.Entry<Object, Object> entry: Environment.getProperties().entrySet()) {
            if(CommonUtils.isHadoopConfigurationInjected(entry.getKey().toString())) {
                conf.set(entry.getKey().toString(), entry.getValue().toString());
            }
        }

        int threads = parseThreadNum();
        setMapperMemory(conf, threads);

        @SuppressWarnings("deprecation")
        Job job = new Job(conf, "Shifu: Correlation Computing Job : " + this.modelConfig.getModelSetName());
        job.setJarByClass(getClass());
        job.setMapperClass(CorrelationMultithreadedMapper.class);
        CorrelationMultithreadedMapper.setMapperClass(job, CorrelationMapper.class);

        CorrelationMultithreadedMapper.setNumberOfThreads(job, threads);

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
    private void setMapperMemory(Configuration conf, int threads) {
        int memoryBuffer = 500;
        int memoryInContainer = this.columnConfigList.size() > 700 ? ((int) (this.columnConfigList.size() * 1d / 700))
                * 341 * threads : 341 * threads;
        memoryInContainer = (int) (memoryInContainer * 1d / threads);
        if(memoryInContainer < 2048) {
            memoryInContainer = 2048; // at least 2048M
        }
        memoryInContainer += memoryBuffer; // (MB, 500 is buffer)
        conf.set("mapreduce.map.memory.mb", memoryInContainer + "");
        conf.set(
                "mapreduce.map.java.opts",
                "-Xms"
                        + (memoryInContainer - memoryBuffer)
                        + "m -Xmx"
                        + (memoryInContainer - memoryBuffer)
                        + "m -server -XX:MaxPermSize=128M -XX:PermSize=64M -XX:+UseParallelGC -XX:+UseParallelOldGC -XX:ParallelGCThreads=8 -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps");
    }

    private int parseThreadNum() {
        int threads = 0;
        try {
            threads = Integer.parseInt(Environment.getProperty(Constants.SHIFU_CORRELATION_MULTI_THREADS, "6"));
        } catch (Exception e) {
            log.warn("'shifu.correlation.multi.threads' should be a int value, set default value: {}", 6);
            threads = 6;
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
        String localCorrelationCsv = super.pathFinder.getPathBySourceType("correlation.csv", SourceType.LOCAL);
        ShifuFileUtils.createFileIfNotExists(localCorrelationCsv, SourceType.LOCAL);
        BufferedWriter writer = null;
        try {
            writer = ShifuFileUtils.getWriter(localCorrelationCsv, SourceType.LOCAL);
            writer.write(getColumnIndexes());
            writer.newLine();
            writer.write(getColumnNames());
            writer.newLine();

            for(Entry<Integer, CorrelationWritable> entry: corrMap.entrySet()) {
                ColumnConfig xColumnConfig = this.columnConfigList.get(entry.getKey());
                if(xColumnConfig.getColumnFlag() == ColumnFlag.Meta) {
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
                        double[] reverseDoubleArray = this.columnConfigList.get(i).getCorrArray();
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
                }
                this.columnConfigList.get(entry.getKey()).setCorrArray(corrArray);
                String corrStr = Arrays.toString(corrArray);
                String adjustCorrStr = corrStr.substring(1, corrStr.length() - 1);
                writer.write(entry.getKey() + "," + this.columnConfigList.get(entry.getKey()).getColumnName() + ","
                        + adjustCorrStr);
                writer.newLine();
            }
            log.info("Please find corrlation csv file in local {}.", localCorrelationCsv);
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

}
