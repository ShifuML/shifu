package ml.shifu.shifu.core.shuffle;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Splitter;
import ml.shifu.guagua.hadoop.util.HDPUtils;
import ml.shifu.guagua.mapreduce.GuaguaMapReduceConstants;
import ml.shifu.guagua.util.NumberFormatUtils;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
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
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.pig.impl.util.JarManager;
import org.encog.ml.data.MLDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by zhanhu on 2/22/17.
 */
public class MapReduceShuffle {

    private static final Logger log = LoggerFactory.getLogger(MapReduceShuffle.class);

    private PathFinder pathFinder;
    private ModelConfig modelConfig;

    public MapReduceShuffle(ModelConfig modelConfig) {
        this.modelConfig = modelConfig;
        this.pathFinder = new PathFinder(this.modelConfig);
    }

    public void run(String rawNormPath) throws IOException, ClassNotFoundException, InterruptedException {
        RawSourceData.SourceType source = this.modelConfig.getDataSet().getSource();
        Configuration conf = new Configuration();

        // add jars to hadoop mapper and reducer
        new GenericOptionsParser(conf, new String[] { "-libjars", addRuntimeJars() });

        conf.setBoolean(GuaguaMapReduceConstants.MAPRED_MAP_TASKS_SPECULATIVE_EXECUTION, true);
        conf.setBoolean(GuaguaMapReduceConstants.MAPRED_REDUCE_TASKS_SPECULATIVE_EXECUTION, true);
        conf.setBoolean(GuaguaMapReduceConstants.MAPREDUCE_MAP_SPECULATIVE, true);
        conf.setBoolean(GuaguaMapReduceConstants.MAPREDUCE_REDUCE_SPECULATIVE, true);
        conf.set(NNConstants.MAPRED_JOB_QUEUE_NAME, Environment.getProperty(Environment.HADOOP_JOB_QUEUE, "default"));
        conf.setInt(GuaguaMapReduceConstants.MAPREDUCE_JOB_MAX_SPLIT_LOCATIONS, 100);

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

        int shuffleSize = getDataShuffleSize(rawNormPath, source);
        log.info("Try to shuffle data into - {} parts.", shuffleSize);
        conf.set(Constants.SHIFU_NORM_SHUFFLE_SIZE, Integer.toString(shuffleSize));

        Job job = Job.getInstance(conf, "Shifu: Shuffling normalized data - " + this.modelConfig.getModelSetName());
        job.setJarByClass(getClass());
        job.setMapperClass(DataShuffle.ShuffleMapper.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);

        job.setPartitionerClass(DataShuffle.KvalPartitioner.class);

        job.setReducerClass(DataShuffle.ShuffleReducer.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(shuffleSize);

        FileInputFormat.setInputPaths(job, rawNormPath);
        FileOutputFormat.setOutputPath(job, new Path(this.pathFinder.getShuffleDataPath()));

        // clean output firstly
        ShifuFileUtils.deleteFile(this.pathFinder.getShuffleDataPath(), source);

        // submit job
        if(job.waitForCompletion(true)) {
            ShifuFileUtils.deleteFile(rawNormPath, source);
            ShifuFileUtils.move(this.pathFinder.getShuffleDataPath(), rawNormPath, source);
        } else {
            throw new RuntimeException("MapReduce Shuffle Computing Job Failed.");
        }
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

    private int getDataShuffleSize(String srcDataPath, RawSourceData.SourceType sourceType) throws IOException {
        // if user set fixed data shuffle size, then use it
        Integer fsize = Environment.getInt(Constants.SHIFU_NORM_SHUFFLE_SIZE);
        if(fsize != null) {
            return fsize;
        }

        // calculate data shuffle size based on user's prefer
        Long preferPartSize = Environment.getLong(Constants.SHIFU_NORM_PREFER_PART_SIZE);
        Long actualFileSize = ShifuFileUtils.getFileOrDirectorySize(srcDataPath, sourceType);

        if(preferPartSize != null && actualFileSize != null && preferPartSize != 0) {
            int dataShuffleSize = (int) (actualFileSize / preferPartSize);
            return ((actualFileSize % preferPartSize == 0) ? dataShuffleSize : (dataShuffleSize + 1));
        } else {
            return ShifuFileUtils.getFilePartCount(srcDataPath, sourceType);
        }
    }
}
