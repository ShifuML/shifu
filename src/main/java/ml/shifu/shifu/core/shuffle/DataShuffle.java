package ml.shifu.shifu.core.shuffle;

import ml.shifu.shifu.util.Base64Utils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Random;

/**
 * Created by zhanhu on 12/31/16.
 */
public class DataShuffle {

    public static final String POS_TAG = "1";

    public static class ShuffleMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
        private int shuffleSize;
        private int targetIndex;
        private double rblRatio;
        private String delimiter;
        private Random rd;
        private boolean duplicatePos;

        @Override
        public void setup(Context context) throws IOException {
            this.shuffleSize = context.getConfiguration().getInt(Constants.SHIFU_NORM_SHUFFLE_SIZE, 100);
            this.targetIndex = context.getConfiguration().getInt(Constants.SHIFU_NORM_SHUFFLE_RBL_TARGET_INDEX, -1);
            this.rblRatio = context.getConfiguration().getDouble(Constants.SHIFU_NORM_SHUFFLE_RBL_RATIO, -1.0d);
            this.delimiter = Base64Utils.base64Decode(context.getConfiguration().get(Constants.SHIFU_OUTPUT_DATA_DELIMITER, "|"));
            this.rd = new Random(System.currentTimeMillis());

            if (this.rblRatio > 1.0d) {
                duplicatePos = true;
            } else {
                duplicatePos = false;
                this.rblRatio = 1.0d / this.rblRatio;
            }
        }

        @Override
        public void map(LongWritable key, Text line, Context context) throws IOException, InterruptedException {
            if (this.targetIndex >= 0 && this.rblRatio > 0 ) {
                String[] fields = CommonUtils.split(line.toString(), this.delimiter);
                if (this.duplicatePos == POS_TAG.equals(CommonUtils.trimTag(fields[targetIndex]))) {
                    double totalRatio = rblRatio;
                    while (totalRatio > 0) {
                        double seed = rd.nextDouble();
                        if (seed < totalRatio) {
                            IntWritable shuffleIndex = new IntWritable(this.rd.nextInt(this.shuffleSize));
                            context.write(shuffleIndex, line);
                        }
                        totalRatio -= 1.0d;
                    }
                } else {
                    IntWritable shuffleIndex = new IntWritable(this.rd.nextInt(this.shuffleSize));
                    context.write(shuffleIndex, line);
                }
            } else {
                IntWritable shuffleIndex = new IntWritable(this.rd.nextInt(this.shuffleSize));
                context.write(shuffleIndex, line);
            }
        }
    }

    public static class ShuffleReducer extends Reducer<IntWritable, Text, NullWritable, Text> {

        @Override
        public void reduce(IntWritable key, Iterable<Text> iterable, Context context) throws IOException, InterruptedException {
            for ( Text record : iterable ) {
                context.write(NullWritable.get(), record);
            }
        }

    }

    public static class KvalPartitioner extends Partitioner<IntWritable, Text> {
        @Override
        public int getPartition(IntWritable key, Text text, int numReduceTasks) {
            return key.get() % numReduceTasks;
        }
    }
}
