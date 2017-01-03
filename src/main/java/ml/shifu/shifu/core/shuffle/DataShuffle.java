package ml.shifu.shifu.core.shuffle;

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

    public static class ShuffleMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
        private int shuffleSize;
        private Random rd;

        @Override
        public void setup(Context context) {
            this.shuffleSize = context.getConfiguration().getInt(Constants.SHIFU_NORM_SHUFFLE_SIZE, 100);
            this.rd = new Random(System.currentTimeMillis());
        }

        @Override
        public void map(LongWritable key, Text line, Context context) throws IOException, InterruptedException {
            IntWritable shuffleIndex = new IntWritable(this.rd.nextInt(this.shuffleSize));
            context.write(shuffleIndex, line);
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
