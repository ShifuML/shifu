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

    public static class ShuffleMapper extends Mapper<LongWritable, Text, IntWritable, Text> {

        private int shuffleSize;
        private AbstractDataMapper dataMapper;

        @Override
        public void setup(Context context) throws IOException {
            this.shuffleSize = context.getConfiguration()
                    .getInt(Constants.SHIFU_NORM_SHUFFLE_SIZE, 100);
            int targetIndex = context.getConfiguration()
                    .getInt(Constants.SHIFU_NORM_SHUFFLE_RBL_TARGET_INDEX, -1);

            if (targetIndex < 0) { // no duplicate
                dataMapper = new RandomConstDataMapper();
            } else {
                double rblRatio = context.getConfiguration()
                        .getDouble(Constants.SHIFU_NORM_SHUFFLE_RBL_RATIO, -1.0d);
                boolean rblUpdateWeight = context.getConfiguration()
                        .getBoolean(Constants.SHIFU_NORM_SHUFFLE_RBL_UPDATE_WEIGHT, false);
                String delimiter = Base64Utils.base64Decode(context.getConfiguration()
                        .get(Constants.SHIFU_OUTPUT_DATA_DELIMITER, "|"));

                if (rblUpdateWeight) { // duplicate by update weight column
                    dataMapper = new UpdateWeightDataMapper(rblRatio, targetIndex, delimiter);
                } else { // duplicate by add duplicate records
                    dataMapper = new DuplicateDataMapper(rblRatio, targetIndex, delimiter);
                }
            }
        }

        @Override
        public void map(LongWritable key, Text line, Context context) throws IOException, InterruptedException {
            dataMapper.mapData(context, line, this.shuffleSize);
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
