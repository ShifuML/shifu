package ml.shifu.shifu.combo;

import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

/**
 * Created by zhanhu on 12/13/16.
 */
// TODO
public class DataJoinMapReducer {

    public static class DJMapper extends Mapper<LongWritable, Text, Text, ArrayWritable> {
        @SuppressWarnings("unused")
        private String fileName;

        @Override
        public void setup(Context context) {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            this.fileName = context.getConfiguration().get(fileSplit.getPath().getParent().getName());
        }

        @Override
        public void map(LongWritable key, Text input, Context context) {

        }
    }

    public static class DJReducer extends Reducer<Text, ArrayWritable, Text, Text> {

    }
}
