package ml.shifu.shifu.core.varselect.itsa;

import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.worker.AbstractWorkerComputable;
import ml.shifu.guagua.worker.WorkerContext;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import java.io.IOException;

/**
 * Created by zhanhu on 11/7/16.
 */
public class IteSAWorker extends AbstractWorkerComputable<MasterIteSAParams, WorkerIteSAParams,
        GuaguaWritableAdapter<LongWritable>, GuaguaWritableAdapter<Text>> {

    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {

    }

    @Override
    public void init(WorkerContext<MasterIteSAParams, WorkerIteSAParams> context) {

    }

    @Override
    public WorkerIteSAParams doCompute(WorkerContext<MasterIteSAParams, WorkerIteSAParams> context) {
        return null;
    }

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue, WorkerContext<MasterIteSAParams, WorkerIteSAParams> context) {

    }
}
