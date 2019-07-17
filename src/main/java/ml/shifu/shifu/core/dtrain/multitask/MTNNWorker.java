package ml.shifu.shifu.core.dtrain.multitask;

import ml.shifu.guagua.ComputableMonitor;
import ml.shifu.guagua.hadoop.io.GuaguaWritableAdapter;
import ml.shifu.guagua.io.GuaguaFileSplit;
import ml.shifu.guagua.worker.AbstractWorkerComputable;
import ml.shifu.guagua.worker.WorkerContext;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

/**
 * @author haillu
 * @date 7/17/2019 5:05 PM
 */
@ComputableMonitor(timeUnit = TimeUnit.SECONDS, duration = 3600)
public class MTNNWorker extends
        AbstractWorkerComputable<MTNNParams, MTNNParams, GuaguaWritableAdapter<LongWritable>, GuaguaWritableAdapter<Text>> {

    protected static final Logger LOG = LoggerFactory.getLogger(MTNNWorker.class);





    @Override
    public void initRecordReader(GuaguaFileSplit fileSplit) throws IOException {

    }

    @Override
    public void init(WorkerContext<MTNNParams, MTNNParams> context) {

    }

    @Override
    public MTNNParams doCompute(WorkerContext<MTNNParams, MTNNParams> context) {
        return null;
    }

    @Override
    public void load(GuaguaWritableAdapter<LongWritable> currentKey, GuaguaWritableAdapter<Text> currentValue, WorkerContext<MTNNParams, MTNNParams> context) {

    }
}
