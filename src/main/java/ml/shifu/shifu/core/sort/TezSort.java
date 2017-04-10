package ml.shifu.shifu.core.sort;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.Tool;
import org.apache.tez.client.TezClient;
import org.apache.tez.dag.api.DAG;
import org.apache.tez.dag.api.TezConfiguration;
import org.apache.tez.dag.api.TezException;
import org.apache.tez.runtime.api.ProcessorContext;
import org.apache.tez.runtime.library.api.KeyValueReader;
import org.apache.tez.runtime.library.api.KeyValuesWriter;
import org.apache.tez.runtime.library.processor.SimpleProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Created by Mark on 3/22/2017.
 */
public class TezSort extends Configured implements Tool {

    private final static Logger log = LoggerFactory.getLogger(TezSort.class);
    private final static String JOBNAME = "ml.shifu.shifu.tez.sort.name";

    private String[] header;

    private final static String SAMPLER = "Sampler";
    private final static String SORTER = "Sorter";

    public TezSort(String[] header) {
        this.header = header;
    }

    @Override
    public int run(String[] args) throws Exception {
        Configuration conf = getConf();
        new GenericOptionsParser(conf, args);
        return execute(conf);
    }

    private int execute(Configuration conf) throws IOException, TezException {
        TezConfiguration tezConf = new TezConfiguration(conf);
        TezClient tezClient = TezClient.create(getClass().getSimpleName(), tezConf);

        log.info("Start connect Yarn cluster");
        long startTime = System.currentTimeMillis();
        tezClient.start();

        log.info("Finish connect Yarn cluster in {} seconds", (System.currentTimeMillis() - startTime) / 1000 );

        DAG dag = createDAG(tezConf);

        return 0;
    }

    private DAG createDAG(TezConfiguration tezConf) {

        DAG dag = DAG.create(tezConf.get(JOBNAME));


        return dag;
    }

    public static class SamplerProcessor extends SimpleProcessor {

        public SamplerProcessor(ProcessorContext context) {
            super(context);
        }

        @Override
        public void run() throws Exception {
            KeyValueReader kvReader = (KeyValueReader) getInputs().get(SAMPLER).getReader();
            KeyValuesWriter kvsWriter = (KeyValuesWriter) getOutputs().get(SORTER).getWriter();
        }
    }



}
