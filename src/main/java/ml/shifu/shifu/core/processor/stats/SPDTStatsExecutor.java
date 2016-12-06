package ml.shifu.shifu.core.processor.stats;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.processor.BasicModelProcessor;
import ml.shifu.shifu.pig.PigExecutor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

/**
 * Created by zhanhu on 7/1/16.
 */
public class SPDTStatsExecutor extends MapReducerStatsWorker {

    private static Logger log = LoggerFactory.getLogger(SPDTStatsExecutor.class);

    public SPDTStatsExecutor(BasicModelProcessor processor, ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(processor, modelConfig, columnConfigList);
    }

    @Override
    protected void runStatsPig(Map<String, String> paramsMap) throws Exception {
        log.info("Run SPDT to stats ... ");
        PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getAbsolutePath("scripts/PreTrainingStats.pig"),
                paramsMap, modelConfig.getDataSet().getSource(), super.pathFinder);
    }

}
