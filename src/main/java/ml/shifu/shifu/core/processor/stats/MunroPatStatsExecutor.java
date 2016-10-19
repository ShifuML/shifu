package ml.shifu.shifu.core.processor.stats;

import java.util.List;
import java.util.Map;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.processor.BasicModelProcessor;
import ml.shifu.shifu.pig.PigExecutor;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by zhanhu on 7/1/16.
 */
public class MunroPatStatsExecutor extends MapReducerStatsWorker {

    private static Logger log = LoggerFactory.getLogger(MunroPatStatsExecutor.class);

    public MunroPatStatsExecutor(BasicModelProcessor processor, ModelConfig modelConfig,
            List<ColumnConfig> columnConfigList) {
        super(processor, modelConfig, columnConfigList);
    }

    @Override
    protected void runStatsPig(Map<String, String> paramsMap) throws Exception {
        log.info("Run MunroPat to stats ... ");

        PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getAbsolutePath("scripts/Stats.pig"), paramsMap,
                modelConfig.getDataSet().getSource(), super.pathFinder);
    }

}
