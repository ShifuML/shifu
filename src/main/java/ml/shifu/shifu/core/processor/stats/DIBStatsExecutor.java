package ml.shifu.shifu.core.processor.stats;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.processor.BasicModelProcessor;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.pig.PigExecutor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

/**
 * Created by zhanhu on 7/1/16.
 */
public class DIBStatsExecutor extends MapReducerStatsWorker {

    private static Logger log = LoggerFactory.getLogger(DIBStatsExecutor.class);

    public DIBStatsExecutor(BasicModelProcessor processor,
                            ModelConfig modelConfig,
                            List<ColumnConfig> columnConfigList) {
        super(processor, modelConfig, columnConfigList);
    }

    @Override
    protected void runStatsPig(Map<String, String> paramsMap) throws Exception {
        log.info("Run DynamicBinning to stats ... ");

        paramsMap.put("histo_scale_factor", Integer.toString(1000));
        ShifuFileUtils.deleteFile(pathFinder.getUpdatedBinningInfoPath(modelConfig.getDataSet().getSource()),
                modelConfig.getDataSet().getSource());

        PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getAbsolutePath("scripts/StatsDynamicBinning.pig"),
                paramsMap);

        // update
        log.info("Updating binning info ...");
        updateBinningInfoWithMRJob();
    }

}
