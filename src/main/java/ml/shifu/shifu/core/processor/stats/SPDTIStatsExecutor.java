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
public class SPDTIStatsExecutor extends MapReducerStatsWorker {

    private static Logger log = LoggerFactory.getLogger(SPDTIStatsExecutor.class);

    public SPDTIStatsExecutor(BasicModelProcessor processor,
                             ModelConfig modelConfig,
                             List<ColumnConfig> columnConfigList) {
        super(processor, modelConfig, columnConfigList);
    }

    @Override
    protected void runStatsPig(Map<String, String> paramsMap) throws Exception {
        log.info("Run SPDTI to stats ... ");

        paramsMap.put("group_binning_parallel", Integer.toString(columnConfigList.size() / (5 * 8)));
        ShifuFileUtils.deleteFile(pathFinder.getUpdatedBinningInfoPath(modelConfig.getDataSet().getSource()),
                modelConfig.getDataSet().getSource());

        PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getAbsolutePath("scripts/StatsSpdtI.pig"),
                paramsMap);

        // update
        log.info("Updating binning info ...");
        updateBinningInfoWithMRJob();
    }
}
