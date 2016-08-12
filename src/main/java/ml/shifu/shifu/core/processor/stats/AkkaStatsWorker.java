package ml.shifu.shifu.core.processor.stats;

import ml.shifu.shifu.actor.AkkaSystemExecutor;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.core.processor.BasicModelProcessor;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.ShifuFileUtils;
import org.apache.commons.collections.CollectionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.Scanner;

/**
 * Created by zhanhu on 6/30/16.
 */
public class AkkaStatsWorker extends AbstractStatsExecutor {

    private static Logger log = LoggerFactory.getLogger(AkkaStatsWorker.class);

    public AkkaStatsWorker(BasicModelProcessor processor, ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(processor, modelConfig, columnConfigList);
    }

    @Override
    public boolean doStats() throws Exception {
        List<Scanner> scanners = null;

        try {
            RawSourceData.SourceType sourceType = modelConfig.getDataSet().getSource();
            // the bug is caused when merging code? please take care
            scanners = ShifuFileUtils.getDataScanners(
                    ShifuFileUtils.expandPath(modelConfig.getDataSetRawPath(), sourceType), sourceType);
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_INPUT_NOT_FOUND, e);
        }

        if (CollectionUtils.isEmpty(scanners)) {
            throw new ShifuException(ShifuErrorCode.ERROR_INPUT_NOT_FOUND,
                    ", please check your data and start from init");
        }

        log.info("Num of Scanners: " + scanners.size());

        AkkaSystemExecutor.getExecutor().submitStatsCalJob(modelConfig, columnConfigList, scanners);

        // release
        processor.closeScanners(scanners);

        return true;
    }
}
