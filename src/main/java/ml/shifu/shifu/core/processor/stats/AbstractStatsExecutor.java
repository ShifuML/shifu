package ml.shifu.shifu.core.processor.stats;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.processor.BasicModelProcessor;
import org.apache.commons.collections.CollectionUtils;

import java.util.List;
import java.util.Scanner;

/**
 * Created by zhanhu on 6/30/16.
 */
public abstract class AbstractStatsExecutor {

    protected BasicModelProcessor processor;
    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;

    public AbstractStatsExecutor(BasicModelProcessor processor, ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        this.processor = processor;
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
    }

    public abstract boolean doStats() throws Exception;

}
