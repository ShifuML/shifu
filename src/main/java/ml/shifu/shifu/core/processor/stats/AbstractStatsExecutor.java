package ml.shifu.shifu.core.processor.stats;

import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.processor.BasicModelProcessor;

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
