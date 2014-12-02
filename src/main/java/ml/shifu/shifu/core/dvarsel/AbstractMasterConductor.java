package ml.shifu.shifu.core.dvarsel;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;

import java.util.List;

/**
 * Created on 11/24/2014.
 */
public abstract class AbstractMasterConductor {
    protected ModelConfig modelConfig;
    protected List<ColumnConfig> columnConfigList;

    public AbstractMasterConductor(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
    }

    public abstract int getEstimateIterationCnt();
    public abstract boolean isToStop();

    public abstract List<Integer> getNextWorkingSet();
    public abstract void consumeWorkerResults(Iterable<VarSelWorkerResult> workerResults);
}
