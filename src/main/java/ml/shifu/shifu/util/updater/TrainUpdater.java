package ml.shifu.shifu.util.updater;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;

import java.io.IOException;

/**
 * Created by zhanhu on 2/23/17.
 */
public class TrainUpdater extends BasicUpdater {

    public TrainUpdater(ModelConfig modelConfig) throws IOException {
        super(modelConfig);
    }

    public void updateColumnConfig(ColumnConfig columnConfig) {
        String varName = columnConfig.getColumnName();

        if (this.setMeta.contains(varName)) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Meta);
            columnConfig.setFinalSelect(false);
        } else if (this.setForceRemove.contains(varName)) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.ForceRemove);
            columnConfig.setFinalSelect(false);
        } else if (this.setForceSelect.contains(varName)) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.ForceSelect);
            columnConfig.setFinalSelect(true);
        }
    }
}
