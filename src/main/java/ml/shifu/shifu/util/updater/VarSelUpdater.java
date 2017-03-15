package ml.shifu.shifu.util.updater;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;

import java.io.IOException;

/**
 * Created by zhanhu on 2/23/17.
 */
public class VarSelUpdater extends BasicUpdater {

    public VarSelUpdater(ModelConfig modelConfig) throws IOException {
        super(modelConfig);
    }

    public void updateColumnConfig(ColumnConfig columnConfig) {
        String varName = columnConfig.getColumnName();

        if ( this.targetColumnName.equals(varName) ) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Target);
        } if (this.setMeta.contains(varName)) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Meta);
        } else if (this.setForceRemove.contains(varName)) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.ForceRemove);
        } else if (this.setForceSelect.contains(varName)) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.ForceSelect);
        }
    }

}
