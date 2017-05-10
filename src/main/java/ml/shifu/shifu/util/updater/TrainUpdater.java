package ml.shifu.shifu.util.updater;

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import org.apache.commons.collections.CollectionUtils;

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

        if(this.setMeta.contains(new NSColumn(varName))) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Meta);
            columnConfig.setFinalSelect(false);
        } else if(this.setForceRemove.contains(new NSColumn(varName))) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.ForceRemove);
            columnConfig.setFinalSelect(false);
        } else if(this.setForceSelect.contains(new NSColumn(varName))) {
            if ( CollectionUtils.isEmpty(this.setCandidates)
                    || (CollectionUtils.isNotEmpty(this.setCandidates) && this.setCandidates.contains(new NSColumn(varName))) ) {
                columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.ForceSelect);
                columnConfig.setFinalSelect(true);
            }
        }
    }
}
