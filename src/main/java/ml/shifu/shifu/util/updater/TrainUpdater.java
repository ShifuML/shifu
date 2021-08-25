package ml.shifu.shifu.util.updater;

import java.io.IOException;
import java.util.List;
import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.column.NSColumnUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnType;
import ml.shifu.shifu.container.obj.ModelConfig;

import org.apache.commons.collections.CollectionUtils;

/**
 * Created by zhanhu on 2/23/17.
 */
public class TrainUpdater extends BasicUpdater {

    public TrainUpdater(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, int mtlIndex) throws IOException {
        super(modelConfig, columnConfigList, mtlIndex);
    }

    @Override
    public void updateColumnConfig(ColumnConfig columnConfig) {
        // reset flag at first
        columnConfig.setColumnFlag(null);

        String varName = columnConfig.getColumnName();

        if(NSColumnUtils.isColumnEqual(this.targetColumnName, varName)) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Target);
            if(CollectionUtils.isEmpty(this.modelConfig.getTags())) {
                // allow tags are empty to support linear target
                // set columnType to N
                columnConfig.setColumnType(ColumnType.N);
            } else {
                // target column is set to categorical column
                columnConfig.setColumnType(ColumnType.C);
            }
        } else if(this.setMeta.contains(new NSColumn(varName))) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Meta);
            // set to false is OK as if no column are selected, set to false still no one selected
            columnConfig.setFinalSelect(false);
        } else if(this.setForceRemove.contains(new NSColumn(varName))) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.ForceRemove);
            // set to false is OK as if no column are selected, set to false still no one selected
            columnConfig.setFinalSelect(false);
        } else if(this.setForceSelect.contains(new NSColumn(varName))) {
            if(CollectionUtils.isEmpty(this.setCandidates) || (CollectionUtils.isNotEmpty(this.setCandidates)
                    // candidates is not empty
                    && this.setCandidates.contains(new NSColumn(varName)))) {
                columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.ForceSelect);
                // WARN: should not set final select here, imagine user take varsel by SE, the first time is to call
                // training a model, then forceselected columns will be set to final selected, then all varaibles
                // selected are only in current final selected columns which is not correct.

                // There is a situation like this - after variable selection, user may want to update forselect list
                // and train the model again, if we don't set finalSelect = true, those new added variables won't be
                // used. Or user need to run variable selection again. Let's figure out a solution to fix this.
            }
        } else if(this.setCandidates.contains(new NSColumn(varName))) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Candidate);
        }
    }
}
