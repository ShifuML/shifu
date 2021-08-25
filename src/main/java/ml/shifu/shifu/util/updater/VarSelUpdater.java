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
public class VarSelUpdater extends BasicUpdater {

    public VarSelUpdater(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, int mtlIndex) throws IOException {
        super(modelConfig, columnConfigList, mtlIndex);
    }

    @Override
    public void updateColumnConfig(ColumnConfig columnConfig) {
        String varName = columnConfig.getColumnName();

        // TODO check me: Before varselect, user can still change forceselect and force remove files while can user
        // change meta and target columns???

        // set column flag to null, before reset it
        columnConfig.setColumnFlag(null);

        // No need reset ColumnType since column type should be set well in stats and later cannot be changed
        if(NSColumnUtils.isColumnEqual(this.targetColumnName, varName)) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Target);
            List<String> tags = this.modelConfig.isMultiTask() ? this.modelConfig.getMTLTags(this.getMtlIndex())
                    : this.modelConfig.getTags();
            if(CollectionUtils.isEmpty(tags)) {
                // allow tags are empty to support linear target
                // set columnType to N
                columnConfig.setColumnType(ColumnType.N);
            } else {
                // target column is set to categorical column
                columnConfig.setColumnType(ColumnType.C);
            }
        } else if(this.setMeta.contains(new NSColumn(varName))) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Meta);
        } else if(this.setForceRemove.contains(new NSColumn(varName))) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.ForceRemove);
        } else if(this.setForceSelect.contains(new NSColumn(varName))) {
            if(CollectionUtils.isEmpty(this.setCandidates) || (CollectionUtils.isNotEmpty(this.setCandidates) // candidates
                                                                                                              // is not
                                                                                                              // empty
                    && this.setCandidates.contains(new NSColumn(varName)))) {
                columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.ForceSelect);
            }
        } else if(this.setCandidates.contains(new NSColumn(varName))) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Candidate);
        } else if(this.setCategoricalColumns.contains(new NSColumn(varName))) {
            columnConfig.setColumnType(ColumnType.C);
        }
    }
}
