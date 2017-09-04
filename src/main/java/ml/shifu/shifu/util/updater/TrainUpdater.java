package ml.shifu.shifu.util.updater;

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;

import org.apache.commons.collections.CollectionUtils;

/**
 * Created by zhanhu on 2/23/17.
 */
public class TrainUpdater extends BasicUpdater {

    private boolean isForSegs = false;

    private List<String> segs;

    public TrainUpdater(ModelConfig modelConfig) throws IOException {
        super(modelConfig);

        segs = modelConfig.getSegmentFilterExpressions();
        if(segs.size() > 0) {
            this.isForSegs = true;
        }

        this.setMeta = new HashSet<NSColumn>();
        if(CollectionUtils.isNotEmpty(modelConfig.getMetaColumnNames())) {
            for(String meta: modelConfig.getMetaColumnNames()) {
                setMeta.add(new NSColumn(meta));
                if(this.isForSegs) {
                    for(int i = 0; i < segs.size(); i++) {
                        setMeta.add(new NSColumn(meta + "_" + (i + 1)));
                    }
                }
            }
        }

        setHybridColumns = new HashSet<NSColumn>();
        hybridColumnNames = modelConfig.getHybridColumnNames();
        if(hybridColumnNames != null && hybridColumnNames.size() > 0) {
            for(Entry<String, Double> entry: hybridColumnNames.entrySet()) {
                setHybridColumns.add(new NSColumn(entry.getKey()));
                if(this.isForSegs) {
                    for(int i = 0; i < segs.size(); i++) {
                        setHybridColumns.add(new NSColumn(entry.getKey() + "_" + (i + 1)));
                    }
                }
            }
        }

    }

    @Override
    public void updateColumnConfig(ColumnConfig columnConfig) {
        String varName = columnConfig.getColumnName();

        if(this.setMeta.contains(new NSColumn(varName))) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Meta);
            // set to false is OK as if no column are selected, set to false still no one selected
            columnConfig.setFinalSelect(false);
        } else if(this.setForceRemove.contains(new NSColumn(varName))) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.ForceRemove);
            // set to false is OK as if no column are selected, set to false still no one selected
            columnConfig.setFinalSelect(false);
        } else if(this.setForceSelect.contains(new NSColumn(varName))) {
            if(CollectionUtils.isEmpty(this.setCandidates) || (CollectionUtils.isNotEmpty(this.setCandidates) // candidates
                                                                                                              // is not
                                                                                                              // empty
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
