package ml.shifu.shifu.util.updater;

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;

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

    private boolean isForSegs = false;

    private List<String> segs;

    public VarSelUpdater(ModelConfig modelConfig) throws IOException {
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

        this.setCategorialColumns = new HashSet<NSColumn>();
        if (CollectionUtils.isNotEmpty(modelConfig.getCategoricalColumnNames())) {
            for(String category: modelConfig.getCategoricalColumnNames()) {
                this.setCategorialColumns.add(new NSColumn(category));
                if(this.isForSegs) {
                    for(int i = 0; i < segs.size(); i++) {
                        this.setCategorialColumns.add(new NSColumn(category + "_" + (i + 1)));
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

        // TODO check me: Before varselect, user can still change forceselect and force remove files while can user
        // change meta and target columns???

        // set column flag to null, before reset it
        columnConfig.setColumnFlag(null);

        // No need reset ColumnType since column type should be set well in stats and later cannot be changed
        if(NSColumnUtils.isColumnEqual(this.targetColumnName, varName)) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Target);
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
        } else if(NSColumnUtils.isColumnEqual(this.weightColumnName, varName)) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Weight);
        } else if(this.setCandidates.contains(new NSColumn(varName))) {
            columnConfig.setColumnFlag(ColumnConfig.ColumnFlag.Candidate);
        } else if ( this.setCategorialColumns.contains(new NSColumn(varName)) ) {
            columnConfig.setColumnType(ColumnType.C);
        }
    }
}
