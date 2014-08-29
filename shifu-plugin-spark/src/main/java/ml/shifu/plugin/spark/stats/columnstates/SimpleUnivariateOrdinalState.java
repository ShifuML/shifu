package ml.shifu.plugin.spark.stats.columnstates;

import java.util.ArrayList;
import java.util.List;

import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.interfaces.ColumnState;
import ml.shifu.plugin.spark.stats.interfaces.UnitState;
import ml.shifu.plugin.spark.stats.unitstates.FrequencyUnitState;
import ml.shifu.plugin.spark.stats.unitstates.HistogramUnitState;

import org.dmg.pmml.DataField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.UnivariateStats;

public class SimpleUnivariateOrdinalState extends ColumnState {
    
    public SimpleUnivariateOrdinalState(String name, Params parameters) {
        states= new ArrayList<UnitState>();
        states.add(new FrequencyUnitState());
        params= parameters;
        fieldName= name;
    }
    
    public ColumnState getNewBlank() {
        return new SimpleUnivariateOrdinalState(fieldName, params);
    }

    @Override
    public void checkClass(ColumnState colState) throws Exception {
        if(!(colState instanceof SimpleUnivariateOrdinalState))
            throw new Exception("Expected UnivariateOrdinalState in merge, got " + colState.getClass().toString());    
    }
}
