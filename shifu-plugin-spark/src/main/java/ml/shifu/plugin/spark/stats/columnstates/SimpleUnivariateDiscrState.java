package ml.shifu.plugin.spark.stats.columnstates;

import java.util.ArrayList;

import org.dmg.pmml.DataField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.UnivariateStats;

import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.interfaces.ColumnState;
import ml.shifu.plugin.spark.stats.interfaces.UnitState;
import ml.shifu.plugin.spark.stats.unitstates.FrequencyUnitState;
import ml.shifu.plugin.spark.stats.unitstates.HistogramUnitState;

public class SimpleUnivariateDiscrState extends ColumnState {
    
    public SimpleUnivariateDiscrState(String name, Params parameters) {
        params= parameters;
        int maxHistogramSize= Integer.parseInt(params.get("maxHistogramSize", "10000").toString());
        states= new ArrayList<UnitState>();
        states.add(new FrequencyUnitState());
        states.add(new HistogramUnitState(maxHistogramSize));
        fieldName= name;
    }
    
    public ColumnState getNewBlank() {
        return new SimpleUnivariateDiscrState(this.fieldName, this.params);
    }

    @Override
    public void checkClass(ColumnState colState) throws Exception {
        if(!(colState instanceof SimpleUnivariateDiscrState))
            throw new Exception("Expected UnivariateDiscrState in merge, got " + colState.getClass().toString());
    }

}
