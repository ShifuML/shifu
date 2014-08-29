package ml.shifu.plugin.spark.stats.columnstates;

import java.util.ArrayList;

import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.interfaces.ColumnState;
import ml.shifu.plugin.spark.stats.interfaces.UnitState;
import ml.shifu.plugin.spark.stats.unitstates.DiscreteBinningUnitState;
import ml.shifu.plugin.spark.stats.unitstates.FrequencyUnitState;

public class BinomialDiscrState extends ColumnState {
    private static final long serialVersionUID = 1L;

    public BinomialDiscrState(String name, Params parameters) {
        states= new ArrayList<UnitState>();
        states.add(new FrequencyUnitState());
        states.add(new DiscreteBinningUnitState());
        params= parameters;
        fieldName= name;        
    }

    @Override
    public ColumnState getNewBlank() {
        return new BinomialDiscrState(fieldName, params);
    }

    @Override
    public void checkClass(ColumnState colState) throws Exception {
        if(!(colState instanceof BinomialDiscrState))
            throw new Exception("Expected BinomialDiscrState, got " + colState.getClass().toString());
    }
    
}
