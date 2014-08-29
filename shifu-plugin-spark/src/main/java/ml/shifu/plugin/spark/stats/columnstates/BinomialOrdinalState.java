package ml.shifu.plugin.spark.stats.columnstates;

import java.util.ArrayList;

import ml.shifu.core.container.CategoricalValueObject;
import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.interfaces.ColumnState;
import ml.shifu.plugin.spark.stats.interfaces.UnitState;
import ml.shifu.plugin.spark.stats.unitstates.FrequencyUnitState;

/*
 * Similar to SimpleUnivariateOrdinalState, with the exception that it recieves data in the form of CategoricalValueObjects
 * which it unpacks to consider only the value fields.
 * 
 */
public class BinomialOrdinalState extends ColumnState {

    public BinomialOrdinalState(String name, Params parameters) {
        states= new ArrayList<UnitState>();
        states.add(new FrequencyUnitState());
        params= parameters;
        fieldName= name;
    }

    @Override
    public ColumnState getNewBlank() {
        return new BinomialOrdinalState(fieldName, params);
    }

    @Override
    public void checkClass(ColumnState colState) throws Exception {
        if(!(colState instanceof BinomialOrdinalState)) 
            throw new Exception("Expected BinomialOrdinalState, got " + colState.getClass().toString());
    }
    
}
