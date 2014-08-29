package ml.shifu.plugin.spark.stats.columnstates;

import java.util.ArrayList;

import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.SerializedNumericalValueObject;
import ml.shifu.plugin.spark.stats.interfaces.ColumnState;
import ml.shifu.plugin.spark.stats.interfaces.UnitState;
import ml.shifu.plugin.spark.stats.unitstates.BasicNumericInfoUnitState;
import ml.shifu.plugin.spark.stats.unitstates.BinomialRSampleUnitState;
import ml.shifu.plugin.spark.stats.unitstates.FrequencyUnitState;
/*
 * Expects data in the form of a NumericalValueObject.
 */
public class BinomialContState extends ColumnState {
    private static final long serialVersionUID = 1L;

    public BinomialContState(String name, Params parameters) {
        fieldName= name;
        params= parameters;
        int sampleSize= Integer.parseInt(params.get("sampleSize", "100000").toString());
        int numBins= Integer.parseInt(params.get("numBins", "11").toString());
        states= new ArrayList<UnitState>();
        states.add(new BasicNumericInfoUnitState());
        states.add(new FrequencyUnitState());
        states.add(new BinomialRSampleUnitState(sampleSize, numBins));
    }

    @Override
    public ColumnState getNewBlank() {
        return new BinomialContState(fieldName, params);
    }

    @Override
    public void checkClass(ColumnState colState) throws Exception {
        if(!(colState instanceof BinomialContState))
            throw new Exception("Expected BinomialContState, got " + colState.getClass().toString());
    }
    
    public void addData(Object data) {
        try {
        // send only double values to first two states
        SerializedNumericalValueObject nvo= (SerializedNumericalValueObject) data;
        Double value= nvo.getValue();
        states.get(0).addData(value);
        states.get(1).addData(value);

        // pass whole object to sampler
        states.get(2).addData(nvo);
        } catch (NullPointerException | ClassCastException e) {
            System.out.println(e.getMessage());
        }
    }
}
