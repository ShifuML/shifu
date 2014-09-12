package ml.shifu.plugin.spark.stats.factory;

import java.util.ArrayList;
import java.util.List;

import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.interfaces.ColumnState;
import ml.shifu.plugin.spark.stats.interfaces.UnitState;
import ml.shifu.plugin.spark.stats.unitstates.BasicNumericInfoUnitState;
import ml.shifu.plugin.spark.stats.unitstates.BinomialRSampleUnitState;
import ml.shifu.plugin.spark.stats.unitstates.DiscreteBinningUnitState;
import ml.shifu.plugin.spark.stats.unitstates.DoubleRSampleUnitState;
import ml.shifu.plugin.spark.stats.unitstates.FrequencyUnitState;
import ml.shifu.plugin.spark.stats.unitstates.HistogramUnitState;
import ml.shifu.plugin.spark.stats.unitstates.MockUnitState;
import ml.shifu.plugin.spark.utils.ColType;

/**
 * This class contains the logic which dictates the UnitStates columns of each type (continuous, categorical, ordinal) should have, 
 * based on the type of stats- univariate or binomial
 * @author apalnitkar
 *
 */

public class ColumnStateFactory {
	
	public static ColumnState getUnivariateColumnState(String fieldName, Params params, ColType colType) {
		List<UnitState> unitStates= new ArrayList<UnitState>();
		if(colType== ColType.CONTINUOUS) {
	        int sampleSize= Integer.parseInt(params.get("sampleSize", "10000").toString());
	        unitStates.add(new BasicNumericInfoUnitState());
	        unitStates.add(new FrequencyUnitState());
	        unitStates.add(new DoubleRSampleUnitState(sampleSize));
	        //return new UnivariateContState(fieldName, params, ColType.CONTINUOUS, unitStates);
	        return new ColumnState(fieldName, params, colType, unitStates);
		}
		else if(colType== ColType.CATEGORICAL) {
	        int maxHistogramSize= Integer.parseInt(params.get("maxHistogramSize", "10000").toString());
	        unitStates.add(new FrequencyUnitState());
	        unitStates.add(new HistogramUnitState(maxHistogramSize));
	        return new ColumnState(fieldName, params, colType, unitStates);
		}
		else {
			unitStates.add(new FrequencyUnitState());
	        return new ColumnState(fieldName, params, colType, unitStates);
		}
		
	}

	public static ColumnState getBinomialColumnState(String fieldName, Params params, ColType colType) {
		List<UnitState> unitStates= new ArrayList<UnitState>();
		
		if(colType== ColType.CONTINUOUS) {
	        int sampleSize= Integer.parseInt(params.get("sampleSize", "100000").toString());
	        unitStates.add(new BasicNumericInfoUnitState());
	        unitStates.add(new FrequencyUnitState());
	        unitStates.add(new BinomialRSampleUnitState(sampleSize));

		}
		else if(colType== ColType.CATEGORICAL) {
	        unitStates.add(new FrequencyUnitState());
	        unitStates.add(new DiscreteBinningUnitState());
		}
		else if(colType== ColType.ORDINAL) {
	        unitStates.add(new FrequencyUnitState());
		}
		
		return new ColumnState(fieldName, params, colType, unitStates);
	}
	
	public static ColumnState getMockColumnState(String fieldName, Params params, ColType colType) {
		List<UnitState> unitStates= new ArrayList<UnitState>();
		unitStates.add(new MockUnitState());
		unitStates.add(new MockUnitState());
		return new ColumnState(fieldName, params, colType, unitStates);
	}
}
