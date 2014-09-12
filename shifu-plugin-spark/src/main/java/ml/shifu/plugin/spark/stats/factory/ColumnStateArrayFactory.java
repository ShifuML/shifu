package ml.shifu.plugin.spark.stats.factory;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.dmg.pmml.DataField;

import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.BinomialColumnStateArray;
import ml.shifu.plugin.spark.stats.UnivariateColumnStateArray;
import ml.shifu.plugin.spark.stats.interfaces.ColumnState;
import ml.shifu.plugin.spark.stats.interfaces.ColumnStateArray;
import ml.shifu.plugin.spark.utils.ColType;

public class ColumnStateArrayFactory {
	
	public static ColumnStateArray getUnivariateColumnStates(List<DataField> dataFields, Params params) {
		List<ColumnState> colStates= new ArrayList<ColumnState>();
		
		for(DataField field: dataFields) {
			ColType colType= ColType.convert(field.getOptype());
			colStates.add(ColumnStateFactory.getUnivariateColumnState(field.getName().getValue(), params, colType));
        }
		String delimiter= params.get("delimiter", ",").toString();
		return new UnivariateColumnStateArray(delimiter, colStates);
	}

	public static ColumnStateArray getBinomialColumnStates(List<DataField> dataFields, Params params) {
		
		List<ColumnState> colStates= new ArrayList<ColumnState>();
		
		for(DataField field: dataFields) {
			ColType colType= ColType.convert(field.getOptype());
			colStates.add(ColumnStateFactory.getBinomialColumnState(field.getName().getValue(), params, colType));

		}   
		String delimiter= params.get("delimiter", ",").toString();
        Set<String> posTags= new HashSet<String>((List<String>) params.get("posTags"));
        Set<String >negTags= new HashSet<String>((List<String>) params.get("negTags"));
        int targetFieldNum= Integer.parseInt(params.get("targetFieldNum").toString());

		return new BinomialColumnStateArray(delimiter, posTags, negTags, targetFieldNum, colStates);
	}

}
