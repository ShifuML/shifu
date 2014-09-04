/**
 * Copyright [2012-2014] eBay Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.shifu.core.di.builtin.stats;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.core.container.RawValueObject;

import org.apache.commons.math3.stat.inference.OneWayAnova;
import org.dmg.pmml.Anova;
import org.dmg.pmml.AnovaRow;
import org.dmg.pmml.UnivariateStats;

/**
 * Computes one way unbalanced ANOVA values
 * Currently not integrated in shifu-core
 * Requires a continuous target variable and a categorical independent variable
 *
 */

public class AnovaStatsCalculator {

	public static void calculateDiscr(UnivariateStats stats,
			List<RawValueObject> rawValues) {
		// consider every category of value to be in a separate group
		Map<Object, List<Double>> groups= getGroupsMap(rawValues);
		// create list of double arrays
		List<double[]> dList= getListOfDoubleArrays(groups);
		OneWayAnova anovaCalculator= new OneWayAnova();
		
		/*
		 * Variable names:
		 * prefix:	ss= sum of squares
		 * 			ms= mean of sum of squares
		 * 			df= degrees of freedom
		 * 
		 * suffix: 	wg= within group (corresponding to "Error" in PMML)
		 * 			bg= between group (corresponding to "Model" in PMML)
		 */
		
		double sswg= getSSWG(dList);
		double ssbg= getSSBG(dList);
		
		int dfwg= rawValues.size() - dList.size();
		int dfbg= dList.size() -1;
		
		double mswg= sswg/dfwg;
		double msbg= ssbg/dfbg;
		
		Anova anova= new Anova();
		AnovaRow modelRow= new AnovaRow();
		modelRow.setDegreesOfFreedom(dfbg);
		modelRow.setFValue(mswg/msbg);
		modelRow.setPValue(anovaCalculator.anovaPValue(dList));
		modelRow.setMeanOfSquares(msbg);
		modelRow.setType("Model");

		AnovaRow errorRow= new AnovaRow();
		errorRow.setType("Error");
		errorRow.setMeanOfSquares(mswg);
		errorRow.setDegreesOfFreedom(dfwg);
		
		anova.withAnovaRows(modelRow, errorRow);
		stats.withAnova(anova);
		
	}
	
	public static double getSSBG(List<double[]> dList) {
		double ssbg= 0, meanTotal= getMean(dList);
		for(double[] values: dList)
			ssbg+= Math.pow((getMean(values) - meanTotal), 2) * values.length;
		return ssbg;
	}
	
	public static double getSSTotal(List<double[]> dList) {
		double ssTotal= 0;
		double meanTotal= getMean(dList);
		for(double[] values: dList)
			for(double value: values)
				ssTotal += Math.pow((value-meanTotal), 2);
		return ssTotal;
	}
	
	public static double getSSWG(List<double[]> dList) {
		double sswg= 0;
		for(double[] values: dList)
			sswg+= sumOfSquares(values);
		return sswg;
	}
	
	public static Map<Object, List<Double>> getGroupsMap(List<RawValueObject> rawValues) {
		Map<Object, List<Double>> groups= new HashMap<Object, List<Double>>();
		for(RawValueObject value: rawValues) {
			double tagValue= Double.parseDouble(value.getTag());
			incMapCnt(groups, value.getValue(), tagValue);
		}
		return groups;
	}
	

	public static List<double[]> getListOfDoubleArrays(Map<Object, List<Double>> groups) {
		List<double[]> dList= new ArrayList<double[]>();
		for(Object key: groups.keySet()) {
			List<Double> list= groups.get(key);
			double[] dArray= new double[list.size()];
			for(int i= 0; i < list.size(); i++)
				dArray[i]= list.get(i);
			dList.add(dArray);			
		}
		return dList;
	}
	
	public static double getMean(List<double[]> dList) {
		double sum= 0;
		int n= 0;
		for(double[] array: dList) {
			for(double value: array)
				sum+= value;
			n+= array.length;
		}
		return sum/n;
	}
	
	public static double getMean(double[] values) {
		double sum= 0;
		for(double value: values)
			sum+= value;
		return sum/values.length;
	}
	
	public static double sumOfSquares(double[] values) {
		// returns sum((value - mean(values))^2)
		double sum= 0, n= values.length, sumSqr= 0;
		for(double value: values) {
			sum+= value;
			sumSqr+= value * value;
		}
		return sumSqr - (sum*sum)/n;
	}
	
	private static void incMapCnt(Map<Object, List<Double>> groups, Object value, Double tag) {
		if(groups.containsKey(value))
			groups.get(value).add(tag);
		else {
			groups.put(value, new ArrayList<Double>());
			groups.get(value).add(tag);
		}
	}

}
