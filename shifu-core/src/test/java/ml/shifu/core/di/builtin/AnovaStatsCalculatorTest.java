package ml.shifu.core.di.builtin;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ml.shifu.core.container.RawValueObject;
import ml.shifu.core.di.builtin.stats.AnovaStatsCalculator;
import ml.shifu.core.di.spi.SingleThreadFileLoader;
import ml.shifu.core.util.CSVWithHeaderLocalSingleThreadFileLoader;
import ml.shifu.core.util.LocalDataTransposer;

import org.apache.commons.math3.stat.inference.OneWayAnova;
import org.dmg.pmml.Anova;
import org.dmg.pmml.AnovaRow;
import org.dmg.pmml.UnivariateStats;
import org.testng.Assert;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class AnovaStatsCalculatorTest {
	double[] d1= {3.0, 3.0, 2.0, 3.0, 4.0};
	double[] d2= {5.0, 4.0, 3.0, 3.0, 5.0};
	List<double[]> dList;
	
	@BeforeTest
	public void AnovaTest() {
		OneWayAnova anova= new OneWayAnova();
		dList= new ArrayList<double[]>();
		dList.add(d1);
		dList.add(d2);
	}
	
	@Test
	public void testSumOfSquares() {
		Assert.assertTrue(isWithin(AnovaStatsCalculator.sumOfSquares(d1), 2.0, 0.01));
		Assert.assertTrue(isWithin(AnovaStatsCalculator.sumOfSquares(d2), 4.0, 0.01));
	}
	
	@Test
	public void testSS() {
		double sswg= AnovaStatsCalculator.getSSWG(dList);
		double ssbg= AnovaStatsCalculator.getSSBG(dList);
		double ssTotal= AnovaStatsCalculator.getSSTotal(dList);
		Assert.assertTrue(isWithin(ssTotal, sswg + ssbg, 0.05));		
	}
	
	public boolean isWithin(double actual, double computed, double ratio) {
		if(computed <= actual*(1 + ratio) && computed >= actual * (1-ratio))
			return true;
		else
			return false;
	}
	
	@Test
	public void testPMML() {
		UnivariateStats stats= new UnivariateStats();
	}

	@Test
	public void AnovaCalculatorTest() throws IOException {
		UnivariateStats stats= new UnivariateStats();
        SingleThreadFileLoader loader = new CSVWithHeaderLocalSingleThreadFileLoader();
        String pathInputData= "./src/test/resources/example/wdbc/data/wdbc.data";
        List<List<Object>> rows = loader.load(pathInputData);
        List<List<Object>> columns = LocalDataTransposer.transpose(rows);
        List<Object> depVar= columns.get(1);
        List<Object> indVar= columns.get(2);
        // consider mean_radius to be the dependent variable and diagnosis to be the independent variable
        // create list of RawValueObjects
        List<RawValueObject> rvoList= new ArrayList<RawValueObject>();
        for(int i=0; i < depVar.size(); i++) {
        	RawValueObject rvo= new RawValueObject();
        	rvo.setTag(indVar.get(i).toString());
        	rvo.setValue(depVar.get(i));
        	rvoList.add(rvo);
        }
        List<double[]> dList= AnovaStatsCalculator.getListOfDoubleArrays(AnovaStatsCalculator.getGroupsMap(rvoList));
        AnovaStatsCalculator.calculateDiscr(stats, rvoList);
        OneWayAnova anovaCalculator= new OneWayAnova();
        
        Anova anova= stats.getAnova();
        List<AnovaRow> anovaRows= anova.getAnovaRows();
        
        for(AnovaRow row: anovaRows) {
        	if(row.getType().equals("Model")) {
        		Assert.assertTrue(isWithin(row.getPValue(), anovaCalculator.anovaPValue(dList), 0.05));
        	}
        }
        
		
	}
	
}
