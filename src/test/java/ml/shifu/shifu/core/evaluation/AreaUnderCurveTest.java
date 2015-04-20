package ml.shifu.shifu.core.evaluation;

import java.util.ArrayList;
import java.util.List;

import ml.shifu.shifu.container.PerformanceObject;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class AreaUnderCurveTest {
  
    private List<PerformanceObject> roc = new ArrayList<PerformanceObject>();
    private List<PerformanceObject> pr = new ArrayList<PerformanceObject>();
    
    @BeforeClass
    public void setUp() {
        double[][] rocPoints = new double[][] {
                {0.0, 0.0}, {0.5, 0.6}, {1.0, 1.0}
        };
        
        double[][] rocWeightedPoints = new double[][] {
                {0.0, 0.0}, {0.4, 0.6}, {1.0, 1.0}
        };
        
        double[][] prPoints = new double[][] {
                {0.0, 1.0}, {0.5, 0.6}, {1.0, 0.0}
        };
        
        double[][] prWeightedPoints = new double[][] {
                {0.0, 1.0}, {0.6, 0.4}, {1.0, 0.0}
        };
        
        for(int i = 0; i < rocPoints.length; i++) {
            PerformanceObject obj = new PerformanceObject();
            obj.fpr = rocPoints[i][0];
            obj.recall = rocPoints[i][1];
            obj.weightedFpr = rocWeightedPoints[i][0];
            obj.weightedRecall = rocWeightedPoints[i][1];
            roc.add(obj);
        }
        
        for(int i = 0; i < prPoints.length; i++) {
            PerformanceObject obj = new PerformanceObject();
            obj.recall = prPoints[i][0];
            obj.precision = prPoints[i][1];
            obj.weightedRecall = prWeightedPoints[i][0];
            obj.weightedPrecision = prWeightedPoints[i][1];
            pr.add(obj);
        }
    }
    
    
    @Test
    public void trapezoidTest() {
        double[] first = new double[]{1,1};
        double[] second = new double[]{3,2};
        double area = AreaUnderCurve.trapezoid(first, second);
        Assert.assertEquals(area, 3.0);
    }
    
    @Test void ofRocTest() {
        double area = AreaUnderCurve.ofRoc(roc);
        double weightedArea = AreaUnderCurve.ofWeightedRoc(roc);
        Assert.assertEquals(area, 0.55);
        Assert.assertEquals(weightedArea, 0.6);
    }
    
    @Test void ofPrTest() {
        double area = AreaUnderCurve.ofPr(pr);
        double weightedArea = AreaUnderCurve.ofWeightedPr(pr);
        Assert.assertEquals(area, 0.55);
        Assert.assertEquals(weightedArea, 0.5);
    }
}
