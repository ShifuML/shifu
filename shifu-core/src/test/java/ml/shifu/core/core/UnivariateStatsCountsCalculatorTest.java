package ml.shifu.core.core;

import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.List;

public class UnivariateStatsCountsCalculatorTest {

    @Test
    public void test() {

        List<Object> values = Arrays.asList((Object) "NaN", null, "hello", 1.0, Double.NaN);

        //Counts counts = UnivariateStatsCountsCalculator.calculate(values);

        //Assert.assertEquals((int)counts.getCardinality(), 5);
        //Assert.assertEquals(counts.getMissingFreq(), 1.0);
        //Assert.assertEquals(counts.getInvalidFreq(), 2.0);
        //Assert.assertEquals(counts.getTotalFreq(), 5.0);


    }
}
