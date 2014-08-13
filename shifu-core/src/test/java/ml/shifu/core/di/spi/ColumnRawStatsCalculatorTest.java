package ml.shifu.core.di.spi;

import ml.shifu.core.container.ColumnRawStatsResult;
import ml.shifu.core.container.RawValueObject;
import ml.shifu.core.di.builtin.DefaultColumnRawStatsCalculator;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ColumnRawStatsCalculatorTest {

    @Test
    public void testRawValueObjectScreener() {

        ColumnRawStatsCalculator screener = new DefaultColumnRawStatsCalculator();

        ColumnRawStatsResult result1 = screener.calculate(genData1(), Arrays.asList("P"), new ArrayList<String>());

        Assert.assertEquals((int) result1.getCntIsNumber(), 10);
        Assert.assertEquals((int) result1.getCntValidPositive(), 10);

        ColumnRawStatsResult result2 = screener.calculate(genData2(), Arrays.asList("P"), Arrays.asList("N"));

        Assert.assertEquals((int) result2.getCntTotal(), 10);
        Assert.assertEquals((int) result2.getCntValidPositive(), 4);
        Assert.assertEquals((int) result2.getCntValidNegative(), 3);
        Assert.assertEquals((int) result2.getCntIgnoredByTag(), 3);

        ColumnRawStatsResult result3 = screener.calculate(genData3(), Arrays.asList("P"), Arrays.asList("N"));
        System.out.println(Double.MAX_VALUE);
        Assert.assertEquals((int) result3.getCntTotal(), 6);
        Assert.assertEquals((int) result3.getCntUniqueValues(), 6);
        Assert.assertEquals((int) result3.getCntIsNumber(), 2);

    }


    private List<RawValueObject> genData1() {
        List<RawValueObject> rvoList = new ArrayList<RawValueObject>();

        for (int i = 0; i < 10; i++) {
            RawValueObject rvo = new RawValueObject();
            rvo.setValue(i);
            rvo.setTag("P");
            rvo.setWeight(1.0);
            rvoList.add(rvo);
        }
        return rvoList;
    }

    private List<RawValueObject> genData2() {
        List<RawValueObject> rvoList = new ArrayList<RawValueObject>();

        for (int i = 0; i < 10; i++) {
            RawValueObject rvo = new RawValueObject();
            rvo.setValue(i);
            rvo.setTag(i % 3 == 0 ? "P" : (i % 3 == 1 ? "N" : "NA"));
            rvo.setWeight(1.0);
            rvoList.add(rvo);
        }
        return rvoList;
    }

    private List<RawValueObject> genData3() {

        List<Object> list = Arrays.asList((Object) "NaN", 1.0, Double.MAX_VALUE, Double.NaN, "hello", null);

        List<RawValueObject> rvoList = new ArrayList<RawValueObject>();

        for (Object o : list) {
            RawValueObject rvo = new RawValueObject();
            rvo.setValue(o);
            rvo.setTag("P");
            rvo.setWeight(1.0);
            rvoList.add(rvo);
        }
        return rvoList;
    }
}
