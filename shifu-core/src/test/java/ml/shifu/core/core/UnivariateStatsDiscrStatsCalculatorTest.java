package ml.shifu.core.core;


import ml.shifu.core.container.CategoricalValueObject;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.List;

public class UnivariateStatsDiscrStatsCalculatorTest {

    @Test
    public void test() {

        List<CategoricalValueObject> voList = new ArrayList<CategoricalValueObject>();

        for (int i = 0; i < 100; i++) {
            CategoricalValueObject vo = new CategoricalValueObject();
            vo.setValue(i % 3 == 1 ? "Cat" : "Dog");
            vo.setIsPositive(i % 2 == 1 ? true : false);
            voList.add(vo);
        }

        //DiscrStats discrStats = UnivariateStatsDiscrCalculator.calculate(null, voList);

        //Assert.assertEquals(discrStats.getArrays().get(0).getValue(), "67 33");


    }
}
