package ml.shifu.core.di.spi;

import ml.shifu.core.container.ColumnBinningResult;
import ml.shifu.core.container.NumericalValueObject;
import ml.shifu.core.di.builtin.EqualPositiveColumnNumBinningCalculator;
import ml.shifu.core.di.builtin.TotalPercentileColumnNumBinningCalculator;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.List;


public class ColumnNumBinningCalculatorTest {


    @Test
    public void testEqualPositiveNumericalBinning() {

        List<NumericalValueObject> voList = generateData();

        ColumnNumBinningCalculator binning = new EqualPositiveColumnNumBinningCalculator();
        ColumnBinningResult columnBinningResult = binning.calculate(voList, 10);


        System.out.println(columnBinningResult.getBinBoundary());
        System.out.println(columnBinningResult.getBinCountPos());
        System.out.println(columnBinningResult.getBinCountNeg());
        //Assert.assertNull(columnBinningResult);
    }

    @Test
    public void testTotalPercentileNumericalBinning() {
        List<NumericalValueObject> voList = generateData();

        ColumnNumBinningCalculator binning = new TotalPercentileColumnNumBinningCalculator();
        ColumnBinningResult columnBinningResult = binning.calculate(voList, 10);

    }


    private List<NumericalValueObject> generateData() {
        List<NumericalValueObject> voList = new ArrayList<NumericalValueObject>();
        for (int i = 0; i < 100; i++) {
            NumericalValueObject vo = new NumericalValueObject();
            vo.setValue((double) (i % 20));
            vo.setIsPositive(i % 3 == 0);
            vo.setWeight(1.0);
            voList.add(vo);
        }
        return voList;
    }

    private List<NumericalValueObject> generateData2() {
        List<NumericalValueObject> voList = new ArrayList<NumericalValueObject>();
        for (int i = 0; i < 100; i++) {
            NumericalValueObject vo = new NumericalValueObject();
            vo.setValue(1.0);
            vo.setIsPositive(i % 3 == 0);
            vo.setWeight(1.0);
            voList.add(vo);
        }
        return voList;
    }
}
