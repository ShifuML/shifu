package ml.shifu.shifu.di.spi;


import ml.shifu.shifu.container.CategoricalValueObject;
import ml.shifu.shifu.container.obj.ColumnBinningResult;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.di.builtin.DefaultColumnCatBinningCalculator;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ColumnCatBinningCalculatorTest {

    @Test
    public void testSimpleCategoricalBinning() {
        List<CategoricalValueObject> voList = new ArrayList<CategoricalValueObject>();
        for (int i = 0; i < 100; i++) {
            CategoricalValueObject vo = new CategoricalValueObject();
            vo.setValue(i % 3 == 1 ? "Foo" : "Bar");
            vo.setIsPositive(true);
            vo.setWeight(1.0);
            voList.add(vo);
        }

        ModelConfig modelConfig = generateModelConfig();

        ColumnCatBinningCalculator binning = new DefaultColumnCatBinningCalculator();
        ColumnBinningResult columnBinningResult = binning.calculate(voList);

        System.out.println(columnBinningResult.getBinCategory());
        System.out.println(columnBinningResult.getBinCountNeg());
        System.out.println(columnBinningResult.getBinCountPos());
        System.out.println(columnBinningResult.getBinPosRate());


    }

    private ModelConfig generateModelConfig() {
        ModelConfig modelConfig = new ModelConfig();
        modelConfig.getDataSet().setPosTags(Arrays.asList("P"));
        modelConfig.getDataSet().setNegTags(Arrays.asList("N"));
        modelConfig.getStats().setMaxNumBin(10);
        return modelConfig;
    }

}
