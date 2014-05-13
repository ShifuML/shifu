package ml.shifu.shifu.di.module;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.shifu.container.RawValueObject;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.di.builtin.*;
import ml.shifu.shifu.di.service.StatsService;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.*;

public class StatsModuleTest {

    @Test
    public void testMockStatsProcessor() {

        StatsModule statsModule = new StatsModule();
        statsModule.setStatsProcessorImplClass(MockStatsProcessor.class);

        Injector injector = Guice.createInjector(statsModule);
        StatsService statsService = injector.getInstance(StatsService.class);

        ColumnConfig columnConfig = new ColumnConfig();
        columnConfig.setColumnType(ColumnConfig.ColumnType.N);
        statsService.exec(columnConfig, genData1());

        Assert.assertEquals(columnConfig.getColumnBinningResult().getBinBoundary(), Arrays.asList(Double.NEGATIVE_INFINITY, 1.0, 2.0, 10.0, 100.0));

    }

    public void testMock() {
        /*statsModule.setNumBinningCalculatorImplClass(TotalPercentileColumnNumBinningCalculator.class);
        statsModule.setCatBinningCalculatorImplClass(SimpleColumnCatBinningCalculator.class);
        statsModule.setNumStatsCalculatorImplClass(DefaultColumnNumStatsCalculator.class);
        statsModule.setBinStatsCalculatorImplClass(DefaultColumnBinStatsCalculator.class);
        statsModule.setRawStatsCalculatorImplClass(MockColumnRawStatsCalculator.class);
         */


        Map<String, Object> params = new HashMap<String, Object>();
      /*
        params.put("posTags", Arrays.asList("P"));
        params.put("negTags", Arrays.asList("N"));
        params.put("numBins", 4);

        */


        //statsService.setParams(params);



        //System.out.println(config.getBinBoundary());
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

}
